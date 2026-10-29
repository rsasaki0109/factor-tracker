//! Camera + LiDAR commands: fixture, prepare, replay, evaluate, report, demo.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use factor_tracker_engine::engine::{EngineConfig, TrackingMode};
use factor_tracker_engine::eval::{
    evaluate_mode, CameraAssociationTruth, ComparisonReport, EvalConfig, GroundTruthFrame,
};
use factor_tracker_engine::geometry::Calibration;
use factor_tracker_engine::inject::{self, InjectionConfig, SensorKind};
use factor_tracker_engine::io::{
    read_inputs, read_truth, write_jsonl, write_outputs, EngineInputRecord, EngineOutputRecord,
};
use factor_tracker_engine::kitti::{prepare_sequence, KittiSequencePaths};
use factor_tracker_engine::setup::EngineSetup;
use factor_tracker_engine::{
    fixture, replay, AllCounters, CameraAssistedLidarEngine, FgoConfig, FixedLagRefiner,
    FixedLagSmoother,
};

use crate::jsonl::CliResult;
use crate::report;

/// Options for a replay.
pub struct ReplayOptions {
    /// Tracking mode.
    pub mode: TrackingMode,
    /// Injection configuration.
    pub injection: InjectionConfig,
    /// Optional overrides of the 3D tracker defaults.
    pub tracker: TrackerOverrides,
}

impl Default for ReplayOptions {
    fn default() -> Self {
        ReplayOptions {
            mode: TrackingMode::CameraAssociated,
            injection: InjectionConfig::default(),
            tracker: TrackerOverrides::default(),
        }
    }
}

/// Optional overrides of [`factor_tracker_core::Cartesian3DTrackerConfig`] fields.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrackerOverrides {
    /// Initial velocity variance (m^2/s^2).
    pub initial_velocity_variance: Option<f64>,
    /// Acceleration spectral density (m^2/s^3).
    pub accel_spectral_density: Option<f64>,
    /// Squared-Mahalanobis association gate.
    pub gate_squared_mahalanobis: Option<f64>,
    /// Horizontal-distance association gate (m); 0 disables.
    pub max_association_distance_m: Option<f64>,
    /// Matching-cascade window in nanoseconds; `Some(0)` disables the cascade.
    pub recent_track_window_ns: Option<i64>,
    /// Appearance cost weight (0 disables appearance).
    pub appearance_weight: Option<f64>,
    /// Hard gate on the appearance cosine distance.
    pub appearance_max_distance: Option<f64>,
    /// Minimum distance (m) from a confirmed track for a new track; 0 disables.
    pub min_birth_distance_m: Option<f64>,
    /// Sightings after which a suppressed birth starts a track anyway.
    pub birth_persistence: Option<u32>,
    /// Fixed-lag re-association window in LiDAR scans; 0 disables.
    pub fgo_lag: usize,
    /// Refiner birth cost.
    pub fgo_birth_cost: Option<f64>,
    /// Refiner miss cost.
    pub fgo_miss_cost: Option<f64>,
    /// Refiner clutter cost.
    pub fgo_clutter_cost: Option<f64>,
    /// Refiner detection position sigma (m).
    pub fgo_sigma: Option<f64>,
    /// Refiner comparison radius (m).
    pub fgo_radius: Option<f64>,
    /// Refiner coasting report time (ms).
    pub fgo_coast_ms: Option<i64>,
    /// Observations before confirmation.
    pub confirm_min_observations: Option<u32>,
    /// Confirmed-to-lost time (ms).
    pub confirmed_to_lost_ms: Option<i64>,
    /// Lost-to-delete time (ms).
    pub lost_to_delete_ms: Option<i64>,
    /// Report only tracks observed within this long (ms).
    pub report_coast_ms: Option<i64>,
    /// Weight of the newest detection in a track's confidence.
    pub confidence_alpha: Option<f64>,
    /// Report only tracks with at least this confidence.
    pub report_min_confidence: Option<f64>,
    /// Standard deviation (m/s) of the detections' measured velocities.
    pub velocity_std: Option<f64>,
    /// Minimum detection score to start a track.
    pub new_track_score_threshold: Option<f64>,
    /// Correct each scan's ego pose with its static landmarks.
    pub pose_correction: bool,
    /// Fixed-lag smoothing window in LiDAR scans; 0 disables.
    pub smooth_lag: usize,
    /// With `smooth_lag` in `stream`: also write the real-time frames.
    pub smooth_realtime: bool,
}

impl TrackerOverrides {
    /// Drops reported tracks not observed within `report_coast_ms` or below
    /// `report_min_confidence`.
    fn filter_reported(&self, frames: &mut [factor_tracker_engine::LidarFrameOutput]) {
        if let Some(ms) = self.report_coast_ms {
            for frame in frames.iter_mut() {
                frame
                    .tracks
                    .retain(|t| t.age_since_spatial_update_ns <= ms * 1_000_000);
            }
        }
        if let Some(min) = self.report_min_confidence {
            for frame in frames.iter_mut() {
                frame
                    .tracks
                    .retain(|t| t.confidence.is_none_or(|c| c >= min));
            }
        }
    }

    /// The smoother, when `smooth_lag` > 0, with the tracker's motion model.
    fn smoother(&self, config: &EngineConfig) -> CliResult<Option<FixedLagSmoother>> {
        if self.smooth_lag == 0 {
            return Ok(None);
        }
        if self.fgo_lag > 0 {
            return Err("--smooth-lag and --fgo-lag cannot be combined (the refiner re-estimates the states)".into());
        }
        Ok(Some(FixedLagSmoother::new(
            self.smooth_lag,
            config.tracker.accel_spectral_density,
        )))
    }

    /// The refiner, when `fgo_lag` > 0.
    fn refiner(&self) -> Option<FixedLagRefiner> {
        if self.fgo_lag == 0 {
            return None;
        }
        let d = FgoConfig::default();
        Some(FixedLagRefiner::new(FgoConfig {
            lag_frames: self.fgo_lag,
            birth_cost: self.fgo_birth_cost.unwrap_or(d.birth_cost),
            miss_cost: self.fgo_miss_cost.unwrap_or(d.miss_cost),
            clutter_cost: self.fgo_clutter_cost.unwrap_or(d.clutter_cost),
            position_sigma_m: self.fgo_sigma.unwrap_or(d.position_sigma_m),
            radius_m: self.fgo_radius.unwrap_or(d.radius_m),
            coast_ns: self.fgo_coast_ms.map_or(d.coast_ns, |ms| ms * 1_000_000),
            ..d
        }))
    }
}

/// Parses a mode string.
pub fn parse_mode(value: &str) -> CliResult<TrackingMode> {
    match value {
        "lidar_only" | "lidar-only" | "a" => Ok(TrackingMode::LidarOnly),
        "camera_associated" | "camera-associated" | "b" => Ok(TrackingMode::CameraAssociated),
        "camera_assisted" | "camera-assisted" | "c" => Ok(TrackingMode::CameraAssisted),
        other => Err(format!(
            "unknown mode '{other}' (expected lidar_only, camera_associated or camera_assisted)"
        )
        .into()),
    }
}

fn load_setup(path: &Path) -> CliResult<(Calibration, Option<factor_tracker_engine::PoseTrack>)> {
    let text = std::fs::read_to_string(path)?;
    let setup: EngineSetup = serde_json::from_str(&text)?;
    let calibration = setup.to_calibration()?;
    let poses = setup.to_poses()?;
    Ok((calibration, poses))
}

/// Writes the synthetic camera+LiDAR fixture.
pub fn run_fixture(out_dir: &Path, seed: u64) -> CliResult<()> {
    let paths = fixture::write(out_dir, seed)?;
    println!(
        "wrote synthetic camera+LiDAR fixture (seed {seed}):\n  setup  : {}\n  inputs : {}\n  truth  : {}",
        paths.setup.display(),
        paths.inputs.display(),
        paths.truth.display()
    );
    println!("NOTE: this is synthetic data, not real sensor data.");
    Ok(())
}

/// Converts a KITTI sequence directory into engine inputs and truth.
#[allow(clippy::too_many_arguments)]
pub fn run_prepare(
    calib: &Path,
    times: &Path,
    poses: Option<&Path>,
    lidar_detections: &Path,
    camera_detections: &Path,
    lidar_labels: Option<&Path>,
    camera_labels: Option<&Path>,
    image_width: u32,
    image_height: u32,
    near_plane: f64,
    default_covariance: f64,
    pose_max_gap_ns: i64,
    out_setup: &Path,
    out_inputs: &Path,
    out_truth: &Path,
) -> CliResult<()> {
    let _ = camera_labels;
    let paths = KittiSequencePaths {
        calib: calib.to_path_buf(),
        times: times.to_path_buf(),
        poses: poses.map(Path::to_path_buf),
        lidar_detections: lidar_detections.to_path_buf(),
        camera_detections: camera_detections.to_path_buf(),
        lidar_labels: lidar_labels.map(Path::to_path_buf),
        camera_labels: None,
        lidar_sensor_id: 1,
        camera_sensor_id: 2,
    };
    let prepared = prepare_sequence(
        &paths,
        image_width,
        image_height,
        near_plane,
        default_covariance,
        pose_max_gap_ns,
    )?;
    let setup = EngineSetup::from_calibration(
        &prepared.calibration,
        prepared.poses.as_ref(),
        pose_max_gap_ns,
    )?;
    std::fs::write(out_setup, serde_json::to_string_pretty(&setup)?)?;
    write_jsonl(out_inputs, &prepared.inputs)?;
    write_jsonl(out_truth, &prepared.truth)?;
    println!(
        "prepared {} input records and {} truth records",
        prepared.inputs.len(),
        prepared.truth.len()
    );
    println!("  setup  : {}", out_setup.display());
    println!("  inputs : {}", out_inputs.display());
    println!("  truth  : {}", out_truth.display());
    Ok(())
}

/// The data produced by a replay.
pub struct ReplayResult {
    /// LiDAR frame outputs.
    pub lidar_frames: Vec<factor_tracker_engine::LidarFrameOutput>,
    /// Camera frame outputs.
    pub camera_frames: Vec<factor_tracker_engine::CameraFrameOutput>,
    /// Counters.
    pub counters: AllCounters,
    /// Diagnostic events.
    pub diagnostics: Vec<factor_tracker_engine::DiagnosticEvent>,
}

/// Builds the engine configuration for a mode, the calibration's tracking
/// frame and the tracker overrides.
fn engine_config(
    mode: TrackingMode,
    calibration: &Calibration,
    overrides: TrackerOverrides,
) -> EngineConfig {
    let mut config = EngineConfig {
        mode,
        ..EngineConfig::default()
    };
    config.tracker.frame_id = calibration.tracking_frame.clone();
    if let Some(value) = overrides.initial_velocity_variance {
        config.tracker.initial_velocity_variance = value;
    }
    if let Some(value) = overrides.accel_spectral_density {
        config.tracker.accel_spectral_density = value;
    }
    if let Some(value) = overrides.gate_squared_mahalanobis {
        config.tracker.gate_squared_mahalanobis = value;
    }
    if let Some(value) = overrides.max_association_distance_m {
        config.tracker.max_association_distance_m = (value > 0.0).then_some(value);
    }
    if let Some(value) = overrides.recent_track_window_ns {
        config.tracker.recent_track_window_ns = (value > 0).then_some(value);
    }
    if let Some(value) = overrides.appearance_weight {
        config.tracker.appearance_weight = value;
    }
    if let Some(value) = overrides.appearance_max_distance {
        config.tracker.appearance_max_distance = Some(value);
    }
    if let Some(value) = overrides.min_birth_distance_m {
        config.tracker.min_birth_distance_m = (value > 0.0).then_some(value);
    }
    if let Some(value) = overrides.birth_persistence {
        config.tracker.birth_persistence = Some(value);
    }
    if let Some(value) = overrides.confirm_min_observations {
        config.tracker.common.confirm_min_observation_times = value;
    }
    if let Some(value) = overrides.confirmed_to_lost_ms {
        config.tracker.common.confirmed_to_lost_ns = value * 1_000_000;
    }
    if let Some(value) = overrides.new_track_score_threshold {
        config.tracker.common.new_track_score_threshold = Some(value);
    }
    if overrides.pose_correction {
        config.pose_correction = Some(factor_tracker_engine::LandmarkConfig::default());
    }
    if let Some(value) = overrides.confidence_alpha {
        config.confidence_alpha = value;
    }
    if let Some(value) = overrides.velocity_std {
        config.tracker.velocity_measurement_std = Some(value);
    }
    if let Some(value) = overrides.lost_to_delete_ms {
        config.tracker.common.lost_to_delete_ns = value * 1_000_000;
    }
    config
}

/// Runs a replay and returns the results.
pub fn replay_setup(
    setup_path: &Path,
    inputs_path: &Path,
    options: &ReplayOptions,
) -> CliResult<ReplayResult> {
    let (calibration, poses) = load_setup(setup_path)?;
    let inputs = read_inputs(inputs_path)?;
    let injected = inject::apply(&inputs, &options.injection);
    let config = engine_config(options.mode, &calibration, options.tracker);
    let mut smoother = options.tracker.smoother(&config)?;
    let engine = replay::run(calibration, poses, config, &injected)?;
    let mut lidar_frames = engine.lidar_frames().to_vec();
    if let Some(smoother) = smoother.as_mut() {
        let mut smoothed: Vec<_> = lidar_frames
            .into_iter()
            .flat_map(|frame| smoother.push(frame))
            .collect();
        smoothed.extend(smoother.finish());
        lidar_frames = smoothed;
    }
    if let Some(mut refiner) = options.tracker.refiner() {
        let mut refined = Vec::with_capacity(lidar_frames.len());
        for frame in lidar_frames {
            refined.extend(refiner.push(frame));
        }
        refined.extend(refiner.finish());
        lidar_frames = refined;
    }
    options.tracker.filter_reported(&mut lidar_frames);
    Ok(ReplayResult {
        lidar_frames,
        camera_frames: engine.camera_frames().to_vec(),
        counters: engine.counters(),
        diagnostics: engine.log().events().to_vec(),
    })
}

/// Runs a replay, writing outputs and a diagnostics JSON file.
pub fn run_replay(
    setup: &Path,
    inputs: &Path,
    mode: &str,
    output: &Path,
    diagnostics: &Path,
    injection: InjectionConfig,
    tracker: TrackerOverrides,
) -> CliResult<()> {
    let options = ReplayOptions {
        mode: parse_mode(mode)?,
        injection,
        tracker,
    };
    let result = replay_setup(setup, inputs, &options)?;
    write_outputs(output, &result.lidar_frames, &result.camera_frames)?;
    write_diagnostics(diagnostics, &result)?;
    println!(
        "replayed mode={} : {} lidar frames, {} camera frames -> {}",
        options.mode.as_str(),
        result.lidar_frames.len(),
        result.camera_frames.len(),
        output.display()
    );
    Ok(())
}

/// Runs the engine live: input records (one JSON object per line) are read
/// from `input`, and every output frame is written to `output` as soon as it
/// is produced, one JSON object per line. Malformed lines are reported on
/// stderr and skipped. At end of input the reorder buffer is flushed.
///
/// `reorder_window_ns` is how long (in event time) a batch waits for
/// out-of-order neighbours before it is processed; it is the output latency.
/// A single in-order LiDAR stream needs none.
pub fn run_stream(
    setup: &Path,
    mode: &str,
    reorder_window_ns: i64,
    tracker: TrackerOverrides,
    input: impl BufRead,
    mut output: impl Write,
) -> CliResult<()> {
    let (calibration, poses) = load_setup(setup)?;
    let mut config = engine_config(parse_mode(mode)?, &calibration, tracker);
    config.reorder.window_ns = reorder_window_ns;
    let mut smoother = tracker.smoother(&config)?;
    let mut engine = CameraAssistedLidarEngine::new(config, calibration, poses)?;
    let mut refiner = tracker.refiner();
    let report = tracker;
    for (number, line) in input.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed = serde_json::from_str::<EngineInputRecord>(&line)
            .map_err(|error| error.to_string())
            .and_then(|record| record.into_batch().map_err(|error| error.to_string()));
        let (batch, arrival) = match parsed {
            Ok(value) => value,
            Err(error) => {
                eprintln!("stream: line {}: skipped: {error}", number + 1);
                continue;
            }
        };
        engine.ingest(batch, arrival);
        engine.poll(arrival)?;
        write_stream_frames(
            &mut engine,
            refiner.as_mut(),
            smoother.as_mut(),
            &report,
            false,
            &mut output,
        )?;
    }
    engine.flush()?;
    write_stream_frames(
        &mut engine,
        refiner.as_mut(),
        smoother.as_mut(),
        &report,
        true,
        &mut output,
    )?;
    Ok(())
}

/// Refines an existing output file's LiDAR frames (`--fgo-lag`, default 10).
pub fn run_refine(input: &Path, output: &Path, mut tracker: TrackerOverrides) -> CliResult<()> {
    if tracker.fgo_lag == 0 {
        tracker.fgo_lag = 10;
    }
    let mut refiner = tracker.refiner().expect("fgo_lag > 0");
    let (lidar, camera) = read_output_frames(input)?;
    let mut refined = Vec::with_capacity(lidar.len());
    for frame in lidar {
        refined.extend(refiner.push(frame));
    }
    refined.extend(refiner.finish());
    tracker.filter_reported(&mut refined);
    write_outputs(output, &refined, &camera)?;
    let c = refiner.counters();
    println!(
        "refined {} frames: {} swaps, {} merges, {} splits, {} detections to clutter -> {}",
        refined.len(),
        c.swaps,
        c.merges,
        c.splits,
        c.drops,
        output.display()
    );
    Ok(())
}

fn write_stream_frames(
    engine: &mut CameraAssistedLidarEngine,
    refiner: Option<&mut FixedLagRefiner>,
    smoother: Option<&mut FixedLagSmoother>,
    report: &TrackerOverrides,
    finish: bool,
    output: &mut impl Write,
) -> CliResult<()> {
    let mut lidar = engine.take_lidar_frames();
    if let Some(smoother) = smoother {
        let realtime = if report.smooth_realtime {
            lidar.clone()
        } else {
            Vec::new()
        };
        let mut smoothed: Vec<_> = lidar
            .into_iter()
            .flat_map(|frame| smoother.push(frame))
            .collect();
        if finish {
            smoothed.extend(smoother.finish());
        }
        lidar = realtime.into_iter().chain(smoothed).collect();
    }
    if let Some(refiner) = refiner {
        let mut refined: Vec<_> = lidar
            .into_iter()
            .flat_map(|frame| refiner.push(frame))
            .collect();
        if finish {
            refined.extend(refiner.finish());
        }
        lidar = refined;
    }
    report.filter_reported(&mut lidar);
    let mut records: Vec<EngineOutputRecord> = lidar
        .into_iter()
        .map(EngineOutputRecord::Lidar)
        .chain(
            engine
                .take_camera_frames()
                .into_iter()
                .map(EngineOutputRecord::Camera),
        )
        .collect();
    if records.is_empty() {
        return Ok(());
    }
    records.sort_by_key(EngineOutputRecord::timestamp_ns);
    for record in &records {
        serde_json::to_writer(&mut *output, record)?;
        output.write_all(b"\n")?;
    }
    output.flush()?;
    Ok(())
}

fn write_diagnostics(path: &Path, result: &ReplayResult) -> CliResult<()> {
    let events: Vec<_> = result
        .diagnostics
        .iter()
        .map(|event| {
            serde_json::json!({
                "arrival_ns": event.arrival_time.as_nanos(),
                "event_ns": event.event_time.map(|t| t.as_nanos()),
                "sensor_id": event.sensor_id.map(|s| s.value()),
                "code": event.code,
                "level": format!("{:?}", event.level),
                "message": event.message,
            })
        })
        .collect();
    let value = serde_json::json!({
        "counters": result.counters,
        "events": events,
    });
    std::fs::write(path, serde_json::to_string_pretty(&value)?)?;
    Ok(())
}

/// Reads an engine output file into lidar/camera frames.
pub fn read_output_frames(
    path: &Path,
) -> CliResult<(
    Vec<factor_tracker_engine::LidarFrameOutput>,
    Vec<factor_tracker_engine::CameraFrameOutput>,
)> {
    let records: Vec<EngineOutputRecord> = factor_tracker_engine::io::read_jsonl(path)?;
    let mut lidar = Vec::new();
    let mut camera = Vec::new();
    for record in records {
        match record {
            EngineOutputRecord::Lidar(frame) => lidar.push(frame),
            EngineOutputRecord::Camera(frame) => camera.push(frame),
        }
    }
    Ok((lidar, camera))
}

/// Evaluates a replay output file against truth and writes metrics JSON.
pub fn run_evaluate(
    output: &Path,
    truth: &Path,
    metrics_out: &Path,
    config: &EvalConfig,
    zone: Option<(&factor_tracker_engine::eval::EvalZone, Option<&Path>)>,
) -> CliResult<()> {
    let (lidar_frames, camera_frames) = read_output_frames(output)?;
    let (mut truth_frames, camera_truth): (Vec<GroundTruthFrame>, Vec<CameraAssociationTruth>) =
        read_truth(truth)?;
    if let Some((zone, setup)) = zone {
        let poses = match setup {
            Some(setup) => load_setup(setup)?.1,
            None => None,
        };
        truth_frames =
            factor_tracker_engine::eval::filter_truth_to_zone(&truth_frames, poses.as_ref(), zone);
    }
    let metrics = evaluate_mode(
        &lidar_frames,
        &camera_frames,
        &truth_frames,
        &camera_truth,
        AllCounters::default(),
        config,
    );
    let report = ComparisonReport {
        modes: vec![metrics],
    };
    std::fs::write(metrics_out, serde_json::to_string_pretty(&report)?)?;
    println!("wrote metrics to {}", metrics_out.display());
    Ok(())
}

/// Renders an HTML report for one mode's frames plus optional comparison metrics.
#[allow(clippy::too_many_arguments)]
pub fn run_report(
    setup: &Path,
    output: &Path,
    truth: &Path,
    metrics: Option<&Path>,
    html: &Path,
    title: &str,
    diagnostics: Option<&Path>,
) -> CliResult<()> {
    let (calibration, _poses) = load_setup(setup)?;
    let (lidar_frames, camera_frames) = read_output_frames(output)?;
    let (truth_frames, _camera_truth) = read_truth(truth)?;
    let modes = match metrics {
        Some(path) => {
            let text = std::fs::read_to_string(path)?;
            let report: ComparisonReport = serde_json::from_str(&text)?;
            report.modes
        }
        None => Vec::new(),
    };
    let counters = AllCounters::default();
    let diagnostics = match diagnostics {
        Some(path) => {
            let text = std::fs::read_to_string(path)?;
            let value: serde_json::Value = serde_json::from_str(&text)?;
            let mut events = Vec::new();
            if let Some(array) = value.get("events").and_then(|v| v.as_array()) {
                for item in array {
                    events.push(factor_tracker_engine::DiagnosticEvent {
                        arrival_time: factor_tracker_core::Timestamp::from_nanos(
                            item["arrival_ns"].as_i64().unwrap_or(0),
                        ),
                        event_time: item["event_ns"]
                            .as_i64()
                            .map(factor_tracker_core::Timestamp::from_nanos),
                        sensor_id: item["sensor_id"]
                            .as_u64()
                            .map(|v| factor_tracker_core::SensorId::new(v as u32)),
                        code: "replay",
                        level: if item["level"] == "Warn" {
                            factor_tracker_engine::DiagnosticLevel::Warn
                        } else {
                            factor_tracker_engine::DiagnosticLevel::Info
                        },
                        message: item["message"].as_str().unwrap_or("").to_string(),
                    });
                }
            }
            events
        }
        None => Vec::new(),
    };
    report::write_report(
        html,
        title,
        &calibration,
        &modes,
        &lidar_frames,
        &camera_frames,
        &truth_frames,
        &counters,
        &diagnostics,
    )?;
    println!("wrote report to {}", html.display());
    Ok(())
}

fn paths(out_dir: &Path) -> (PathBuf, PathBuf, PathBuf, PathBuf, PathBuf) {
    (
        out_dir.join("setup.json"),
        out_dir.join("inputs.jsonl"),
        out_dir.join("truth.jsonl"),
        out_dir.join("comparison.json"),
        out_dir.join("report.html"),
    )
}

/// Runs the full synthetic camera+LiDAR demo: fixture, three replays, evaluation
/// and an HTML report.
pub fn run_demo(out_dir: &Path, seed: u64, injection: InjectionConfig) -> CliResult<()> {
    println!("factor-tracker camera+LiDAR demo (seed {seed})");
    println!("NOTE: synthetic fixture data; not real sensor data.");
    fixture::write(out_dir, seed)?;
    let (setup, inputs, truth, comparison, report_html) = paths(out_dir);

    let (calibration, poses) = load_setup(&setup)?;
    let all_inputs = read_inputs(&inputs)?;
    let injected = inject::apply(&all_inputs, &injection);
    let (truth_frames, camera_truth): (Vec<GroundTruthFrame>, Vec<CameraAssociationTruth>) =
        read_truth(&truth)?;

    let mut metrics = Vec::new();
    let mut assisted = None;
    for mode in [
        TrackingMode::LidarOnly,
        TrackingMode::CameraAssociated,
        TrackingMode::CameraAssisted,
    ] {
        let config = EngineConfig {
            mode,
            ..EngineConfig::default()
        };
        let engine = replay::run(calibration.clone(), poses.clone(), config, &injected)?;
        let mode_metrics = evaluate_mode(
            engine.lidar_frames(),
            engine.camera_frames(),
            &truth_frames,
            &camera_truth,
            engine.counters(),
            &EvalConfig::default(),
        );
        println!(
            "  {:>17}: coverage={:.2} class_acc={} cam_assoc={} id_switches={}",
            mode.as_str(),
            mode_metrics.track_coverage,
            mode_metrics
                .class_total
                .checked_sub(0)
                .map(|_| if mode_metrics.class_total > 0 {
                    format!(
                        "{:.2}",
                        mode_metrics.class_correct as f64 / mode_metrics.class_total as f64
                    )
                } else {
                    "n/a".to_string()
                })
                .unwrap_or_else(|| "n/a".to_string()),
            mode_metrics
                .camera_assoc_accuracy
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "n/a".to_string()),
            mode_metrics.id_switches,
        );
        metrics.push(mode_metrics);
        if mode == TrackingMode::CameraAssisted {
            assisted = Some(engine);
        }
    }

    let comparison_report = ComparisonReport {
        modes: metrics.clone(),
    };
    std::fs::write(
        &comparison,
        serde_json::to_string_pretty(&comparison_report)?,
    )?;

    let assisted = assisted.expect("camera_assisted mode ran");
    report::write_report(
        &report_html,
        "factor-tracker camera-assisted LiDAR tracking (synthetic)",
        &calibration,
        &metrics,
        assisted.lidar_frames(),
        assisted.camera_frames(),
        &truth_frames,
        &assisted.counters(),
        assisted.log().events(),
    )?;

    println!();
    println!("Outputs under {}:", out_dir.display());
    println!("  setup          : {}", setup.display());
    println!("  inputs         : {}", inputs.display());
    println!("  truth          : {}", truth.display());
    println!("  comparison     : {}", comparison.display());
    println!("  html report    : {}", report_html.display());
    println!();
    println!("Open the HTML report in a browser. The 3D view is a top-down XY projection.");
    Ok(())
}

/// Builds an injection configuration from primitive flag values.
#[allow(clippy::too_many_arguments)]
pub fn injection_from_flags(
    lidar_delay_ns: i64,
    camera_delay_ns: i64,
    duplicate_every: u64,
    reverse_window: usize,
    pause_sensor: Option<&str>,
    pause_start_ns: i64,
    pause_end_ns: i64,
) -> CliResult<InjectionConfig> {
    let pause_sensor = match pause_sensor {
        None => None,
        Some("lidar") => Some(SensorKind::Lidar),
        Some("camera") => Some(SensorKind::Camera),
        Some(other) => return Err(format!("unknown pause sensor '{other}' (lidar|camera)").into()),
    };
    Ok(InjectionConfig {
        lidar_delay_ns,
        camera_delay_ns,
        duplicate_every,
        reverse_window,
        pause_sensor,
        pause_start_ns,
        pause_end_ns,
    })
}
