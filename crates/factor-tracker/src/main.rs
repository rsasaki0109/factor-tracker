//! Command line interface and demos for `factor-tracker`.
//!
//! The CLI is intentionally a thin wrapper around the library: it performs
//! file I/O, scenario generation and visualization, and delegates all tracking
//! to `factor-tracker-core`.

mod cam;
mod jsonl;
mod render;
mod report;
mod scenario;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use factor_tracker_core::{
    BBox2DTracker, BBox2DTrackerConfig, Cartesian3DTracker, Cartesian3DTrackerConfig,
    ObservationBatch2D, ObservationBatch3D, Timestamp, Track2DSnapshot, Track3DSnapshot,
};

use crate::jsonl::CliResult;
use crate::jsonl::{
    read_jsonl, record_from_batch_2d, record_from_batch_3d, track_record_2d, track_record_3d,
    write_jsonl, Bbox2dTrackRecord, ObservationRecord, Position3dTrackRecord, TrackRecord,
    SCHEMA_VERSION,
};
use crate::scenario::{generate_2d, generate_3d};

#[derive(Parser)]
#[command(
    name = "factor-tracker",
    version,
    about = "Synthetic scenarios, JSONL tracking and local HTML demos for factor-tracker"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Overrides of the 3D tracker defaults.
#[derive(clap::Args)]
struct TrackerArgs {
    /// Override the tracker's initial velocity variance (m^2/s^2).
    #[arg(long)]
    initial_velocity_variance: Option<f64>,
    /// Override the tracker's acceleration spectral density (m^2/s^3).
    #[arg(long)]
    accel_spectral_density: Option<f64>,
    /// Override the tracker's squared-Mahalanobis gate.
    #[arg(long)]
    gate_squared_mahalanobis: Option<f64>,
    /// Associate by horizontal distance (m) to the predicted position, gated at
    /// this distance, instead of by squared Mahalanobis distance; 0 disables.
    #[arg(long)]
    max_association_distance_m: Option<f64>,
    /// Override the matching-cascade window in nanoseconds (0 disables it).
    #[arg(long)]
    recent_track_window_ns: Option<i64>,
    /// Weight of the appearance (ReID) cost; 0 disables it.
    #[arg(long)]
    appearance_weight: Option<f64>,
    /// Hard gate on the appearance cosine distance.
    #[arg(long)]
    appearance_max_distance: Option<f64>,
    /// Do not start tracks closer than this (m) to a confirmed track; 0 disables.
    #[arg(long)]
    min_birth_distance_m: Option<f64>,
    /// With --min-birth-distance-m: start the track anyway once the
    /// suppressed detection has been seen this many times in a row.
    #[arg(long)]
    birth_persistence: Option<u32>,
    /// Re-decide associations over the last N LiDAR scans by factor-graph
    /// cost before output (frames come out N scans late); 0 disables.
    #[arg(long, default_value_t = 0)]
    fgo_lag: usize,
    /// With --fgo-lag: cost of starting a track.
    #[arg(long)]
    fgo_birth_cost: Option<f64>,
    /// With --fgo-lag: cost per scan without a detection.
    #[arg(long)]
    fgo_miss_cost: Option<f64>,
    /// With --fgo-lag: cost of a detection that belongs to no track.
    #[arg(long)]
    fgo_clutter_cost: Option<f64>,
    /// With --fgo-lag: detection position standard deviation (m).
    #[arg(long)]
    fgo_sigma: Option<f64>,
    /// With --fgo-lag: tracks closer than this (m) are compared.
    #[arg(long)]
    fgo_radius: Option<f64>,
    /// With --fgo-lag: report a track for this long after its last
    /// detection (ms); 0 reports only tracks updated in the scan.
    #[arg(long)]
    fgo_coast_ms: Option<i64>,
    /// Observations (distinct scans) before a track is confirmed.
    #[arg(long)]
    confirm_min_observations: Option<u32>,
    /// A confirmed track without observations for this long (ms) is lost.
    #[arg(long)]
    confirmed_to_lost_ms: Option<i64>,
    /// A lost track without observations for this long (ms) is deleted.
    #[arg(long)]
    lost_to_delete_ms: Option<i64>,
    /// Only report tracks observed within this long (ms) before the scan;
    /// 0 reports only tracks updated in the scan. Default: report all.
    #[arg(long)]
    report_coast_ms: Option<i64>,
    /// Detections below this score only update existing tracks and never
    /// start one (ByteTrack-style use of low-score detections).
    #[arg(long)]
    new_track_score_threshold: Option<f64>,
    /// Correct each scan's ego pose with the static landmarks of its batch
    /// (`landmarks` in lidar_batch records): the scan is aligned with a map
    /// of the landmarks seen so far before its detections are tracked.
    #[arg(long)]
    pose_correction: bool,
}

impl TrackerArgs {
    fn overrides(&self) -> cam::TrackerOverrides {
        cam::TrackerOverrides {
            initial_velocity_variance: self.initial_velocity_variance,
            accel_spectral_density: self.accel_spectral_density,
            gate_squared_mahalanobis: self.gate_squared_mahalanobis,
            max_association_distance_m: self.max_association_distance_m,
            recent_track_window_ns: self.recent_track_window_ns,
            appearance_weight: self.appearance_weight,
            appearance_max_distance: self.appearance_max_distance,
            min_birth_distance_m: self.min_birth_distance_m,
            birth_persistence: self.birth_persistence,
            fgo_lag: self.fgo_lag,
            fgo_birth_cost: self.fgo_birth_cost,
            fgo_miss_cost: self.fgo_miss_cost,
            fgo_clutter_cost: self.fgo_clutter_cost,
            fgo_sigma: self.fgo_sigma,
            fgo_radius: self.fgo_radius,
            fgo_coast_ms: self.fgo_coast_ms,
            confirm_min_observations: self.confirm_min_observations,
            confirmed_to_lost_ms: self.confirmed_to_lost_ms,
            lost_to_delete_ms: self.lost_to_delete_ms,
            report_coast_ms: self.report_coast_ms,
            new_track_score_threshold: self.new_track_score_threshold,
            pose_correction: self.pose_correction,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Generate the synthetic 2D scenario (observations + ground truth).
    Gen2d {
        /// Directory to write `observations_2d.jsonl` and `ground_truth_2d.jsonl`.
        #[arg(long, default_value = "demo-out")]
        out_dir: PathBuf,
        /// Deterministic seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Generate the synthetic 3D multi-sensor scenario.
    Gen3d {
        /// Directory to write `observations_3d.jsonl` and `ground_truth_3d.jsonl`.
        #[arg(long, default_value = "demo-out")]
        out_dir: PathBuf,
        /// Deterministic seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Run 2D tracking on a JSONL observation file.
    Track2d {
        /// Input JSONL of `bbox2d_batch` records.
        #[arg(long)]
        input: PathBuf,
        /// Output JSONL of `bbox2d_tracks` records.
        #[arg(long)]
        output: PathBuf,
        /// Optional HTML visualization output.
        #[arg(long)]
        html: Option<PathBuf>,
        /// Optional ground-truth JSONL for visualization.
        #[arg(long)]
        truth: Option<PathBuf>,
    },
    /// Run 3D tracking on a JSONL observation file.
    Track3d {
        /// Input JSONL of `position3d_batch` records.
        #[arg(long)]
        input: PathBuf,
        /// Output JSONL of `position3d_tracks` records.
        #[arg(long)]
        output: PathBuf,
        /// Optional HTML visualization output.
        #[arg(long)]
        html: Option<PathBuf>,
        /// Optional ground-truth JSONL for visualization.
        #[arg(long)]
        truth: Option<PathBuf>,
    },
    /// Generate both scenarios, track them and write JSONL and HTML outputs.
    Demo {
        /// Output directory.
        #[arg(long, default_value = "demo-out")]
        out_dir: PathBuf,
        /// Deterministic seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Write the synthetic camera+LiDAR fixture (setup, inputs, truth).
    CamFixture {
        /// Output directory.
        #[arg(long, default_value = "cam-demo-out")]
        out_dir: PathBuf,
        /// Deterministic seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Prepare a KITTI sequence directory into engine inputs and truth.
    Prepare {
        /// KITTI calib.txt.
        #[arg(long)]
        calib: PathBuf,
        /// KITTI times.txt.
        #[arg(long)]
        times: PathBuf,
        /// Optional KITTI poses.txt.
        #[arg(long)]
        poses: Option<PathBuf>,
        /// Directory of per-frame 3D detection files.
        #[arg(long)]
        lidar_detections: PathBuf,
        /// Directory of per-frame 2D detection files.
        #[arg(long)]
        camera_detections: PathBuf,
        /// Optional directory of per-frame 3D label files (truth).
        #[arg(long)]
        lidar_labels: Option<PathBuf>,
        /// Image width.
        #[arg(long, default_value_t = 1242)]
        image_width: u32,
        /// Image height.
        #[arg(long, default_value_t = 375)]
        image_height: u32,
        /// Near plane in metres.
        #[arg(long, default_value_t = 0.3)]
        near_plane: f64,
        /// Default position variance for 3D detections (m^2).
        #[arg(long, default_value_t = 0.25)]
        default_covariance: f64,
        /// Maximum pose interpolation gap in nanoseconds.
        #[arg(long, default_value_t = 200_000_000)]
        pose_max_gap_ns: i64,
        /// Output setup JSON.
        #[arg(long, default_value = "cam-demo-out/setup.json")]
        out_setup: PathBuf,
        /// Output inputs JSONL.
        #[arg(long, default_value = "cam-demo-out/inputs.jsonl")]
        out_inputs: PathBuf,
        /// Output truth JSONL.
        #[arg(long, default_value = "cam-demo-out/truth.jsonl")]
        out_truth: PathBuf,
    },
    /// Replay camera+LiDAR inputs in a given mode.
    Replay {
        /// Setup JSON.
        #[arg(long)]
        setup: PathBuf,
        /// Inputs JSONL.
        #[arg(long)]
        inputs: PathBuf,
        /// Mode: lidar_only | camera_associated | camera_assisted.
        #[arg(long, default_value = "camera_assisted")]
        mode: String,
        /// Output records JSONL.
        #[arg(long)]
        output: PathBuf,
        /// Diagnostics JSON.
        #[arg(long, default_value = "cam-demo-out/diagnostics.json")]
        diagnostics: PathBuf,
        /// Extra LiDAR arrival delay (ns).
        #[arg(long, default_value_t = 0)]
        lidar_delay_ns: i64,
        /// Extra camera arrival delay (ns).
        #[arg(long, default_value_t = 0)]
        camera_delay_ns: i64,
        /// Duplicate every N-th batch.
        #[arg(long, default_value_t = 0)]
        duplicate_every: u64,
        /// Reverse arrival order in windows of this size.
        #[arg(long, default_value_t = 0)]
        reverse_window: usize,
        /// Pause a sensor: lidar | camera.
        #[arg(long)]
        pause_sensor: Option<String>,
        /// Pause start event time (ns).
        #[arg(long, default_value_t = 0)]
        pause_start_ns: i64,
        /// Pause end event time (ns).
        #[arg(long, default_value_t = 0)]
        pause_end_ns: i64,
        #[command(flatten)]
        tracker: TrackerArgs,
    },
    /// Track live: read input records from stdin (one JSON object per line)
    /// and write each output frame to stdout as soon as it is produced.
    Stream {
        /// Setup JSON.
        #[arg(long)]
        setup: PathBuf,
        /// Mode: lidar_only | camera_associated | camera_assisted.
        #[arg(long, default_value = "lidar_only")]
        mode: String,
        /// How long (event time, ns) a batch waits for out-of-order inputs
        /// before it is tracked; this is the added latency. 0 suits a single
        /// in-order LiDAR stream; use ~200 ms when camera batches arrive late.
        #[arg(long, default_value_t = 0)]
        reorder_window_ns: i64,
        #[command(flatten)]
        tracker: TrackerArgs,
    },
    /// Apply the fixed-lag re-association (--fgo-lag) to an existing output
    /// file, e.g. one written by another tracker in this format (LiDAR
    /// frames with observations and associations).
    Refine {
        /// Input output-records JSONL.
        #[arg(long)]
        input: PathBuf,
        /// Refined output JSONL.
        #[arg(long)]
        output: PathBuf,
        #[command(flatten)]
        tracker: TrackerArgs,
    },
    /// Evaluate replay outputs against ground truth.
    Evaluate {
        /// Output records JSONL.
        #[arg(long)]
        output: PathBuf,
        /// Truth JSONL.
        #[arg(long)]
        truth: PathBuf,
        /// Metrics JSON output.
        #[arg(long, default_value = "cam-demo-out/metrics.json")]
        metrics: PathBuf,
        /// Track-to-truth matching gate in metres.
        #[arg(long, default_value_t = 2.0)]
        match_distance_m: f64,
        /// Match every output track, not only confirmed ones.
        #[arg(long)]
        all_tracks: bool,
        /// Only score ground truth within this range of the robot (metres).
        #[arg(long)]
        zone_range_m: Option<f64>,
        /// Only score ground truth within this bearing of the robot's +x axis
        /// (degrees, default 180 = all around). Used with --zone-range-m.
        #[arg(long, default_value_t = 180.0)]
        zone_half_angle_deg: f64,
        /// Setup JSON with the ego poses used to place the zone (required for
        /// world-frame tracking).
        #[arg(long)]
        setup: Option<PathBuf>,
    },
    /// Render an HTML report for one mode plus optional comparison metrics.
    Report {
        /// Setup JSON.
        #[arg(long)]
        setup: PathBuf,
        /// Output records JSONL.
        #[arg(long)]
        output: PathBuf,
        /// Truth JSONL.
        #[arg(long)]
        truth: PathBuf,
        /// Optional comparison metrics JSON.
        #[arg(long)]
        metrics: Option<PathBuf>,
        /// Optional diagnostics JSON.
        #[arg(long)]
        diagnostics: Option<PathBuf>,
        /// HTML output.
        #[arg(long, default_value = "cam-demo-out/report.html")]
        html: PathBuf,
        /// Report title.
        #[arg(long, default_value = "factor-tracker report")]
        title: String,
    },
    /// Run the full synthetic camera+LiDAR demo (fixture, 3 modes, report).
    CamDemo {
        /// Output directory.
        #[arg(long, default_value = "cam-demo-out")]
        out_dir: PathBuf,
        /// Deterministic seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Extra LiDAR arrival delay (ns).
        #[arg(long, default_value_t = 0)]
        lidar_delay_ns: i64,
        /// Extra camera arrival delay (ns).
        #[arg(long, default_value_t = 0)]
        camera_delay_ns: i64,
        /// Duplicate every N-th batch.
        #[arg(long, default_value_t = 0)]
        duplicate_every: u64,
        /// Reverse arrival order in windows of this size.
        #[arg(long, default_value_t = 0)]
        reverse_window: usize,
        /// Pause a sensor: lidar | camera.
        #[arg(long)]
        pause_sensor: Option<String>,
        /// Pause start event time (ns).
        #[arg(long, default_value_t = 0)]
        pause_start_ns: i64,
        /// Pause end event time (ns).
        #[arg(long, default_value_t = 0)]
        pause_end_ns: i64,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> CliResult<()> {
    match cli.command {
        Command::Gen2d { out_dir, seed } => {
            let scenario = generate_and_write_2d(&out_dir, seed)?;
            println!(
                "wrote 2D scenario (seed {seed}) with {} batches to {}",
                scenario.batches.len(),
                out_dir.display()
            );
            Ok(())
        }
        Command::Gen3d { out_dir, seed } => {
            let scenario = generate_and_write_3d(&out_dir, seed)?;
            println!(
                "wrote 3D scenario (seed {seed}) with {} batches to {}",
                scenario.batches.len(),
                out_dir.display()
            );
            Ok(())
        }
        Command::Track2d {
            input,
            output,
            html,
            truth,
        } => {
            let records = read_jsonl::<ObservationRecord>(&input)?;
            let batches = records
                .into_iter()
                .map(ObservationRecord::into_batch_2d)
                .collect::<CliResult<Vec<_>>>()?;
            let (track_records, track_frames) = track_2d(&batches)?;
            write_jsonl(&output, &track_records)?;
            let sensors = unique_sensors_2d(&batches);
            if let Some(html_path) = html {
                let truth_frames = match truth {
                    Some(path) => scenario::truth_frames_2d(&read_jsonl(&path)?)?,
                    None => Vec::new(),
                };
                render::render_2d_html(
                    &html_path,
                    "factor-tracker 2D tracking (synthetic)",
                    &batches,
                    &truth_frames,
                    &track_frames,
                    None,
                    &sensors,
                )?;
                println!("wrote visualization to {}", html_path.display());
            }
            println!(
                "tracked {} observation batches -> {} track records at {}",
                batches.len(),
                track_records.len(),
                output.display()
            );
            Ok(())
        }
        Command::Track3d {
            input,
            output,
            html,
            truth,
        } => {
            let records = read_jsonl::<ObservationRecord>(&input)?;
            let batches = records
                .into_iter()
                .map(ObservationRecord::into_batch_3d)
                .collect::<CliResult<Vec<_>>>()?;
            let (track_records, track_frames) = track_3d(&batches)?;
            write_jsonl(&output, &track_records)?;
            let sensors = unique_sensors_3d(&batches);
            if let Some(html_path) = html {
                let truth_frames = match truth {
                    Some(path) => scenario::truth_frames_3d(&read_jsonl(&path)?)?,
                    None => Vec::new(),
                };
                render::render_3d_html(
                    &html_path,
                    "factor-tracker 3D multi-sensor fusion (synthetic, top-down XY)",
                    &batches,
                    &truth_frames,
                    &track_frames,
                    &sensors,
                )?;
                println!("wrote visualization to {}", html_path.display());
            }
            println!(
                "tracked {} observation batches -> {} track records at {}",
                batches.len(),
                track_records.len(),
                output.display()
            );
            Ok(())
        }
        Command::Demo { out_dir, seed } => run_demo(&out_dir, seed),
        Command::CamFixture { out_dir, seed } => cam::run_fixture(&out_dir, seed),
        Command::Prepare {
            calib,
            times,
            poses,
            lidar_detections,
            camera_detections,
            lidar_labels,
            image_width,
            image_height,
            near_plane,
            default_covariance,
            pose_max_gap_ns,
            out_setup,
            out_inputs,
            out_truth,
        } => cam::run_prepare(
            &calib,
            &times,
            poses.as_deref(),
            &lidar_detections,
            &camera_detections,
            lidar_labels.as_deref(),
            None,
            image_width,
            image_height,
            near_plane,
            default_covariance,
            pose_max_gap_ns,
            &out_setup,
            &out_inputs,
            &out_truth,
        ),
        Command::Replay {
            setup,
            inputs,
            mode,
            output,
            diagnostics,
            lidar_delay_ns,
            camera_delay_ns,
            duplicate_every,
            reverse_window,
            pause_sensor,
            pause_start_ns,
            pause_end_ns,
            tracker,
        } => {
            let injection = cam::injection_from_flags(
                lidar_delay_ns,
                camera_delay_ns,
                duplicate_every,
                reverse_window,
                pause_sensor.as_deref(),
                pause_start_ns,
                pause_end_ns,
            )?;
            let tracker = tracker.overrides();
            cam::run_replay(
                &setup,
                &inputs,
                &mode,
                &output,
                &diagnostics,
                injection,
                tracker,
            )
        }
        Command::Stream {
            setup,
            mode,
            reorder_window_ns,
            tracker,
        } => {
            let stdin = std::io::stdin();
            let stdout = std::io::stdout();
            cam::run_stream(
                &setup,
                &mode,
                reorder_window_ns,
                tracker.overrides(),
                stdin.lock(),
                std::io::BufWriter::new(stdout.lock()),
            )
        }
        Command::Refine {
            input,
            output,
            tracker,
        } => cam::run_refine(&input, &output, tracker.overrides()),
        Command::Evaluate {
            output,
            truth,
            metrics,
            match_distance_m,
            all_tracks,
            zone_range_m,
            zone_half_angle_deg,
            setup,
        } => {
            let zone = zone_range_m.map(|max_range_m| factor_tracker_engine::eval::EvalZone {
                max_range_m,
                max_bearing_rad: zone_half_angle_deg.to_radians(),
            });
            cam::run_evaluate(
                &output,
                &truth,
                &metrics,
                &factor_tracker_engine::eval::EvalConfig {
                    match_distance_m,
                    confirmed_only: !all_tracks,
                },
                zone.as_ref().map(|zone| (zone, setup.as_deref())),
            )
        }
        Command::Report {
            setup,
            output,
            truth,
            metrics,
            diagnostics,
            html,
            title,
        } => cam::run_report(
            &setup,
            &output,
            &truth,
            metrics.as_deref(),
            &html,
            &title,
            diagnostics.as_deref(),
        ),
        Command::CamDemo {
            out_dir,
            seed,
            lidar_delay_ns,
            camera_delay_ns,
            duplicate_every,
            reverse_window,
            pause_sensor,
            pause_start_ns,
            pause_end_ns,
        } => {
            let injection = cam::injection_from_flags(
                lidar_delay_ns,
                camera_delay_ns,
                duplicate_every,
                reverse_window,
                pause_sensor.as_deref(),
                pause_start_ns,
                pause_end_ns,
            )?;
            cam::run_demo(&out_dir, seed, injection)
        }
    }
}

fn observations_path_2d(out_dir: &Path) -> PathBuf {
    out_dir.join("observations_2d.jsonl")
}
fn ground_truth_path_2d(out_dir: &Path) -> PathBuf {
    out_dir.join("ground_truth_2d.jsonl")
}
fn tracks_path_2d(out_dir: &Path) -> PathBuf {
    out_dir.join("tracks_2d.jsonl")
}
fn html_path_2d(out_dir: &Path) -> PathBuf {
    out_dir.join("demo_2d.html")
}
fn observations_path_3d(out_dir: &Path) -> PathBuf {
    out_dir.join("observations_3d.jsonl")
}
fn ground_truth_path_3d(out_dir: &Path) -> PathBuf {
    out_dir.join("ground_truth_3d.jsonl")
}
fn tracks_path_3d(out_dir: &Path) -> PathBuf {
    out_dir.join("tracks_3d.jsonl")
}
fn html_path_3d(out_dir: &Path) -> PathBuf {
    out_dir.join("demo_3d_fusion.html")
}

fn generate_and_write_2d(out_dir: &Path, seed: u64) -> CliResult<scenario::Scenario2d> {
    std::fs::create_dir_all(out_dir)?;
    let scenario = generate_2d(seed);
    let observation_records: Vec<_> = scenario.batches.iter().map(record_from_batch_2d).collect();
    write_jsonl(&observations_path_2d(out_dir), &observation_records)?;
    write_jsonl(&ground_truth_path_2d(out_dir), &scenario.truth_records())?;
    Ok(scenario)
}

fn generate_and_write_3d(out_dir: &Path, seed: u64) -> CliResult<scenario::Scenario3d> {
    std::fs::create_dir_all(out_dir)?;
    let scenario = generate_3d(seed);
    let observation_records: Vec<_> = scenario.batches.iter().map(record_from_batch_3d).collect();
    write_jsonl(&observations_path_3d(out_dir), &observation_records)?;
    write_jsonl(&ground_truth_path_3d(out_dir), &scenario.truth_records())?;
    Ok(scenario)
}

fn run_demo(out_dir: &Path, seed: u64) -> CliResult<()> {
    println!("factor-tracker demo (seed {seed})");
    println!("NOTE: all data is synthetic; nothing here is validated on real sensors.");

    let scenario_2d = generate_and_write_2d(out_dir, seed)?;
    let records_2d = read_jsonl::<ObservationRecord>(&observations_path_2d(out_dir))?;
    let batches_2d = records_2d
        .into_iter()
        .map(ObservationRecord::into_batch_2d)
        .collect::<CliResult<Vec<_>>>()?;
    let (track_records_2d, track_frames_2d) = track_2d(&batches_2d)?;
    write_jsonl(&tracks_path_2d(out_dir), &track_records_2d)?;
    let truth_2d = scenario::truth_frames_2d(&read_jsonl(&ground_truth_path_2d(out_dir))?)?;
    render::render_2d_html(
        &html_path_2d(out_dir),
        "factor-tracker 2D tracking (synthetic)",
        &batches_2d,
        &truth_2d,
        &track_frames_2d,
        Some((scenario_2d.image_width, scenario_2d.image_height)),
        &scenario_2d.sensor_ids,
    )?;

    let scenario_3d = generate_and_write_3d(out_dir, seed)?;
    let records_3d = read_jsonl::<ObservationRecord>(&observations_path_3d(out_dir))?;
    let batches_3d = records_3d
        .into_iter()
        .map(ObservationRecord::into_batch_3d)
        .collect::<CliResult<Vec<_>>>()?;
    let (track_records_3d, track_frames_3d) = track_3d(&batches_3d)?;
    write_jsonl(&tracks_path_3d(out_dir), &track_records_3d)?;
    let truth_3d = scenario::truth_frames_3d(&read_jsonl(&ground_truth_path_3d(out_dir))?)?;
    render::render_3d_html(
        &html_path_3d(out_dir),
        "factor-tracker 3D multi-sensor fusion (synthetic, top-down XY)",
        &batches_3d,
        &truth_3d,
        &track_frames_3d,
        &scenario_3d.sensor_ids,
    )?;

    println!();
    println!("Outputs written under {}:", out_dir.display());
    println!(
        "  2D observations : {}",
        observations_path_2d(out_dir).display()
    );
    println!(
        "  2D ground truth : {}",
        ground_truth_path_2d(out_dir).display()
    );
    println!("  2D tracks       : {}", tracks_path_2d(out_dir).display());
    println!("  2D visualization: {}", html_path_2d(out_dir).display());
    println!(
        "  3D observations : {}",
        observations_path_3d(out_dir).display()
    );
    println!(
        "  3D ground truth : {}",
        ground_truth_path_3d(out_dir).display()
    );
    println!("  3D tracks       : {}", tracks_path_3d(out_dir).display());
    println!("  3D visualization: {}", html_path_3d(out_dir).display());
    println!();
    println!("Open the HTML files in a browser. The 3D view is a top-down XY projection.");
    Ok(())
}

type TrackFrames2d = Vec<(Timestamp, Vec<Track2DSnapshot>)>;
type TrackFrames3d = Vec<(Timestamp, Vec<Track3DSnapshot>)>;

fn track_2d(batches: &[ObservationBatch2D]) -> CliResult<(Vec<TrackRecord>, TrackFrames2d)> {
    let mut tracker = BBox2DTracker::new(BBox2DTrackerConfig::default())?;
    let mut records = Vec::new();
    let mut frames = Vec::new();
    let mut index = 0;
    while index < batches.len() {
        let timestamp = batches[index].timestamp;
        let mut end = index;
        while end < batches.len() && batches[end].timestamp == timestamp {
            tracker.process_batch(&batches[end])?;
            end += 1;
        }
        let snapshots = tracker.snapshots();
        let track_records: Vec<Bbox2dTrackRecord> = snapshots.iter().map(track_record_2d).collect();
        records.push(TrackRecord::Bbox2dTracks {
            schema_version: SCHEMA_VERSION,
            timestamp_ns: timestamp.as_nanos(),
            tracks: track_records,
        });
        frames.push((timestamp, snapshots));
        index = end;
    }
    Ok((records, frames))
}

fn track_3d(batches: &[ObservationBatch3D]) -> CliResult<(Vec<TrackRecord>, TrackFrames3d)> {
    let mut tracker = Cartesian3DTracker::new(Cartesian3DTrackerConfig::default())?;
    let mut records = Vec::new();
    let mut frames = Vec::new();
    let mut index = 0;
    while index < batches.len() {
        let timestamp = batches[index].timestamp;
        let mut end = index;
        while end < batches.len() && batches[end].timestamp == timestamp {
            tracker.process_batch(&batches[end])?;
            end += 1;
        }
        let snapshots = tracker.snapshots();
        let track_records: Vec<Position3dTrackRecord> =
            snapshots.iter().map(track_record_3d).collect();
        records.push(TrackRecord::Position3dTracks {
            schema_version: SCHEMA_VERSION,
            timestamp_ns: timestamp.as_nanos(),
            tracks: track_records,
        });
        frames.push((timestamp, snapshots));
        index = end;
    }
    Ok((records, frames))
}

fn unique_sensors_2d(batches: &[ObservationBatch2D]) -> Vec<u32> {
    let mut sensors: Vec<u32> = batches
        .iter()
        .map(|batch| batch.sensor_id.value())
        .collect();
    sensors.sort_unstable();
    sensors.dedup();
    sensors
}

fn unique_sensors_3d(batches: &[ObservationBatch3D]) -> Vec<u32> {
    let mut sensors: Vec<u32> = batches
        .iter()
        .map(|batch| batch.sensor_id.value())
        .collect();
    sensors.sort_unstable();
    sensors.dedup();
    sensors
}
