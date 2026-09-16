//! Command line interface and demos for `perception-engine`.
//!
//! The CLI is intentionally a thin wrapper around the library: it performs
//! file I/O, scenario generation and visualization, and delegates all tracking
//! to `perception-core`.

mod jsonl;
mod render;
mod scenario;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use perception_core::{
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
    name = "perception-cli",
    version,
    about = "Synthetic scenarios, JSONL tracking and local HTML demos for perception-engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
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
                    "perception-engine 2D tracking (synthetic)",
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
                    "perception-engine 3D multi-sensor fusion (synthetic, top-down XY)",
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
    println!("perception-engine demo (seed {seed})");
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
        "perception-engine 2D tracking (synthetic)",
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
        "perception-engine 3D multi-sensor fusion (synthetic, top-down XY)",
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
