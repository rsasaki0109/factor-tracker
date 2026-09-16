//! Integration tests for the CLI, exercised through the compiled binary.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

fn temp_dir(name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!("perception-cli-it-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_perception-cli"))
}

fn read_lines(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn demo_writes_all_outputs() {
    let dir = temp_dir("demo");
    let out = dir.join("demo");
    let status = cli()
        .args(["demo", "--out-dir"])
        .arg(&out)
        .args(["--seed", "7"])
        .status()
        .unwrap();
    assert!(status.success());

    for name in [
        "observations_2d.jsonl",
        "ground_truth_2d.jsonl",
        "tracks_2d.jsonl",
        "demo_2d.html",
        "observations_3d.jsonl",
        "ground_truth_3d.jsonl",
        "tracks_3d.jsonl",
        "demo_3d_fusion.html",
    ] {
        let path = out.join(name);
        assert!(path.exists(), "missing {name}");
        assert!(std::fs::metadata(&path).unwrap().len() > 0, "empty {name}");
    }

    let tracks = read_lines(&out.join("tracks_2d.jsonl"));
    assert!(!tracks.is_empty());
    assert_eq!(tracks[0]["type"], "bbox2d_tracks");
    assert_eq!(tracks[0]["schema_version"], 1);
    assert!(tracks[0]["tracks"].is_array());

    let confirmed_seen = tracks.iter().any(|record| {
        record["tracks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|track| track["lifecycle"] == "confirmed")
    });
    assert!(confirmed_seen, "expected at least one confirmed 2D track");

    let html = std::fs::read_to_string(out.join("demo_2d.html")).unwrap();
    assert!(html.contains("const DATA ="));
    assert!(html.contains("top-down"));
    assert!(html.contains("Synthetic data"));
}

#[test]
fn gen_then_track_round_trips_through_jsonl() {
    let dir = temp_dir("round-trip");
    let observations = dir.join("observations_2d.jsonl");
    let truth = dir.join("ground_truth_2d.jsonl");
    let tracks = dir.join("tracks_2d.jsonl");
    let html = dir.join("demo_2d.html");

    assert!(cli()
        .args(["gen2d", "--out-dir"])
        .arg(&dir)
        .status()
        .unwrap()
        .success());

    let status = cli()
        .args(["track2d", "--input"])
        .arg(&observations)
        .args(["--output"])
        .arg(&tracks)
        .args(["--html"])
        .arg(&html)
        .args(["--truth"])
        .arg(&truth)
        .status()
        .unwrap();
    assert!(status.success());

    let records = read_lines(&tracks);
    assert_eq!(records[0]["type"], "bbox2d_tracks");
    // Timestamps must be non-decreasing in the output.
    let mut previous = i64::MIN;
    for record in &records {
        let timestamp = record["timestamp_ns"].as_i64().unwrap();
        assert!(timestamp >= previous);
        previous = timestamp;
    }
    assert!(html.exists());
}

#[test]
fn generation_is_deterministic_for_a_fixed_seed() {
    let dir_a = temp_dir("det-a");
    let dir_b = temp_dir("det-b");
    for dir in [&dir_a, &dir_b] {
        assert!(cli()
            .args(["gen3d", "--out-dir"])
            .arg(dir)
            .args(["--seed", "1234"])
            .status()
            .unwrap()
            .success());
    }
    let a = std::fs::read_to_string(dir_a.join("observations_3d.jsonl")).unwrap();
    let b = std::fs::read_to_string(dir_b.join("observations_3d.jsonl")).unwrap();
    assert_eq!(a, b);
}

#[test]
fn wrong_record_type_is_rejected() {
    let dir = temp_dir("wrong-type");
    assert!(cli()
        .args(["gen2d", "--out-dir"])
        .arg(&dir)
        .status()
        .unwrap()
        .success());

    let status = cli()
        .args(["track3d", "--input"])
        .arg(dir.join("observations_2d.jsonl"))
        .args(["--output"])
        .arg(dir.join("tracks.jsonl"))
        .status()
        .unwrap();
    assert!(!status.success(), "track3d should reject 2D input");
}

#[test]
fn track3d_fuses_two_sensors_into_few_tracks() {
    let dir = temp_dir("track3d");
    let observations = dir.join("observations_3d.jsonl");
    let tracks = dir.join("tracks_3d.jsonl");
    assert!(cli()
        .args(["gen3d", "--out-dir"])
        .arg(&dir)
        .status()
        .unwrap()
        .success());
    assert!(cli()
        .args(["track3d", "--input"])
        .arg(&observations)
        .args(["--output"])
        .arg(&tracks)
        .status()
        .unwrap()
        .success());

    let records = read_lines(&tracks);
    let last = records.last().unwrap();
    let confirmed = last["tracks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|track| track["lifecycle"] == "confirmed")
        .count();
    // Three objects are tracked; allow one extra short-lived false-positive
    // track to have survived to the end without being over-specific.
    assert!(
        (3..=4).contains(&confirmed),
        "expected 3..=4 confirmed 3D tracks, found {confirmed}"
    );
    // At least one track must have been updated by both sensors.
    let multi_sensor = last["tracks"]
        .as_array()
        .unwrap()
        .iter()
        .any(|track| track["contributing_sensors"].as_array().unwrap().len() >= 2);
    assert!(
        multi_sensor,
        "expected at least one fused multi-sensor track"
    );
}
