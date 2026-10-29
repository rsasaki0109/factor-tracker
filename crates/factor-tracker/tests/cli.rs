//! Integration tests for the CLI, exercised through the compiled binary.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

fn temp_dir(name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!("factor-tracker-it-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_factor-tracker"))
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

#[test]
fn stream_matches_replay_and_skips_bad_lines() {
    let dir = temp_dir("stream");
    assert!(cli()
        .args(["cam-fixture", "--out-dir"])
        .arg(&dir)
        .status()
        .unwrap()
        .success());
    let replayed = dir.join("replay.jsonl");
    assert!(cli()
        .args(["replay", "--mode", "camera_assisted", "--setup"])
        .arg(dir.join("setup.json"))
        .arg("--inputs")
        .arg(dir.join("inputs.jsonl"))
        .arg("--output")
        .arg(&replayed)
        .arg("--diagnostics")
        .arg(dir.join("diagnostics.json"))
        .status()
        .unwrap()
        .success());

    // The replay default reorder window, and one malformed line in the middle.
    let inputs = std::fs::read_to_string(dir.join("inputs.jsonl")).unwrap();
    let mut lines: Vec<&str> = inputs.lines().collect();
    lines.insert(lines.len() / 2, "{\"type\":\"not_a_batch\"}");
    let mut child = cli()
        .args([
            "stream",
            "--mode",
            "camera_assisted",
            "--reorder-window-ns",
            "200000000",
            "--setup",
        ])
        .arg(dir.join("setup.json"))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    {
        use std::io::Write;
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(lines.join("\n").as_bytes()).unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("skipped"));

    let streamed: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(!streamed.is_empty());
    assert_eq!(streamed, read_lines(&replayed));
}

#[test]
fn stream_with_fgo_lag_matches_replay_with_fgo_lag() {
    let dir = temp_dir("stream-fgo");
    assert!(cli()
        .args(["cam-fixture", "--out-dir"])
        .arg(&dir)
        .status()
        .unwrap()
        .success());
    let replayed = dir.join("replay.jsonl");
    assert!(cli()
        .args([
            "replay",
            "--mode",
            "lidar_only",
            "--fgo-lag",
            "5",
            "--setup"
        ])
        .arg(dir.join("setup.json"))
        .arg("--inputs")
        .arg(dir.join("inputs.jsonl"))
        .arg("--output")
        .arg(&replayed)
        .arg("--diagnostics")
        .arg(dir.join("diagnostics.json"))
        .status()
        .unwrap()
        .success());
    let output = cli()
        .args([
            "stream",
            "--mode",
            "lidar_only",
            "--fgo-lag",
            "5",
            "--setup",
        ])
        .arg(dir.join("setup.json"))
        .stdin(std::fs::File::open(dir.join("inputs.jsonl")).unwrap())
        .output()
        .unwrap();
    assert!(output.status.success());
    let streamed: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let expected = read_lines(&replayed);
    assert_eq!(streamed.len(), expected.len());
    assert_eq!(streamed, expected);
}

#[test]
fn stream_with_smoothing_writes_realtime_and_smoothed_frames() {
    let dir = temp_dir("stream-smooth");
    assert!(cli()
        .args(["cam-fixture", "--out-dir"])
        .arg(&dir)
        .status()
        .unwrap()
        .success());
    let run = |extra: &[&str]| -> Vec<Value> {
        let output = cli()
            .args(["stream", "--mode", "lidar_only", "--setup"])
            .arg(dir.join("setup.json"))
            .args(extra)
            .stdin(std::fs::File::open(dir.join("inputs.jsonl")).unwrap())
            .output()
            .unwrap();
        assert!(output.status.success());
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .filter(|v: &Value| v["type"] == "lidar_frame")
            .collect()
    };
    let plain = run(&[]);
    let dual = run(&["--smooth-lag", "3", "--smooth-realtime"]);
    let realtime: Vec<&Value> = dual
        .iter()
        .filter(|v| v.get("smoothed").is_none())
        .collect();
    let smoothed: Vec<&Value> = dual.iter().filter(|v| v["smoothed"] == true).collect();
    // The real-time frames are the unsmoothed ones; every frame is also
    // emitted smoothed, with the same timestamps and track ids.
    assert_eq!(realtime, plain.iter().collect::<Vec<_>>());
    assert_eq!(smoothed.len(), plain.len());
    for (s, p) in smoothed.iter().zip(&plain) {
        assert_eq!(s["timestamp_ns"], p["timestamp_ns"]);
        let ids = |v: &Value| -> Vec<u64> {
            v["tracks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t["track_id"].as_u64().unwrap())
                .collect()
        };
        assert_eq!(ids(s), ids(p));
    }
}
