//! Self-contained HTML visualizations for the 2D and 3D demos.
//!
//! The generated page embeds all data inline and renders on a `<canvas>` with
//! plain JavaScript, so it can be opened directly from the file system with no
//! server or external assets.
//!
//! The 3D view is a **top-down (XY) projection**, not a 3D rendering.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use factor_tracker_core::{
    ObservationBatch2D, ObservationBatch3D, Timestamp, Track2DSnapshot, Track3DSnapshot,
};
use serde::Serialize;

use crate::jsonl::CliResult;
use crate::scenario::{Truth2dFrame, Truth3dFrame};

const SYNTHETIC_NOTE: &str =
    "Synthetic data, not validated on real sensors. The 3D view is a top-down (XY) projection, not a 3D rendering.";

#[derive(Serialize)]
struct Obs2dItem {
    s: u32,
    bbox: [f64; 4],
}

#[derive(Serialize)]
struct Track2dItem {
    id: u64,
    state: String,
    bbox: [f64; 4],
    pred: bool,
}

#[derive(Serialize)]
struct Truth2dItem {
    id: u32,
    class: u32,
    bbox: [f64; 4],
    detected: bool,
}

#[derive(Serialize)]
struct Frame2d {
    ts: i64,
    obs: Vec<Obs2dItem>,
    tracks: Vec<Track2dItem>,
    truth: Vec<Truth2dItem>,
}

#[derive(Serialize)]
struct Obs3dItem {
    s: u32,
    p: [f64; 3],
}

#[derive(Serialize)]
struct Track3dItem {
    id: u64,
    state: String,
    p: [f64; 3],
    v: [f64; 3],
    pred: bool,
}

#[derive(Serialize)]
struct Truth3dItem {
    id: u32,
    class: u32,
    p: [f64; 3],
    detected: bool,
}

#[derive(Serialize)]
struct Frame3d {
    ts: i64,
    obs: Vec<Obs3dItem>,
    tracks: Vec<Track3dItem>,
    truth: Vec<Truth3dItem>,
}

/// Renders a 2D tracking result to a standalone HTML file.
#[allow(clippy::too_many_arguments)]
pub fn render_2d_html(
    path: &Path,
    title: &str,
    observations: &[ObservationBatch2D],
    truth: &[Truth2dFrame],
    track_frames: &[(Timestamp, Vec<Track2DSnapshot>)],
    image_size: Option<(f64, f64)>,
    sensor_ids: &[u32],
) -> CliResult<()> {
    let mut observed: BTreeMap<i64, Vec<Obs2dItem>> = BTreeMap::new();
    for batch in observations {
        let entry = observed.entry(batch.timestamp.as_nanos()).or_default();
        for observation in &batch.observations {
            entry.push(Obs2dItem {
                s: batch.sensor_id.value(),
                bbox: [
                    observation.bbox.x_min,
                    observation.bbox.y_min,
                    observation.bbox.x_max,
                    observation.bbox.y_max,
                ],
            });
        }
    }

    let mut truth_map: BTreeMap<i64, Vec<Truth2dItem>> = BTreeMap::new();
    for frame in truth {
        truth_map.insert(
            frame.timestamp.as_nanos(),
            frame
                .objects
                .iter()
                .map(|object| Truth2dItem {
                    id: object.truth_id,
                    class: object.class,
                    bbox: [
                        object.bbox.x_min,
                        object.bbox.y_min,
                        object.bbox.x_max,
                        object.bbox.y_max,
                    ],
                    detected: object.detected,
                })
                .collect(),
        );
    }

    let mut track_map: BTreeMap<i64, Vec<Track2dItem>> = BTreeMap::new();
    for (timestamp, snapshots) in track_frames {
        track_map.insert(
            timestamp.as_nanos(),
            snapshots
                .iter()
                .map(|snapshot| Track2dItem {
                    id: snapshot.track_id.value(),
                    state: snapshot.lifecycle.as_str().to_string(),
                    bbox: [
                        snapshot.bbox.x_min,
                        snapshot.bbox.y_min,
                        snapshot.bbox.x_max,
                        snapshot.bbox.y_max,
                    ],
                    pred: snapshot.is_prediction_only(),
                })
                .collect(),
        );
    }

    let mut timestamps: Vec<i64> = observed
        .keys()
        .chain(truth_map.keys())
        .chain(track_map.keys())
        .copied()
        .collect();
    timestamps.sort_unstable();
    timestamps.dedup();

    let frames: Vec<Frame2d> = timestamps
        .iter()
        .map(|&ts| Frame2d {
            ts,
            obs: observed.remove(&ts).unwrap_or_default(),
            tracks: track_map.remove(&ts).unwrap_or_default(),
            truth: truth_map.remove(&ts).unwrap_or_default(),
        })
        .collect();

    let (width, height) = image_size.unwrap_or_else(|| bounds_2d(&frames));

    let data = serde_json::json!({
        "mode": "2d",
        "title": title,
        "synthetic": true,
        "note": SYNTHETIC_NOTE,
        "canvas": { "width": 960, "height": 540 },
        "width": width.max(1.0),
        "height": height.max(1.0),
        "sensors": sensor_ids,
        "frames": frames,
    });
    write_html(path, &data, &build_legend("2d"))
}

/// Renders a 3D tracking result to a standalone HTML file (top-down XY view).
pub fn render_3d_html(
    path: &Path,
    title: &str,
    observations: &[ObservationBatch3D],
    truth: &[Truth3dFrame],
    track_frames: &[(Timestamp, Vec<Track3DSnapshot>)],
    sensor_ids: &[u32],
) -> CliResult<()> {
    let mut observed: BTreeMap<i64, Vec<Obs3dItem>> = BTreeMap::new();
    for batch in observations {
        let entry = observed.entry(batch.timestamp.as_nanos()).or_default();
        for observation in &batch.observations {
            entry.push(Obs3dItem {
                s: batch.sensor_id.value(),
                p: [
                    observation.position.x,
                    observation.position.y,
                    observation.position.z,
                ],
            });
        }
    }

    let mut truth_map: BTreeMap<i64, Vec<Truth3dItem>> = BTreeMap::new();
    for frame in truth {
        truth_map.insert(
            frame.timestamp.as_nanos(),
            frame
                .objects
                .iter()
                .map(|object| Truth3dItem {
                    id: object.truth_id,
                    class: object.class,
                    p: [object.position.x, object.position.y, object.position.z],
                    detected: object.detected,
                })
                .collect(),
        );
    }

    let mut track_map: BTreeMap<i64, Vec<Track3dItem>> = BTreeMap::new();
    for (timestamp, snapshots) in track_frames {
        track_map.insert(
            timestamp.as_nanos(),
            snapshots
                .iter()
                .map(|snapshot| Track3dItem {
                    id: snapshot.track_id.value(),
                    state: snapshot.lifecycle.as_str().to_string(),
                    p: [
                        snapshot.position.x,
                        snapshot.position.y,
                        snapshot.position.z,
                    ],
                    v: [
                        snapshot.velocity.x,
                        snapshot.velocity.y,
                        snapshot.velocity.z,
                    ],
                    pred: snapshot.is_prediction_only(),
                })
                .collect(),
        );
    }

    let mut timestamps: Vec<i64> = observed
        .keys()
        .chain(truth_map.keys())
        .chain(track_map.keys())
        .copied()
        .collect();
    timestamps.sort_unstable();
    timestamps.dedup();

    let frames: Vec<Frame3d> = timestamps
        .iter()
        .map(|&ts| Frame3d {
            ts,
            obs: observed.remove(&ts).unwrap_or_default(),
            tracks: track_map.remove(&ts).unwrap_or_default(),
            truth: truth_map.remove(&ts).unwrap_or_default(),
        })
        .collect();

    let bounds = bounds_3d(&frames);

    let data = serde_json::json!({
        "mode": "3d",
        "title": title,
        "synthetic": true,
        "note": SYNTHETIC_NOTE,
        "canvas": { "width": 820, "height": 600 },
        "bounds": bounds,
        "sensors": sensor_ids,
        "frames": frames,
    });
    write_html(path, &data, &build_legend("3d"))
}

fn bounds_2d(frames: &[Frame2d]) -> (f64, f64) {
    let mut max_x = 1.0_f64;
    let mut max_y = 1.0_f64;
    for frame in frames {
        for item in &frame.truth {
            max_x = max_x.max(item.bbox[2]);
            max_y = max_y.max(item.bbox[3]);
        }
        for item in &frame.obs {
            max_x = max_x.max(item.bbox[2]);
            max_y = max_y.max(item.bbox[3]);
        }
    }
    (max_x + 20.0, max_y + 20.0)
}

fn bounds_3d(frames: &[Frame3d]) -> [f64; 4] {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut update = |x: f64, y: f64| {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    };
    for frame in frames {
        for item in &frame.truth {
            update(item.p[0], item.p[1]);
        }
        for item in &frame.obs {
            update(item.p[0], item.p[1]);
        }
        for item in &frame.tracks {
            update(item.p[0], item.p[1]);
        }
    }
    if !min_x.is_finite() {
        return [0.0, 1.0, 0.0, 1.0];
    }
    let pad_x = (max_x - min_x).max(1.0) * 0.1;
    let pad_y = (max_y - min_y).max(1.0) * 0.1;
    [min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y]
}

fn write_html(path: &Path, data: &serde_json::Value, legend: &str) -> CliResult<()> {
    let json = serde_json::to_string(data)?;
    let html = TEMPLATE
        .replace("__LEGEND__", legend)
        .replace("__DATA__", &json);
    std::fs::write(path, html)?;
    Ok(())
}

struct LegendEntry {
    label: &'static str,
    color: &'static str,
    dash: bool,
}

fn build_legend(mode: &str) -> String {
    let mut entries = vec![
        LegendEntry {
            label: "observation (sensor 1)",
            color: "#3b82f6",
            dash: false,
        },
        LegendEntry {
            label: "observation (sensor 2)",
            color: "#d946ef",
            dash: false,
        },
        LegendEntry {
            label: "track: tentative",
            color: "#e6a23c",
            dash: false,
        },
        LegendEntry {
            label: "track: confirmed",
            color: "#2eb872",
            dash: false,
        },
        LegendEntry {
            label: "track: lost (prediction only)",
            color: "#e04f5f",
            dash: false,
        },
    ];
    if mode == "2d" {
        entries.push(LegendEntry {
            label: "ground truth (dashed)",
            color: "#94a3b8",
            dash: true,
        });
    }
    let mut out = String::new();
    for entry in entries {
        let _ = write!(
            out,
            "<span class=\"legend-item\"><span class=\"swatch{}\" style=\"background:{}\"></span>{}</span>",
            if entry.dash { " dashed" } else { "" },
            entry.color,
            entry.label
        );
    }
    out
}

const TEMPLATE: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>factor-tracker demo</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #0f172a; color: #e2e8f0; font-family: system-ui, sans-serif; }
  #app { max-width: 1000px; margin: 0 auto; padding: 16px; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  #note { font-size: 12px; color: #94a3b8; margin-bottom: 10px; }
  canvas { width: 100%; height: auto; background: #111827; border: 1px solid #1f2937; border-radius: 8px; display: block; }
  #controls { display: flex; gap: 10px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
  button { background: #2563eb; color: white; border: 0; border-radius: 6px; padding: 6px 12px; cursor: pointer; }
  button.secondary { background: #334155; }
  input[type=range] { flex: 1; min-width: 200px; }
  #info { font-variant-numeric: tabular-nums; font-size: 13px; color: #cbd5e1; }
  #legend { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; color: #cbd5e1; }
  .legend-item { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { display: inline-block; width: 14px; height: 10px; border-radius: 2px; }
  .swatch.dashed { background: none !important; border: 2px dashed #94a3b8; height: 8px; }
  label.toggle { font-size: 12px; color: #cbd5e1; display: inline-flex; gap: 4px; align-items: center; }
</style>
</head>
<body>
<div id="app">
  <h1 id="title"></h1>
  <div id="note"></div>
  <canvas id="canvas"></canvas>
  <div id="controls">
    <button id="play">Pause</button>
    <button id="prev" class="secondary">&lt;</button>
    <button id="next" class="secondary">&gt;</button>
    <input id="slider" type="range" min="0" max="0" value="0">
    <span id="info"></span>
    <label class="toggle"><input id="show-truth" type="checkbox" checked> ground truth</label>
  </div>
  <div id="legend">__LEGEND__</div>
</div>
<script>
const DATA = __DATA__;
const LIFECYCLE_COLORS = { tentative: '#e6a23c', confirmed: '#2eb872', lost: '#e04f5f', deleted: '#64748b' };
const SENSOR_COLORS = ['#3b82f6', '#d946ef', '#14b8a6', '#f59e0b', '#8b5cf6'];

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const W = DATA.canvas.width, H = DATA.canvas.height;
canvas.width = W; canvas.height = H;

const slider = document.getElementById('slider');
const playButton = document.getElementById('play');
const info = document.getElementById('info');
const showTruth = document.getElementById('show-truth');

slider.max = String(Math.max(0, DATA.frames.length - 1));

let frameIndex = 0;
let playing = true;
let accumulator = 0;
let lastTime = performance.now();

function sensorColor(sensor) {
  const index = DATA.sensors.indexOf(sensor);
  return SENSOR_COLORS[(index < 0 ? 0 : index) % SENSOR_COLORS.length];
}

let project;
if (DATA.mode === '2d') {
  const scale = Math.min(W / DATA.width, H / DATA.height);
  const offsetX = (W - DATA.width * scale) / 2;
  const offsetY = (H - DATA.height * scale) / 2;
  project = (x, y) => [offsetX + x * scale, offsetY + y * scale];
} else {
  const [minX, maxX, minY, maxY] = DATA.bounds;
  const scale = Math.min(W / (maxX - minX), H / (maxY - minY));
  const offsetX = (W - (maxX - minX) * scale) / 2;
  const offsetY = (H - (maxY - minY) * scale) / 2;
  project = (x, y) => [offsetX + (x - minX) * scale, H - (offsetY + (y - minY) * scale)];
}

// Precompute trails.
const trails = new Map();
DATA.frames.forEach((frame, index) => {
  frame.tracks.forEach((track) => {
    let arr = trails.get(track.id);
    if (!arr) { arr = new Array(DATA.frames.length).fill(null); trails.set(track.id, arr); }
    if (DATA.mode === '2d') {
      arr[index] = [(track.bbox[0] + track.bbox[2]) / 2, (track.bbox[1] + track.bbox[3]) / 2];
    } else {
      arr[index] = [track.p[0], track.p[1]];
    }
  });
});

function draw2d(frame) {
  ctx.strokeStyle = '#1f2937';
  ctx.lineWidth = 1;
  ctx.strokeRect(project(0, 0)[0], project(0, 0)[1], DATA.width * (project(1, 0)[0] - project(0, 0)[0]), DATA.height * (project(0, 1)[1] - project(0, 0)[1]));

  if (showTruth.checked) {
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.5;
    frame.truth.forEach((truth) => {
      ctx.strokeStyle = truth.detected ? '#94a3b8' : '#facc15';
      const [x0, y0] = project(truth.bbox[0], truth.bbox[1]);
      const [x1, y1] = project(truth.bbox[2], truth.bbox[3]);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.fillStyle = '#cbd5e1';
      ctx.font = '11px system-ui';
      ctx.fillText('gt' + truth.id, x0, y0 - 2);
    });
    ctx.setLineDash([]);
  }

  frame.obs.forEach((observation) => {
    ctx.strokeStyle = sensorColor(observation.s);
    ctx.lineWidth = 1.5;
    const [x0, y0] = project(observation.bbox[0], observation.bbox[1]);
    const [x1, y1] = project(observation.bbox[2], observation.bbox[3]);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  });

  frame.tracks.forEach((track) => {
    const color = LIFECYCLE_COLORS[track.state] || '#e2e8f0';
    ctx.strokeStyle = color;
    ctx.lineWidth = track.pred ? 1.5 : 2.5;
    ctx.setLineDash(track.pred ? [4, 3] : []);
    const [x0, y0] = project(track.bbox[0], track.bbox[1]);
    const [x1, y1] = project(track.bbox[2], track.bbox[3]);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    ctx.font = 'bold 12px system-ui';
    ctx.fillText('#' + track.id, x0, y1 + 12);
  });
}

function draw3d(frame) {
  // Grid every 5 m.
  ctx.strokeStyle = '#1f2937';
  ctx.lineWidth = 1;
  const [minX, maxX, minY, maxY] = DATA.bounds;
  for (let x = Math.floor(minX / 5) * 5; x <= maxX; x += 5) {
    const [sx, syTop] = project(x, minY);
    const [, syBottom] = project(x, maxY);
    ctx.beginPath(); ctx.moveTo(sx, syTop); ctx.lineTo(sx, syBottom); ctx.stroke();
  }
  for (let y = Math.floor(minY / 5) * 5; y <= maxY; y += 5) {
    const [sxLeft, sy] = project(minX, y);
    const [sxRight] = project(maxX, y);
    ctx.beginPath(); ctx.moveTo(sxLeft, sy); ctx.lineTo(sxRight, sy); ctx.stroke();
  }

  if (showTruth.checked) {
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.5;
    frame.truth.forEach((truth) => {
      ctx.strokeStyle = truth.detected ? '#94a3b8' : '#facc15';
      const [sx, sy] = project(truth.p[0], truth.p[1]);
      ctx.beginPath(); ctx.arc(sx, sy, 11, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = '#cbd5e1';
      ctx.font = '11px system-ui';
      ctx.fillText('gt' + truth.id, sx + 12, sy - 6);
    });
    ctx.setLineDash([]);
  }

  frame.obs.forEach((observation) => {
    const [sx, sy] = project(observation.p[0], observation.p[1]);
    ctx.fillStyle = sensorColor(observation.s);
    ctx.beginPath(); ctx.arc(sx, sy, 3, 0, Math.PI * 2); ctx.fill();
  });

  frame.tracks.forEach((track) => {
    const color = LIFECYCLE_COLORS[track.state] || '#e2e8f0';
    const [sx, sy] = project(track.p[0], track.p[1]);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2);
    if (track.pred) { ctx.stroke(); } else { ctx.fill(); }
    // Velocity arrow.
    const [vx, vy] = project(track.p[0] + track.v[0], track.p[1] + track.v[1]);
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(vx, vy); ctx.stroke();
    ctx.font = 'bold 12px system-ui';
    ctx.fillText('#' + track.id, sx + 9, sy + 4);
  });
}

function drawTrails(frame) {
  for (const [id, arr] of trails) {
    const active = frame.tracks.find((track) => track.id === id);
    const color = active ? (LIFECYCLE_COLORS[active.state] || '#e2e8f0') : '#475569';
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i <= frameIndex; i += 1) {
      const point = arr[i];
      if (!point) continue;
      const [sx, sy] = project(point[0], point[1]);
      if (!started) { ctx.moveTo(sx, sy); started = true; } else { ctx.lineTo(sx, sy); }
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
}

function draw() {
  ctx.clearRect(0, 0, W, H);
  const frame = DATA.frames[frameIndex];
  if (!frame) return;
  if (DATA.mode === '2d') { drawTrails(frame); draw2d(frame); } else { drawTrails(frame); draw3d(frame); }
  const seconds = (frame.ts / 1e9).toFixed(2);
  info.textContent = 'frame ' + frameIndex + '/' + (DATA.frames.length - 1) + '  t=' + seconds + 's  objects=' + frame.tracks.length;
  slider.value = String(frameIndex);
}

function tick(now) {
  const delta = now - lastTime;
  lastTime = now;
  if (playing) {
    accumulator += delta;
    const frameDuration = 1000 / 30;
    while (accumulator >= frameDuration) {
      accumulator -= frameDuration;
      if (frameIndex < DATA.frames.length - 1) { frameIndex += 1; } else { playing = false; playButton.textContent = 'Play'; }
    }
    draw();
  }
  requestAnimationFrame(tick);
}

playButton.addEventListener('click', () => {
  playing = !playing;
  playButton.textContent = playing ? 'Pause' : 'Play';
});
document.getElementById('prev').addEventListener('click', () => { frameIndex = Math.max(0, frameIndex - 1); draw(); });
document.getElementById('next').addEventListener('click', () => { frameIndex = Math.min(DATA.frames.length - 1, frameIndex + 1); draw(); });
slider.addEventListener('input', () => { frameIndex = Number(slider.value); draw(); });
showTruth.addEventListener('change', draw);

document.getElementById('title').textContent = DATA.title;
document.getElementById('note').textContent = DATA.note;
draw();
requestAnimationFrame(tick);
</script>
</body>
</html>
"#;
