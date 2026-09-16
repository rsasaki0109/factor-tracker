//! Self-contained HTML report for camera-assisted LiDAR tracking.
//!
//! The report embeds all data inline and renders with plain JavaScript so it
//! can be opened from the file system. The top-down view is a 2D XY
//! projection, not a 3D viewer. Real camera images are not embedded; the
//! camera panel is drawn on a blank canvas sized to the calibrated image.

use std::path::Path;

use perception_engine::engine::{CameraFrameOutput, LidarFrameOutput};
use perception_engine::eval::{GroundTruthFrame, ModeMetrics};
use perception_engine::geometry::Calibration;
use perception_engine::{AllCounters, DiagnosticEvent, EngineResult};
use serde::Serialize;

const NOTE: &str =
    "Synthetic data unless real KITTI files were prepared. Not validated on real sensors. The 3D view is a top-down (XY) projection, not a 3D rendering.";

#[derive(Serialize)]
struct ReportData {
    title: String,
    note: String,
    image_width: u32,
    image_height: u32,
    modes: Vec<ModeMetrics>,
    counters: AllCounters,
    frames: Vec<FrameData>,
    diagnostics: Vec<DiagnosticData>,
}

#[derive(Serialize)]
struct FrameData {
    ts: i64,
    observations: Vec<[f64; 3]>,
    tracks: Vec<TrackData>,
    associations: Vec<AssocData>,
    camera: Vec<CameraItem>,
    camera_rejections: Vec<RejectionData>,
    truth: Vec<[f64; 4]>,
}

#[derive(Serialize)]
struct TrackData {
    id: u64,
    lifecycle: String,
    p: [f64; 3],
    pred: bool,
    age_ns: i64,
}

#[derive(Serialize)]
struct AssocData {
    index: usize,
    track: Option<u64>,
    outcome: String,
}

#[derive(Serialize)]
struct CameraItem {
    detection: [f64; 4],
    projected: Option<[f64; 4]>,
    track: Option<u64>,
    accepted: bool,
    reason: String,
    best_iou: f64,
}

#[derive(Serialize)]
struct RejectionData {
    track: u64,
    reason: String,
}

#[derive(Serialize)]
struct DiagnosticData {
    arrival_ns: i64,
    event_ns: Option<i64>,
    code: String,
    level: String,
    message: String,
}

/// Renders the report to `path`.
#[allow(clippy::too_many_arguments)]
pub fn write_report(
    path: &Path,
    title: &str,
    calibration: &Calibration,
    modes: &[ModeMetrics],
    lidar_frames: &[LidarFrameOutput],
    camera_frames: &[CameraFrameOutput],
    truth: &[GroundTruthFrame],
    counters: &AllCounters,
    diagnostics: &[DiagnosticEvent],
) -> EngineResult<()> {
    let camera_by_time: std::collections::BTreeMap<i64, &CameraFrameOutput> = camera_frames
        .iter()
        .map(|frame| (frame.timestamp_ns, frame))
        .collect();
    let truth_by_time: std::collections::BTreeMap<i64, &GroundTruthFrame> = truth
        .iter()
        .map(|frame| (frame.timestamp_ns, frame))
        .collect();

    let frames: Vec<FrameData> = lidar_frames
        .iter()
        .map(|frame| {
            let observations = frame.observations.iter().filter_map(|p| vec3(p)).collect();
            let tracks = frame
                .tracks
                .iter()
                .filter_map(|track| {
                    vec3(&track.position).map(|p| TrackData {
                        id: track.track_id,
                        lifecycle: track.lifecycle.clone(),
                        p,
                        pred: track.prediction_only,
                        age_ns: track.age_since_spatial_update_ns,
                    })
                })
                .collect();
            let associations = frame
                .associations
                .iter()
                .map(|a| AssocData {
                    index: a.observation_index,
                    track: a.track_id,
                    outcome: a.outcome.clone(),
                })
                .collect();
            let (camera, camera_rejections) = match camera_by_time.get(&frame.timestamp_ns) {
                Some(camera_frame) => (
                    camera_frame
                        .matches
                        .iter()
                        .filter_map(|m| {
                            Some(CameraItem {
                                detection: vec4(&m.detection_bbox)?,
                                projected: m.projected_bbox.as_ref().and_then(|p| vec4(p)),
                                track: m.track_id,
                                accepted: m.accepted,
                                reason: m.reason.clone(),
                                best_iou: m.best_iou,
                            })
                        })
                        .collect(),
                    camera_frame
                        .projection_rejections
                        .iter()
                        .map(|r| RejectionData {
                            track: r.track_id,
                            reason: r.reason.clone(),
                        })
                        .collect(),
                ),
                None => (Vec::new(), Vec::new()),
            };
            let truth = truth_by_time
                .get(&frame.timestamp_ns)
                .map(|truth_frame| {
                    truth_frame
                        .objects
                        .iter()
                        .filter_map(|object| {
                            let p = vec3(&object.position)?;
                            Some([p[0], p[1], p[2], object.class as f64])
                        })
                        .collect()
                })
                .unwrap_or_default();
            FrameData {
                ts: frame.timestamp_ns,
                observations,
                tracks,
                associations,
                camera,
                camera_rejections,
                truth,
            }
        })
        .collect();

    let diagnostics = diagnostics
        .iter()
        .map(|event| DiagnosticData {
            arrival_ns: event.arrival_time.as_nanos(),
            event_ns: event.event_time.map(|t| t.as_nanos()),
            code: event.code.to_string(),
            level: format!("{:?}", event.level),
            message: event.message.clone(),
        })
        .collect();

    let data = ReportData {
        title: title.to_string(),
        note: NOTE.to_string(),
        image_width: calibration.intrinsics.image_width,
        image_height: calibration.intrinsics.image_height,
        modes: modes.to_vec(),
        counters: counters.clone(),
        frames,
        diagnostics,
    };
    let json = serde_json::to_string(&data)?;
    let html = TEMPLATE.replace("__DATA__", &json);
    std::fs::write(path, html)?;
    Ok(())
}

fn vec3(values: &[f64]) -> Option<[f64; 3]> {
    if values.len() != 3 {
        return None;
    }
    Some([values[0], values[1], values[2]])
}

fn vec4(values: &[f64]) -> Option<[f64; 4]> {
    if values.len() != 4 {
        return None;
    }
    Some([values[0], values[1], values[2], values[3]])
}

const TEMPLATE: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>perception-engine report</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #0f172a; color: #e2e8f0; font-family: system-ui, sans-serif; }
  #app { max-width: 1220px; margin: 0 auto; padding: 16px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  h2 { font-size: 15px; margin: 18px 0 8px; color: #cbd5e1; }
  #note { font-size: 12px; color: #94a3b8; margin-bottom: 10px; }
  .panels { display: flex; gap: 14px; flex-wrap: wrap; }
  .panelbox { flex: 1; min-width: 340px; }
  canvas { width: 100%; height: auto; background: #111827; border: 1px solid #1f2937; border-radius: 8px; display: block; }
  table { border-collapse: collapse; width: 100%; font-size: 12px; }
  th, td { border: 1px solid #1f2937; padding: 4px 6px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  #controls { display: flex; gap: 10px; align-items: center; margin: 10px 0; }
  input[type=range] { flex: 1; }
  #info { font-variant-numeric: tabular-nums; font-size: 13px; }
  .diag { font-size: 12px; max-height: 220px; overflow: auto; border: 1px solid #1f2937; border-radius: 6px; padding: 6px; }
  .diag div { border-bottom: 1px solid #16213a; padding: 2px 0; }
  .warn { color: #fbbf24; }
  .info { color: #94a3b8; }
  .legend { font-size: 12px; color: #94a3b8; margin-top: 6px; }
  .sw { display:inline-block; width:12px; height:9px; border-radius:2px; margin-right:4px; }
</style>
</head>
<body>
<div id="app">
  <h1 id="title"></h1>
  <div id="note"></div>
  <h2>Mode comparison (same input)</h2>
  <table id="modes"></table>
  <h2>Frame diagnostics</h2>
  <div id="controls">
    <button id="prev">&lt;</button>
    <input id="slider" type="range" min="0" max="0" value="0">
    <button id="next">&gt;</button>
    <span id="info"></span>
  </div>
  <div class="panels">
    <div class="panelbox">
      <div class="legend">Camera view: detections (blue) and projected tracks (green dashed)</div>
      <canvas id="camera"></canvas>
      <div class="legend" id="camera-diag"></div>
    </div>
    <div class="panelbox">
      <div class="legend">Top-down XY: LiDAR observations (dots) and tracks (lifecycle colour)</div>
      <canvas id="topdown"></canvas>
      <div class="legend">lifecycle: <span class="sw" style="background:#e6a23c"></span>tentative
        <span class="sw" style="background:#2eb872"></span>confirmed
        <span class="sw" style="background:#e04f5f"></span>lost</div>
    </div>
  </div>
  <h2>Engine diagnostics</h2>
  <div class="diag" id="diag"></div>
</div>
<script>
const DATA = __DATA__;
const LIFECYCLE = { tentative: '#e6a23c', confirmed: '#2eb872', lost: '#e04f5f' };
const slider = document.getElementById('slider');
const info = document.getElementById('info');
slider.max = String(Math.max(0, DATA.frames.length - 1));
let index = 0;

function showModes() {
  const rows = ['<tr><th>mode</th><th>frames</th><th>coverage</th><th>pos RMSE</th><th>mean</th><th>p95</th><th>ID switches</th><th>class acc</th><th>cam assoc</th><th>matches</th><th>rejected</th><th>late</th><th>dup</th><th>overflow</th></tr>'];
  for (const m of DATA.modes) {
    const pct = v => v === null || v === undefined ? 'n/a' : (v * 100).toFixed(1) + '%';
    const num = v => v === null || v === undefined ? 'n/a' : v.toFixed(3);
    const cls = m.class_total > 0 ? (m.class_correct / m.class_total) : null;
    rows.push(`<tr><td>${m.mode}</td><td>${m.frames_evaluated}</td><td>${pct(m.track_coverage)}</td>` +
      `<td>${num(m.position_rmse)}</td><td>${num(m.position_mean_abs)}</td><td>${num(m.position_p95_abs)}</td>` +
      `<td>${m.id_switches}</td><td>${cls === null ? 'n/a' : (cls*100).toFixed(1)+'%'}</td>` +
      `<td>${m.camera_assoc_accuracy === null ? 'n/a' : (m.camera_assoc_accuracy*100).toFixed(1)+'%'}</td>` +
      `<td>${m.counters.engine.camera_matches}</td><td>${m.counters.engine.association_rejected}</td>` +
      `<td>${m.counters.reorder.late}</td><td>${m.counters.reorder.duplicate}</td><td>${m.counters.reorder.overflow}</td></tr>`);
  }
  document.getElementById('modes').innerHTML = rows.join('');
}

function setupCanvas(id, w, h) {
  const c = document.getElementById(id);
  const scale = c.parentElement.clientWidth / w;
  c.width = w; c.height = h;
  c.style.width = '100%'; c.style.height = (h * scale) + 'px';
  return c;
}

function drawCamera(frame) {
  const W = DATA.image_width, H = DATA.image_height;
  const c = setupCanvas('camera', W, H);
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0b1220'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#1f2937'; ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
  for (const item of frame.camera) {
    ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 1.5;
    const b = item.detection;
    ctx.strokeRect(b[0], b[1], b[2]-b[0], b[3]-b[1]);
    if (item.projected) {
      const p = item.projected;
      ctx.strokeStyle = item.accepted ? '#2eb872' : '#64748b';
      ctx.setLineDash([5, 4]); ctx.lineWidth = 2;
      ctx.strokeRect(p[0], p[1], p[2]-p[0], p[3]-p[1]);
      ctx.setLineDash([]);
      if (item.track !== null && item.track !== undefined) {
        ctx.fillStyle = '#2eb872'; ctx.font = '13px system-ui';
        ctx.fillText('#' + item.track, p[0], p[1] - 3);
      }
    }
  }
}

function drawTopDown(frame) {
  const W = 640, H = 420;
  const c = setupCanvas('topdown', W, H);
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0b1220'; ctx.fillRect(0, 0, W, H);
  const xs = [], ys = [];
  const consider = p => { xs.push(p[0]); ys.push(p[1]); };
  frame.observations.forEach(consider);
  frame.tracks.forEach(t => consider(t.p));
  frame.truth.forEach(t => consider([t[0], t[1]]));
  if (xs.length === 0) return;
  let minx = Math.min(...xs), maxx = Math.max(...xs), miny = Math.min(...ys), maxy = Math.max(...ys);
  const padx = (maxx - minx) * 0.15 + 1, pady = (maxy - miny) * 0.15 + 1;
  minx -= padx; maxx += padx; miny -= pady; maxy += pady;
  const sx = W / (maxx - minx), sy = H / (maxy - miny);
  const s = Math.min(sx, sy);
  const ox = (W - (maxx - minx) * s) / 2, oy = (H - (maxy - miny) * s) / 2;
  const P = (x, y) => [ox + (x - minx) * s, H - (oy + (y - miny) * s)];
  ctx.strokeStyle = '#1f2937'; ctx.lineWidth = 1;
  for (let x = Math.ceil(minx / 10) * 10; x <= maxx; x += 10) {
    const [px] = P(x, miny); const [, py2] = P(x, maxy);
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
    void py2;
  }
  for (const p of frame.observations) {
    const [px, py] = P(p[0], p[1]);
    ctx.fillStyle = '#3b82f6'; ctx.beginPath(); ctx.arc(px, py, 2.5, 0, Math.PI * 2); ctx.fill();
  }
  for (const t of frame.truth) {
    const [px, py] = P(t[0], t[1]);
    ctx.strokeStyle = '#94a3b8'; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.arc(px, py, 9, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const t of frame.tracks) {
    const [px, py] = P(t.p[0], t.p[1]);
    const color = LIFECYCLE[t.lifecycle] || '#e2e8f0';
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.fill();
    ctx.font = '12px system-ui';
    ctx.fillText('#' + t.id + (t.pred ? ' (pred)' : ''), px + 8, py + 4);
  }
}

function drawDiagnostics() {
  const diag = document.getElementById('diag');
  if (DATA.diagnostics.length === 0) { diag.textContent = 'no diagnostics were recorded'; return; }
  diag.innerHTML = DATA.diagnostics.slice(0, 300).map(d =>
    `<div class="${d.level === 'Warn' ? 'warn' : 'info'}">[${d.code}] arrival=${d.arrival_ns} event=${d.event_ns ?? ''} ${d.message}</div>`
  ).join('');
}

function draw() {
  const frame = DATA.frames[index];
  if (!frame) return;
  drawCamera(frame);
  drawTopDown(frame);
  const accepted = frame.camera.filter(c => c.accepted).length;
  const rejected = frame.camera_rejections.length;
  info.textContent = `frame ${index}/${DATA.frames.length - 1} t=${(frame.ts / 1e9).toFixed(2)}s tracks=${frame.tracks.length} camera accepted=${accepted} projection rejected=${rejected}`;
  const camDiag = frame.camera.filter(c => !c.accepted).map(c => `${c.reason}(iou=${c.best_iou.toFixed(2)})`);
  const rej = frame.camera_rejections.map(r => '#' + r.track + ':' + r.reason);
  document.getElementById('camera-diag').textContent =
    'unmatched detections: ' + (camDiag.length ? camDiag.join(', ') : 'none') +
    ' | projection rejected: ' + (rej.length ? rej.join(', ') : 'none');
  slider.value = String(index);
}

document.getElementById('prev').addEventListener('click', () => { index = Math.max(0, index - 1); draw(); });
document.getElementById('next').addEventListener('click', () => { index = Math.min(DATA.frames.length - 1, index + 1); draw(); });
slider.addEventListener('input', () => { index = Number(slider.value); draw(); });

document.getElementById('title').textContent = DATA.title;
document.getElementById('note').textContent = DATA.note;
showModes();
drawDiagnostics();
draw();
</script>
</body>
</html>
"#;
