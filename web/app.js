// Camera -> MediaPipe person detector -> factor-tracker 2D tracker (wasm) -> overlay.
import { FilesetResolver, ObjectDetector } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { loadTracker } from "./tracker.js";

const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite";

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const status = (text) => { $("status").textContent = text; };

let detector = null;
let tracker = null;
let backend = "";
let stream = null;
let facing = "environment";
let showDetections = false;
let running = false;
let lastVideoTime = -1;
let fps = { frames: 0, since: performance.now() };

async function createDetector(delegate) {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  return ObjectDetector.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: "VIDEO",
    scoreThreshold: 0.3,
    maxResults: 25,
    categoryAllowlist: ["person"],
  });
}

async function init() {
  try {
    tracker = await loadTracker("factor_tracker_wasm.wasm");
  } catch (error) {
    status(`Could not load the tracker: ${error.message}`);
    return;
  }
  for (const delegate of ["GPU", "CPU"]) {
    try {
      detector = await createDetector(delegate);
      backend = delegate;
      break;
    } catch (error) {
      console.warn(`detector with ${delegate} failed`, error);
    }
  }
  if (!detector) {
    status("Could not start the detector on this browser.");
    return;
  }
  $("s-backend").textContent = backend;
  status("Ready.");
}

function stopStream() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    status("This browser does not give pages camera access (it needs HTTPS).");
    return;
  }
  stopStream();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (error) {
    status(`Camera not available: ${error.message}`);
    return;
  }
  video.removeAttribute("src");
  video.srcObject = stream;
  video.loop = false;
  video.classList.toggle("mirror", facing === "user");
  $("flip").disabled = false;
  await begin();
}

async function startFile(file) {
  stopStream();
  video.srcObject = null;
  video.src = URL.createObjectURL(file);
  video.loop = true;
  video.classList.remove("mirror");
  $("flip").disabled = true;
  await begin();
}

async function begin() {
  await video.play();
  tracker.reset();
  lastVideoTime = -1;
  $("placeholder").hidden = true;
  status(detector ? "Tracking." : "Detector still loading…");
  if (!running) {
    running = true;
    schedule();
  }
}

function schedule() {
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback(step);
  } else {
    requestAnimationFrame(step);
  }
}

function step() {
  if (detector && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    process(performance.now());
  }
  schedule();
}

function process(now) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    $("stage").style.aspectRatio = `${w} / ${h}`;
  }
  const t0 = performance.now();
  const result = detector.detectForVideo(video, now);
  const t1 = performance.now();
  const detections = result.detections.map((d) => ({
    x: d.boundingBox.originX, y: d.boundingBox.originY,
    w: d.boundingBox.width, h: d.boundingBox.height,
    score: d.categories[0]?.score ?? 0,
  }));
  const tracks = tracker.step(now, detections);
  const t2 = performance.now();
  draw(detections, tracks, w);

  fps.frames += 1;
  if (now - fps.since > 1000) {
    $("s-fps").textContent = (fps.frames * 1000 / (now - fps.since)).toFixed(1);
    fps = { frames: 0, since: now };
  }
  $("s-det").textContent = (t1 - t0).toFixed(1);
  $("s-trk").textContent = (t2 - t1).toFixed(2);
  $("s-dets").textContent = detections.length;
  $("s-tracks").textContent = tracks.filter((t) => t.state === 1).length;
}

function colour(id) {
  return `hsl(${(id * 137.508) % 360} 85% 55%)`;
}

function draw(detections, tracks, width) {
  const mirror = video.classList.contains("mirror");
  const mx = (x, w) => (mirror ? width - x - w : x);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const line = Math.max(2, canvas.height / 240);
  if (showDetections) {
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    for (const d of detections) ctx.strokeRect(mx(d.x, d.w), d.y, d.w, d.h);
  }
  ctx.font = `600 ${Math.round(line * 7)}px system-ui, sans-serif`;
  ctx.textBaseline = "bottom";
  for (const t of tracks) {
    const confirmed = t.state === 1 && t.updated;
    if (t.state === 0 && !showDetections) continue;
    const x = mx(t.cx - t.w / 2, t.w);
    const y = t.cy - t.h / 2;
    ctx.strokeStyle = colour(t.id);
    ctx.lineWidth = confirmed ? line : line / 2;
    ctx.setLineDash(confirmed ? [] : [line * 3, line * 2]);
    ctx.strokeRect(x, y, t.w, t.h);
    if (t.state !== 0) {
      const label = `#${t.id}`;
      const pad = line * 2;
      const tw = ctx.measureText(label).width + pad * 2;
      const th = line * 9;
      ctx.fillStyle = colour(t.id);
      ctx.fillRect(x, Math.max(0, y - th), tw, th);
      ctx.fillStyle = "#0b0d10";
      ctx.fillText(label, x + pad, Math.max(th, y) - line);
    }
  }
}

$("start").addEventListener("click", () => startCamera());
$("flip").addEventListener("click", () => {
  facing = facing === "environment" ? "user" : "environment";
  startCamera();
});
$("file").addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) startFile(file);
});
$("boxes").addEventListener("click", (event) => {
  showDetections = !showDetections;
  event.currentTarget.setAttribute("aria-pressed", String(showDetections));
  event.currentTarget.textContent = showDetections ? "Hide detections" : "Show detections";
});

init();
