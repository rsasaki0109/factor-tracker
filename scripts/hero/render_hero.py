#!/usr/bin/env python3
"""Renders the README hero GIF: engine tracks over real KITTI camera frames
and LiDAR scans.

Inputs:
  <results>/<seq>/  files written by scripts/kitti/evaluate_sequences.sh
                    (setup.json, inputs.jsonl, camera_assisted.outputs.jsonl)
  <kitti-root>      training/image_02/<seq>/*.png and training/velodyne/<seq>/*.bin
                    for the rendered frames (scripts/kitti/fetch_frames.py)

usage: render_hero.py <results/<seq>> <kitti-root> <out.gif> --start 160 --frames 100
Requires NumPy, Pillow and ffmpeg.

The README hero was produced with:

    python3 scripts/kitti/fetch_frames.py --kitti-root kitti --sequence 0019 --first 150 --last 280
    python3 scripts/runpod/detect_camera_yolo.py --kitti-root kitti --out det --sequences 0019
    python3 scripts/kitti/convert_tracking.py --kitti-root kitti --out seq --sequences 0019 \
        --oracle --detections-root det          # LiDAR: labels, camera: YOLO11x
    scripts/kitti/evaluate_sequences.sh seq res 0019
    python3 scripts/hero/render_hero.py res/0019 kitti assets/hero.gif --start 160 --frames 100
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W = 1000
MARGIN = 32
SURFACE = (11, 18, 32)
PANEL = (15, 23, 42)
BORDER = (30, 41, 59)
GRID = (30, 41, 62)
TEXT = (241, 245, 249)
TEXT_2 = (148, 163, 184)
TEXT_3 = (100, 116, 139)
# Validated categorical slots (dark): vehicle, pedestrian, cyclist.
CLASS_COLOR = {"vehicle": (57, 135, 229), "pedestrian": (217, 89, 38), "cyclist": (25, 158, 112)}
CLASS_GROUP = {1: "vehicle", 4: "vehicle", 5: "vehicle", 7: "vehicle", 8: "vehicle", 2: "pedestrian", 6: "pedestrian", 3: "cyclist"}
# Box edges for corners ordered as Box3DShape: 0-3 bottom, 4-7 top.
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

FONT_DIR = Path("/usr/share/fonts/opentype/inter")


def font(size: int, weight: str = "Regular"):
    path = FONT_DIR / f"Inter-{weight}.otf"
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


F_TITLE, F_SUB, F_LABEL, F_SMALL, F_TAG = font(28, "Bold"), font(14), font(12, "SemiBold"), font(12), font(11, "SemiBold")


def mat4(values):
    return np.array(values, dtype=float).reshape(4, 4)


def group_of(class_id):
    return CLASS_GROUP.get(class_id, "vehicle")


def tint(color, alpha, base=PANEL):
    return tuple(int(base[i] + (color[i] - base[i]) * alpha) for i in range(3))


def load(seq_dir: Path):
    setup = json.loads((seq_dir / "setup.json").read_text())
    poses = {p["timestamp_ns"]: mat4(p["transform"]) for p in setup["poses"]}
    lidar_inputs, camera_inputs = {}, {}
    for line in open(seq_dir / "inputs.jsonl"):
        record = json.loads(line)
        if record["type"] == "lidar_batch":
            lidar_inputs[record["event_time_ns"]] = record["detections"]
        elif record["type"] == "camera_batch":
            camera_inputs[record["event_time_ns"]] = record["detections"]
    lidar_frames, camera_frames = {}, {}
    for line in open(seq_dir / "camera_assisted.outputs.jsonl"):
        record = json.loads(line)
        if record["type"] == "lidar_frame":
            lidar_frames[record["timestamp_ns"]] = record
        elif record["type"] == "camera_frame":
            camera_frames[record["timestamp_ns"]] = record
    return setup, poses, lidar_inputs, camera_inputs, lidar_frames, camera_frames


class Bev:
    """Ego-centric bird's-eye view: velodyne x forward (right), y left (up)."""

    def __init__(self, box, forward):
        self.x0, self.y0, self.x1, self.y1 = box
        self.forward = forward
        self.scale = (self.x1 - self.x0) / (forward[1] - forward[0])
        self.cy = (self.y0 + self.y1) / 2

    def px(self, x, y):
        return (self.x0 + (x - self.forward[0]) * self.scale, self.cy - y * self.scale)

    def inside(self, x, y, margin=3):
        u, v = self.px(x, y)
        return self.x0 + margin < u < self.x1 - margin and self.y0 + margin < v < self.y1 - margin


class Renderer:
    def __init__(self, data, args):
        self.setup, self.poses, self.lidar_inputs, self.camera_inputs, self.lidar_frames, self.camera_frames = data
        self.args = args
        intr = self.setup["intrinsics"]
        self.intr = intr
        self.ref_to_cam = np.linalg.inv(mat4(self.setup["camera_to_reference"]))
        self.cam_w = W - 2 * MARGIN
        self.sx = self.cam_w / intr["image_width"]
        self.cam_box = (MARGIN, 84, MARGIN + self.cam_w, 84 + round(intr["image_height"] * self.sx))
        self.bev_box = (MARGIN, self.cam_box[3] + 12, W - MARGIN, self.cam_box[3] + 12 + args.bev_height)
        self.height = self.bev_box[3] + 40

    def project(self, points_velo):
        cam = (self.ref_to_cam[:3, :3] @ points_velo.T).T + self.ref_to_cam[:3, 3]
        if np.any(cam[:, 2] < 0.5):
            return None
        u = self.intr["fx"] * cam[:, 0] / cam[:, 2] + self.intr["skew"] * cam[:, 1] / cam[:, 2] + self.intr["cx"]
        v = self.intr["fy"] * cam[:, 1] / cam[:, 2] + self.intr["cy"]
        return np.stack([self.cam_box[0] + u * self.sx, self.cam_box[1] + v * self.sx], axis=1)

    def frame(self, t, frame_index, trails, camera_hits):
        img = Image.new("RGB", (W, self.height), SURFACE)
        d = ImageDraw.Draw(img)
        d.text((MARGIN, 18), "perception-engine", font=F_TITLE, fill=TEXT)
        d.text((MARGIN + 262, 30), "camera-assisted LiDAR tracking in Rust", font=F_SUB, fill=TEXT_2)
        subtitle = "KITTI tracking 0019 · LiDAR: ground-truth boxes as oracle detections · camera: YOLO11x · tracks: perception-engine"
        d.text((MARGIN, 56), subtitle, font=F_SMALL, fill=TEXT_3)

        lidar = self.lidar_frames[t]
        tracks = {tr["track_id"]: tr for tr in lidar["tracks"] if tr["lifecycle"] == "confirmed"}
        world_to_velo = np.linalg.inv(self.poses[t])

        def to_velo(p):
            return world_to_velo[:3, :3] @ np.asarray(p) + world_to_velo[:3, 3]

        # Shape of each track from its associated LiDAR detection this frame.
        detections = self.lidar_inputs.get(t, [])
        shapes = {}
        for assoc in lidar["associations"]:
            if assoc["track_id"] in tracks and assoc["observation_index"] is not None:
                det = detections[assoc["observation_index"]]
                if det.get("shape") and det["shape"].get("corners"):
                    shapes[assoc["track_id"]] = np.array(det["shape"]["corners"]).reshape(8, 3)

        # ---- camera panel -------------------------------------------------
        image = Image.open(self.args.kitti_root / "training" / "image_02" / self.args.sequence / f"{frame_index:06d}.png")
        image = image.convert("RGB").resize((self.cam_box[2] - self.cam_box[0], self.cam_box[3] - self.cam_box[1]), Image.LANCZOS)
        image = Image.blend(image, Image.new("RGB", image.size, SURFACE), 0.18)
        img.paste(image, self.cam_box[:2])
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        o = ImageDraw.Draw(overlay)
        for det in self.camera_inputs.get(t, []):
            x1, y1, x2, y2 = (v * self.sx for v in det["bbox"])
            o.rectangle((self.cam_box[0] + x1, self.cam_box[1] + y1, self.cam_box[0] + x2, self.cam_box[1] + y2),
                        outline=(241, 245, 249, 150), width=1)
        tags = []
        for tid, track in tracks.items():
            if tid not in shapes:
                continue
            center = to_velo(track["position"])
            pts = self.project(shapes[tid] + center)
            if pts is None:
                continue
            color = CLASS_COLOR[group_of(track["class_label"])]
            for a, b in EDGES:
                o.line((*pts[a], *pts[b]), fill=color + (255,), width=2)
            tags.append((pts[:, 0].min(), pts[:, 1].min(), tid, color, camera_hits.get(tid) == t))
        region = overlay.crop(self.cam_box)
        img.paste(region, self.cam_box[:2], region)
        d = ImageDraw.Draw(img)
        for u, v, tid, color, seen in tags:
            label = f"#{tid}"
            tw = d.textlength(label, font=F_LABEL)
            v = max(v - 17, self.cam_box[1] + 2)
            u = min(max(u, self.cam_box[0] + 2), self.cam_box[2] - tw - 10)
            d.rectangle((u, v, u + tw + 8, v + 15), fill=color)
            d.text((u + 4, v + 1), label, font=F_LABEL, fill=SURFACE)
        d.rectangle(self.cam_box, outline=BORDER)

        # ---- bird's-eye panel ---------------------------------------------
        d.rounded_rectangle(self.bev_box, 10, fill=PANEL, outline=BORDER)
        bev = Bev((self.bev_box[0] + 1, self.bev_box[1] + 1, self.bev_box[2] - 1, self.bev_box[3] - 1), self.args.forward)
        for r in range(10, int(self.args.forward[1]), 10):
            u, _ = bev.px(r, 0)
            d.line((u, self.bev_box[1] + 24, u, self.bev_box[3] - 28), fill=GRID)
            d.text((u + 4, self.bev_box[3] - 44), f"{r} m", font=F_SMALL, fill=TEXT_3)
        ruler = [(bev.px(r, 0)[0] + 4, self.bev_box[3] - 44, f"{r} m") for r in range(10, int(self.args.forward[1]), 10)]
        scan = np.fromfile(self.args.kitti_root / "training" / "velodyne" / self.args.sequence / f"{frame_index:06d}.bin",
                           dtype=np.float32).reshape(-1, 4)
        keep = (scan[:, 0] > self.args.forward[0]) & (scan[:, 0] < self.args.forward[1]) & (np.abs(scan[:, 1]) < 30)
        scan = scan[keep]
        u = (self.bev_box[0] + 1 + (scan[:, 0] - self.args.forward[0]) * bev.scale).astype(int)
        v = (bev.cy - scan[:, 1] * bev.scale).astype(int)
        inside = (u > self.bev_box[0] + 2) & (u < self.bev_box[2] - 2) & (v > self.bev_box[1] + 2) & (v < self.bev_box[3] - 2)
        # Height above the road (sensor at ~1.73 m): ground dark, obstacles light.
        height = np.clip((scan[:, 2] + 1.73) / 2.5, 0.0, 1.0)
        pixels = np.asarray(img).copy()
        shade = (np.array(PANEL) + (np.array([203, 213, 225]) - np.array(PANEL)) * (0.18 + 0.6 * height[:, None])).astype(np.uint8)
        pixels[v[inside], u[inside]] = shade[inside]
        img = Image.fromarray(pixels)
        d = ImageDraw.Draw(img)
        d.rectangle((self.bev_box[0] + 6, self.bev_box[1] + 5, self.bev_box[0] + 590, self.bev_box[1] + 22), fill=PANEL)
        d.text((self.bev_box[0] + 12, self.bev_box[1] + 8), "BIRD'S-EYE VIEW · Velodyne scan, LiDAR tracks in the world frame drawn around the ego vehicle",
               font=F_TAG, fill=TEXT_3)

        for u_text, v_text, text in ruler:
            tw = d.textlength(text, font=F_SMALL)
            d.rectangle((u_text - 2, v_text, u_text + tw + 2, v_text + 15), fill=PANEL)
            d.text((u_text, v_text), text, font=F_SMALL, fill=TEXT_3)
        for tid, history in trails.items():
            if tid not in tracks or len(history) < 2:
                continue
            color = CLASS_COLOR[group_of(tracks[tid]["class_label"])]
            pts = [to_velo(p) for p in history]
            for k in range(1, len(pts)):
                a, b = pts[k - 1], pts[k]
                if bev.inside(*a[:2]) and bev.inside(*b[:2]):
                    d.line((*bev.px(*a[:2]), *bev.px(*b[:2])), fill=tint(color, 0.2 + 0.7 * k / len(pts)), width=2)
        labels = []
        for tid, track in tracks.items():
            center = to_velo(track["position"])
            if not bev.inside(*center[:2]):
                continue
            color = CLASS_COLOR[group_of(track["class_label"])]
            if tid in shapes:
                poly = [bev.px(*(c + center)[:2]) for c in shapes[tid][:4]]
                d.polygon(poly, fill=tint(color, 0.45 if camera_hits.get(tid) == t else 0.2), outline=color)
                top = min(p[1] for p in poly)
            else:
                top = bev.px(*center[:2])[1]
            cov = np.array(track["covariance"]).reshape(6, 6)[:2, :2]
            radius = 2 * math.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0)) * bev.scale
            cu, cv = bev.px(*center[:2])
            d.ellipse((cu - radius, cv - radius, cu + radius, cv + radius), outline=tint(color, 0.8))
            labels.append((cu, top - 15, f"#{tid}"))
        placed = []
        for u, v, text in sorted(labels, key=lambda item: item[1]):
            tw = d.textlength(text, font=F_LABEL)
            u -= tw / 2
            while any(abs(u - pu) < max(tw, pw) + 2 and abs(v - pv) < 13 for pu, pv, pw in placed):
                v -= 13
            placed.append((u, v, tw))
            d.text((u, v), text, font=F_LABEL, fill=TEXT)
        ego = [bev.px(1.6, 0.85), bev.px(1.6, -0.85), bev.px(-2.6, -0.85), bev.px(-2.6, 0.85)]
        d.polygon(ego, fill=TEXT)
        d.polygon([bev.px(1.6, 0.85), bev.px(1.6, -0.85), bev.px(2.6, 0)], fill=TEXT)

        lx, ly = self.bev_box[0] + 12, self.bev_box[3] - 22
        d.rectangle((self.bev_box[0] + 6, ly - 3, self.bev_box[2] - 6, ly + 16), fill=PANEL)
        for name in ("vehicle", "pedestrian", "cyclist"):
            d.rectangle((lx, ly + 3, lx + 10, ly + 13), fill=CLASS_COLOR[name])
            d.text((lx + 15, ly), name, font=F_SMALL, fill=TEXT_2)
            lx += d.textlength(name, font=F_SMALL) + 32
        d.text((lx + 4, ly), "filled = associated with a camera detection this frame · ring = 2σ position uncertainty",
               font=F_SMALL, fill=TEXT_3)

        camera = self.camera_frames.get(t, {})
        matched = sum(1 for m in camera.get("matches", []) if m["accepted"])
        status = f"frame {frame_index:04d} · {len(tracks)} confirmed tracks · {matched} camera associations"
        d.text((MARGIN, self.height - 28), status, font=F_SMALL, fill=TEXT_2)
        attribution = "KITTI images, scans and labels: Geiger et al., CC BY-NC-SA 3.0"
        d.text((W - MARGIN - d.textlength(attribution, font=F_SMALL), self.height - 28), attribution, font=F_SMALL, fill=TEXT_3)
        return img


def encode_gif(frames_dir: Path, out: Path, fps: int, width: int | None, colors: int, dither: str) -> None:
    """Two-pass ffmpeg GIF encode with one global palette."""
    scale = f"scale={width}:-1:flags=lanczos" if width else "null"
    palette = frames_dir / "palette.png"
    pattern = str(frames_dir / "%04d.png")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", pattern,
                    "-vf", f"{scale},palettegen=max_colors={colors}:stats_mode=diff", str(palette)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", pattern, "-i", str(palette),
                    "-lavfi", f"[0:v]{scale}[x];[x][1:v]paletteuse=dither={dither}:diff_mode=rectangle",
                    "-loop", "0", str(out)], check=True)
    palette.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seq_dir", type=Path)
    parser.add_argument("kitti_root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--start", type=int, default=0, help="first rendered frame index")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--trail", type=int, default=20)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--forward", type=float, nargs=2, default=(-5.0, 60.0))
    parser.add_argument("--bev-height", type=int, default=250)
    parser.add_argument("--width", type=int, default=760, help="scale the GIF to this width (0 = no scaling)")
    parser.add_argument("--colors", type=int, default=80)
    parser.add_argument("--dither", default="none", help="ffmpeg paletteuse dither (none, bayer, sierra2_4a, ...)")
    parser.add_argument("--keep-frames", type=Path, help="also copy the rendered PNG frames here")
    parser.add_argument("--still", type=Path, help="also write one frame as PNG")
    parser.add_argument("--still-index", type=int, default=50)
    args = parser.parse_args()
    args.sequence = args.seq_dir.name

    data = load(args.seq_dir)
    renderer = Renderer(data, args)
    lidar_frames, camera_frames = data[4], data[5]
    times = sorted(lidar_frames)
    trails = defaultdict(lambda: deque(maxlen=args.trail))
    camera_hits = {}
    work = Path(tempfile.mkdtemp())
    rendered = 0
    for index, t in enumerate(times):
        for tr in lidar_frames[t]["tracks"]:
            trails[tr["track_id"]].append(tr["position"])
        for m in camera_frames.get(t, {}).get("matches", []):
            if m["accepted"]:
                camera_hits[m["track_id"]] = t
        if index < args.start:
            continue
        if rendered >= args.frames:
            break
        frame = renderer.frame(t, index, trails, camera_hits)
        frame.save(work / f"{rendered:04d}.png")
        if args.still and rendered == args.still_index:
            frame.save(args.still)
        rendered += 1

    if args.keep_frames:
        shutil.copytree(work, args.keep_frames, dirs_exist_ok=True)
    encode_gif(work, args.out, args.fps, args.width or None, args.colors, args.dither)
    shutil.rmtree(work)
    print(f"wrote {args.out} ({rendered} frames, {args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
