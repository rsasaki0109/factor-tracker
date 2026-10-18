#!/usr/bin/env python3
"""Renders the README hero GIF from one evaluated KITTI tracking sequence.

Inputs are the files written by scripts/kitti/evaluate_sequences.sh for one
sequence (setup.json, inputs.jsonl, camera_assisted.outputs.jsonl). Only
labels, calibration and OXTS-derived poses are used; no KITTI images.

usage: render_hero.py <results/<seq>> <out.gif> --start 0 --frames 100
Requires Pillow and ffmpeg.
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

W, H = 960, 640
SURFACE = (11, 18, 32)
PANEL = (15, 23, 42)
BORDER = (30, 41, 59)
GRID = (26, 36, 56)
TEXT = (241, 245, 249)
TEXT_2 = (148, 163, 184)
TEXT_3 = (91, 107, 133)
# Validated categorical slots (dark): vehicle, pedestrian, cyclist.
CLASS_COLOR = {"vehicle": (57, 135, 229), "pedestrian": (217, 89, 38), "cyclist": (25, 158, 112)}
CLASS_GROUP = {1: "vehicle", 4: "vehicle", 5: "vehicle", 7: "vehicle", 8: "vehicle", 2: "pedestrian", 6: "pedestrian", 3: "cyclist"}

FONT_DIR = Path("/usr/share/fonts/opentype/inter")


def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    path = FONT_DIR / f"Inter-{weight}.otf"
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


F_TITLE, F_SUB, F_LABEL, F_SMALL, F_TAG = font(30, "Bold"), font(15), font(12, "Medium"), font(12), font(11, "SemiBold")


def mat4(values):
    return np.array(values, dtype=float).reshape(4, 4)


def load(seq_dir: Path):
    setup = json.loads((seq_dir / "setup.json").read_text())
    poses = {p["timestamp_ns"]: mat4(p["transform"]) for p in setup["poses"]}
    lidar_inputs = {}
    for line in open(seq_dir / "inputs.jsonl"):
        record = json.loads(line)
        if record["type"] == "lidar_batch":
            lidar_inputs[record["event_time_ns"]] = record["detections"]
    lidar_frames, camera_frames = {}, {}
    for line in open(seq_dir / "camera_assisted.outputs.jsonl"):
        record = json.loads(line)
        if record["type"] == "lidar_frame":
            lidar_frames[record["timestamp_ns"]] = record
        elif record["type"] == "camera_frame":
            camera_frames[record["timestamp_ns"]] = record
    return setup, poses, lidar_inputs, lidar_frames, camera_frames


def group_of(class_id):
    return CLASS_GROUP.get(class_id, "vehicle")


def tint(color, alpha, base=PANEL):
    return tuple(int(base[i] + (color[i] - base[i]) * alpha) for i in range(3))


class Bev:
    """Ego-centric bird's-eye view: velodyne x forward (right), y left (up)."""

    def __init__(self, box, forward=(-6.0, 64.0)):
        self.x0, self.y0, self.x1, self.y1 = box
        self.forward = forward
        self.scale = (self.x1 - self.x0) / (forward[1] - forward[0])
        self.cy = (self.y0 + self.y1) / 2

    def px(self, x, y):
        return (self.x0 + (x - self.forward[0]) * self.scale, self.cy - y * self.scale)

    def inside(self, x, y, margin=4):
        u, v = self.px(x, y)
        return self.x0 + margin < u < self.x1 - margin and self.y0 + margin < v < self.y1 - margin


def draw_frame(t, data, trails, camera_hits, args):
    setup, poses, lidar_inputs, lidar_frames, camera_frames = data
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    # Header.
    d.text((32, 22), "perception-engine", font=F_TITLE, fill=TEXT)
    d.text((34, 60), "Camera-assisted LiDAR tracking in Rust · shared 3D track state, explicit uncertainty", font=F_SUB, fill=TEXT_2)

    # Camera panel (schematic image plane, no pixels).
    intr = setup["intrinsics"]
    # The top of KITTI images is sky; crop it so the panel stays a strip.
    crop_top = args.crop_top
    sx = (W - 64) / intr["image_width"]
    cam_box = (32, 96, W - 32, 96 + int((intr["image_height"] - crop_top) * sx))
    d.rounded_rectangle(cam_box, 10, fill=PANEL, outline=BORDER)
    cam_origin = (cam_box[0], cam_box[1] - crop_top * sx)
    horizon = cam_origin[1] + intr["cy"] * sx
    d.line((cam_box[0] + 1, horizon, cam_box[2] - 1, horizon), fill=GRID, width=1)
    d.text((cam_box[0] + 12, cam_box[1] + 8), "CAMERA · 2D detections vs projected 3D tracks", font=F_TAG, fill=TEXT_3)

    lidar = lidar_frames[t]
    tracks = {tr["track_id"]: tr for tr in lidar["tracks"] if tr["lifecycle"] != "tentative"}
    camera = camera_frames.get(t)
    if camera:
        for m in camera["matches"]:
            x1, y1, x2, y2 = (v * sx for v in m["detection_bbox"])
            box = (cam_origin[0] + x1, max(cam_origin[1] + y1, cam_box[1]), cam_origin[0] + x2, cam_origin[1] + y2)
            d.rectangle(box, outline=TEXT_3, width=1)
            if m["accepted"] and m["track_id"] in tracks and m.get("projected_bbox"):
                color = CLASS_COLOR[group_of(tracks[m["track_id"]]["class_label"])]
                px1, py1, px2, py2 = (v * sx for v in m["projected_bbox"])
                pbox = (cam_origin[0] + px1, max(cam_origin[1] + py1, cam_box[1] + 22), cam_origin[0] + px2,
                        min(cam_origin[1] + py2, cam_box[3] - 1))
                d.rectangle(pbox, outline=color, width=2)
                label = f"#{m['track_id']}"
                tw = d.textlength(label, font=F_LABEL)
                ty = max(pbox[1] - 16, cam_box[1] + 22)
                d.rectangle((pbox[0], ty, pbox[0] + tw + 8, ty + 15), fill=color)
                d.text((pbox[0] + 4, ty + 1), label, font=F_LABEL, fill=SURFACE)

    # BEV panel.
    bev_box = (32, cam_box[3] + 12, W - 32, H - 44)
    d.rounded_rectangle(bev_box, 10, fill=PANEL, outline=BORDER)
    bev = Bev((bev_box[0] + 1, bev_box[1] + 1, bev_box[2] - 1, bev_box[3] - 1), forward=args.forward)
    for r in range(10, int(args.forward[1]), 10):
        u, _ = bev.px(r, 0)
        d.line((u, bev_box[1] + 26, u, bev_box[3] - 30), fill=GRID)
        d.text((u + 4, bev_box[3] - 46), f"{r} m", font=F_SMALL, fill=TEXT_3)
    # Camera field of view, clipped to the panel.
    half_fov = math.atan(intr["image_width"] / 2 / intr["fx"])
    far = 200
    fov = Image.new("L", img.size, 0)
    ImageDraw.Draw(fov).polygon(
        [bev.px(0, 0), bev.px(far, far * math.tan(half_fov)), bev.px(far, -far * math.tan(half_fov))], fill=255
    )
    clip = Image.new("L", img.size, 0)
    ImageDraw.Draw(clip).rounded_rectangle(bev_box, 10, fill=255)
    mask = Image.fromarray((np.asarray(fov, dtype=np.uint16) * np.asarray(clip) // 255 // 14).astype(np.uint8))
    img.paste(Image.new("RGB", img.size, (148, 163, 184)), (0, 0), mask)
    d = ImageDraw.Draw(img)
    d.text((bev_box[0] + 12, bev_box[1] + 8), "BIRD'S-EYE VIEW · LiDAR tracks in the world frame, drawn around the ego vehicle", font=F_TAG, fill=TEXT_3)

    world_to_velo = np.linalg.inv(poses[t])

    def to_velo(p):
        return world_to_velo[:3, :3] @ np.asarray(p) + world_to_velo[:3, 3]

    # Trails.
    for tid, history in trails.items():
        if tid not in tracks or len(history) < 2:
            continue
        color = CLASS_COLOR[group_of(tracks[tid]["class_label"])]
        pts = [to_velo(p) for p in history]
        for k in range(1, len(pts)):
            a, b = pts[k - 1], pts[k]
            if bev.inside(*a[:2]) and bev.inside(*b[:2]):
                alpha = 0.15 + 0.6 * k / len(pts)
                d.line((*bev.px(*a[:2]), *bev.px(*b[:2])), fill=tint(color, alpha), width=2)

    # Boxes from the associated LiDAR detections, then labels.
    detections = lidar_inputs.get(t, [])
    labels = []
    for assoc in lidar["associations"]:
        tid = assoc["track_id"]
        if tid not in tracks:
            continue
        det = detections[assoc["observation_index"]]
        center = np.array(det["position"])
        if not bev.inside(*center[:2]):
            continue
        color = CLASS_COLOR[group_of(tracks[tid]["class_label"])]
        corners = np.array(det["shape"]["corners"]).reshape(8, 3)[:4] + center
        poly = [bev.px(c[0], c[1]) for c in corners]
        seen = camera_hits.get(tid) == t
        d.polygon(poly, fill=tint(color, 0.35 if seen else 0.12), outline=color)
        # Uncertainty: 2-sigma ellipse radius from the track covariance (x, y).
        cov = np.array(tracks[tid]["covariance"]).reshape(6, 6)[:2, :2]
        radius = 2 * math.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0)) * bev.scale
        cu, cv = bev.px(*to_velo(tracks[tid]["position"])[:2])
        d.ellipse((cu - radius, cv - radius, cu + radius, cv + radius), outline=tint(color, 0.7))
        labels.append((sum(p[0] for p in poly) / 4, min(p[1] for p in poly) - 15, f"#{tid}"))
    placed = []
    for u, v, text in sorted(labels, key=lambda item: item[1]):
        tw = d.textlength(text, font=F_LABEL)
        u -= tw / 2
        while any(abs(u - pu) < max(tw, pw) + 2 and abs(v - pv) < 13 for pu, pv, pw in placed):
            v -= 13
        placed.append((u, v, tw))
        d.text((u, v), text, font=F_LABEL, fill=TEXT)

    # Ego vehicle.
    ego = [bev.px(1.6, 0.85), bev.px(1.6, -0.85), bev.px(-2.6, -0.85), bev.px(-2.6, 0.85)]
    d.polygon(ego, fill=TEXT_2)
    d.polygon([bev.px(1.6, 0.85), bev.px(1.6, -0.85), bev.px(2.6, 0)], fill=TEXT_2)
    d.text((bev.px(-2.6, 0)[0], bev.px(0, -1.2)[1]), "ego", font=F_SMALL, fill=TEXT_3)

    # Legend + footer.
    lx = bev_box[0] + 12
    ly = bev_box[3] - 24
    for name in ("vehicle", "pedestrian", "cyclist"):
        d.rectangle((lx, ly + 3, lx + 10, ly + 13), fill=CLASS_COLOR[name])
        d.text((lx + 15, ly), name, font=F_SMALL, fill=TEXT_2)
        lx += d.textlength(name, font=F_SMALL) + 34
    d.text((lx + 4, ly), "filled = camera-associated this frame · ring = 2σ position uncertainty", font=F_SMALL, fill=TEXT_3)
    matched = sum(1 for m in (camera or {}).get("matches", []) if m["accepted"])
    status = f"frame {t // 100_000_000:04d} · {len(tracks)} tracks · {matched} camera associations"
    d.text((34, H - 32), status, font=F_SMALL, fill=TEXT_2)
    attribution = f"KITTI tracking {args.sequence_name}, labels as oracle detections · KITTI data CC BY-NC-SA 3.0"
    d.text((W - 34 - d.textlength(attribution, font=F_SMALL), H - 32), attribution, font=F_SMALL, fill=TEXT_3)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seq_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--trail", type=int, default=20)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--forward", type=float, nargs=2, default=(-5.0, 62.0))
    parser.add_argument("--crop-top", type=float, default=110.0, help="image rows (pixels) to crop at the top")
    parser.add_argument("--auto", action="store_true", help="pick the window with the most camera associations")
    parser.add_argument("--still", type=Path, help="also write this frame index (relative) as PNG")
    parser.add_argument("--still-index", type=int, default=50)
    args = parser.parse_args()
    args.sequence_name = args.seq_dir.name

    data = load(args.seq_dir)
    lidar_frames, camera_frames = data[3], data[4]
    times = sorted(lidar_frames)
    if args.auto:
        counts = [sum(m["accepted"] for m in camera_frames.get(t, {}).get("matches", [])) for t in times]
        sums = [sum(counts[i : i + args.frames]) for i in range(max(1, len(times) - args.frames))]
        args.start = int(np.argmax(sums))
        print(f"auto window starts at frame {args.start} ({sums[args.start]} camera associations)")
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
        frame = draw_frame(t, data, trails, camera_hits, args)
        frame.save(work / f"{rendered:04d}.png")
        if args.still and rendered == args.still_index:
            frame.save(args.still)
        rendered += 1

    palette = work / "palette.png"
    pattern = str(work / "%04d.png")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(args.fps), "-i", pattern,
                    "-vf", "palettegen=max_colors=128:stats_mode=diff", str(palette)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(args.fps), "-i", pattern, "-i", str(palette),
                    "-lavfi", "paletteuse=dither=none:diff_mode=rectangle", "-loop", "0", str(args.out)], check=True)
    shutil.rmtree(work)
    print(f"wrote {args.out} ({rendered} frames, {args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
