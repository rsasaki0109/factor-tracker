#!/usr/bin/env python3
r"""Renders a side-by-side comparison GIF: two trackers' outputs on the same
detections, bird's-eye view around the robot, with every identity switch
marked.

An identity switch is counted as in scripts/eval/mot_metrics.py (CLEAR MOT):
per frame, truth people and confirmed tracks are matched within --max-distance,
keeping the previous match while it stays within it; a person whose matched
track changes is a switch. The counters in the GIF therefore equal the IDSW
that mot_metrics.py reports for the same files.

Inputs: a window directory written by scripts/eval/coda_tracker_eval.sh
(setup.json, truth.jsonl and one output .jsonl per tracker) and the CODa scans
for the LiDAR points.

usage: render_compare.py <window-dir> <out.gif> --coda-root CODa --sequence 20 \
           --left ab3dmot.jsonl "AB3DMOT-style" --right engine_distance.jsonl "factor-tracker"
Requires NumPy, SciPy, Pillow and ffmpeg.

The README hero (the comparison) was produced with (CODa sequence 20, scans 12700-12860;
the clearest of the eight evaluation windows; each tracker at its best setting on dev,
docs/evaluation.md):

    # detector without NMS, then NMS IoU 0.1 (AB3DMOT-style) and 0.3 (factor-tracker)
    python3 scripts/coda/detect_lidar_cpu.py ... --score-threshold 0.1 --nms-threshold 1.0 --out raw/coda20_12700
    python3 scripts/eval/nms_variants.py raw nms iou0.1 iou0.3
    PY=python3 scripts/eval/coda_tracker_eval.sh nms/iou0.1 CODa 0.4 work_a
    PY=python3 scripts/eval/coda_tracker_eval.sh nms/iou0.3 CODa 0.5 work_b
    cp work_b/coda20_12700/20/engine_birth.jsonl work_a/coda20_12700/20/
    python3 scripts/hero/render_compare.py work_a/coda20_12700/20 assets/compare_ab3dmot.gif \
        --coda-root CODa --sequence 20 \
        --left ab3dmot.jsonl "AB3DMOT-style (NMS 0.1, score >= 0.4)" \
        --right engine_birth.jsonl "factor-tracker (NMS 0.3, score >= 0.5, birth suppression)" \
        --title "Same detector, two trackers: identity switches" --subtitle ... --footer ...
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_hero import (  # noqa: E402
    BORDER, F_LABEL, F_SMALL, F_SUB, F_TAG, F_TITLE, GRID, MARGIN, PANEL, SURFACE, TEXT, TEXT_2, TEXT_3,
    encode_gif, font, mat4, tint,
)

SWITCH = (239, 68, 68)
TRUTH = (100, 116, 139)
POINTS = (71, 85, 105)
# Track colours cycle through a validated dark-theme categorical set.
PALETTE = [(57, 135, 229), (217, 89, 38), (25, 158, 112), (196, 140, 32), (152, 102, 214), (36, 160, 178),
           (214, 92, 150), (120, 160, 60)]
F_COUNT = font(22, "Bold")


def load_frames(path: Path) -> dict[int, list]:
    frames = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("type") == "lidar_frame":
            frames[r["timestamp_ns"]] = [(t["track_id"], np.array(t["position"][:3], float)) for t in r["tracks"]
                                         if t["lifecycle"] == "confirmed"]
    return frames


def load_truth(path: Path) -> dict[int, list]:
    truth = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("type") == "truth3d":
            truth[r["timestamp_ns"]] = [(o["truth_id"], np.array(o["position"][:3], float)) for o in r["objects"]]
    return truth


def switches(times, truth, frames, dmax):
    """Per frame: list of (truth position, old track id, new track id), as mot_metrics.clear_and_id."""
    last, events = {}, {}
    for t in times:
        objs, tracks = truth.get(t, []), frames.get(t, [])
        events[t] = []
        if not objs or not tracks:
            continue
        d = np.linalg.norm(np.array([p for _, p in objs])[:, None] - np.array([p for _, p in tracks])[None], axis=2)
        cost = np.where(d <= dmax, d, 1e6)
        for r, (g, _) in enumerate(objs):
            for c, (k, _) in enumerate(tracks):
                if last.get(g) == k and d[r, c] <= dmax:
                    cost[r, c] = -1.0
        for r, c in zip(*linear_sum_assignment(cost)):
            if cost[r, c] >= 1e6:
                continue
            g, k = objs[r][0], tracks[c][0]
            if g in last and last[g] != k:
                events[t].append((objs[r][1], last[g], k))
            last[g] = k
    return events


class Panel:
    def __init__(self, box, half_range):
        self.x0, self.y0, self.x1, self.y1 = box
        self.scale = (self.x1 - self.x0) / (2 * half_range)
        self.cx, self.cy = (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def px(self, p):
        # Robot frame: x forward (up), y left (left).
        return self.cx - p[1] * self.scale, self.cy - p[0] * self.scale

    def inside(self, u, v, margin=4):
        return self.x0 + margin < u < self.x1 - margin and self.y0 + margin < v < self.y1 - margin


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("window", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--coda-root", type=Path, required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--left", nargs=2, required=True, metavar=("OUTPUT", "LABEL"))
    ap.add_argument("--right", nargs=2, required=True, metavar=("OUTPUT", "LABEL"))
    ap.add_argument("--title", default="Same detections, two trackers: identity switches")
    ap.add_argument("--subtitle", default="CODa seq 20, scans 12700-12860 (16 s) · Clearpath Husky in a campus crowd "
                    "· pedestrian detections: CODa PV-RCNN first stage on CPU, score >= 0.4")
    ap.add_argument("--footer", nargs="+", default=[
        "All 8 CODa evaluation windows (pedestrians, same detections): ID switches 1015 -> 952, fewer in every "
        "window;",
        "IDF1 0.746 -> 0.760 (dev), 0.882 -> 0.900 (held-out); HOTA tied (0.618 / 0.618 dev). "
        "This window is the clearest of the eight."])
    ap.add_argument("--max-distance", type=float, default=1.0)
    ap.add_argument("--range-m", type=float, default=24.0, help="half width of each panel around the robot")
    ap.add_argument("--panel", type=int, default=540)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--hold-s", type=float, default=1.5, help="how long a switch marker stays")
    ap.add_argument("--colors", type=int, default=96)
    ap.add_argument("--end-hold-s", type=float, default=2.5, help="how long the last frame stays")
    args = ap.parse_args()

    setup = json.loads((args.window / "setup.json").read_text())
    poses = {p["timestamp_ns"]: mat4(p["transform"]) for p in setup["poses"]}
    lidar_to_ref = mat4(setup["lidar_to_reference"])
    numbers = {}
    for line in open(args.window / "inputs.jsonl"):
        r = json.loads(line)
        if r["type"] == "lidar_batch":
            numbers[r["event_time_ns"]] = r["sequence"]
    times = sorted(t for t in numbers if t in poses)
    truth = load_truth(args.window / "truth.jsonl")
    sides = []
    for output, label in (args.left, args.right):
        frames = load_frames(args.window / output)
        sides.append({"label": label, "frames": frames, "events": switches(times, truth, frames, args.max_distance)})

    width = 2 * args.panel + 3 * MARGIN
    top = 96
    height = top + 44 + args.panel + 52 + 18 * len(args.footer)
    boxes = [(MARGIN, top + 44, MARGIN + args.panel, top + 44 + args.panel),
             (2 * MARGIN + args.panel, top + 44, 2 * MARGIN + 2 * args.panel, top + 44 + args.panel)]
    panels = [Panel(b, args.range_m) for b in boxes]
    hold = int(round(args.hold_s * args.fps))

    with tempfile.TemporaryDirectory() as tmp:
        trails = [defaultdict(lambda: deque(maxlen=15)) for _ in sides]
        markers = [[] for _ in sides]
        counts = [0, 0]
        for n, t in enumerate(times):
            world_to_ref = np.linalg.inv(poses[t])
            to_ref = lambda p: (world_to_ref @ np.append(p, 1.0))[:3]  # noqa: E731
            scan = args.coda_root / "3d_comp" / "os1" / args.sequence / f"3d_comp_os1_{args.sequence}_{numbers[t]}.bin"
            points = np.fromfile(scan, dtype=np.float32).reshape(-1, 4)[:, :3]
            points = points[(points[:, 2] > -1.2) & (points[:, 2] < 1.5)][::3]
            points = points @ lidar_to_ref[:3, :3].T + lidar_to_ref[:3, 3]

            img = Image.new("RGB", (width, height), SURFACE)
            draw = ImageDraw.Draw(img)
            draw.text((MARGIN, 22), args.title, font=F_TITLE, fill=TEXT)
            draw.text((MARGIN, 60), args.subtitle, font=F_SUB, fill=TEXT_2)
            for k, (side, panel, box) in enumerate(zip(sides, panels, boxes)):
                for p, old, new in side["events"][t]:
                    markers[k].append([p, old, new, hold])
                    counts[k] += 1
                draw.text((box[0], top + 8), side["label"], font=F_LABEL, fill=TEXT)
                count = f"ID switches: {counts[k]}"
                draw.text((box[2] - draw.textlength(count, font=F_COUNT), top + 2), count, font=F_COUNT,
                          fill=SWITCH if counts[k] else TEXT_2)
                draw.rectangle(box, fill=PANEL, outline=BORDER)
                for r in range(10, int(args.range_m) + 1, 10):
                    u0, v0 = panel.px((r, r))
                    u1, v1 = panel.px((-r, -r))
                    draw.ellipse((u0, v0, u1, v1), outline=GRID)
                    draw.text((panel.cx + 3, panel.cy - r * panel.scale + 2), f"{r} m", font=F_TAG, fill=TEXT_3)
                for p in points:
                    u, v = panel.px(p)
                    if panel.inside(u, v, 1):
                        draw.point((u, v), fill=POINTS)
                for _, p in truth.get(t, []):
                    u, v = panel.px(to_ref(p))
                    if panel.inside(u, v):
                        draw.ellipse((u - 6, v - 6, u + 6, v + 6), outline=TRUTH)
                tracks = side["frames"].get(t, [])
                for track_id, p in tracks:
                    trails[k][track_id].append(p)
                for track_id, p in tracks:
                    color = PALETTE[track_id % len(PALETTE)]
                    trail = [panel.px(to_ref(q)) for q in trails[k][track_id]]
                    trail = [q for q in trail if panel.inside(*q)]
                    for a, b in zip(trail, trail[1:]):
                        draw.line((a, b), fill=tint(color, 0.55), width=2)
                    u, v = panel.px(to_ref(p))
                    if panel.inside(u, v):
                        draw.ellipse((u - 4, v - 4, u + 4, v + 4), fill=color)
                        draw.text((u + 6, v - 7), str(track_id), font=F_TAG, fill=tint(color, 0.9, TEXT))
                for m in markers[k]:
                    u, v = panel.px(to_ref(m[0]))
                    if panel.inside(u, v):
                        r = 11 + 3 * (m[3] / hold)
                        draw.ellipse((u - r, v - r, u + r, v + r), outline=SWITCH, width=3)
                        draw.text((u - 22, v + r + 2), f"{m[1]}->{m[2]}", font=F_LABEL, fill=SWITCH)
                    m[3] -= 1
                markers[k] = [m for m in markers[k] if m[3] > 0]
                # The robot, facing up.
                draw.polygon([(panel.cx, panel.cy - 9), (panel.cx - 6, panel.cy + 6), (panel.cx + 6, panel.cy + 6)],
                             fill=TEXT)
            legend_y = boxes[0][3] + 12
            x = MARGIN
            for swatch, text in ((None, "person (truth)"), (PALETTE[0], "confirmed track + ID"),
                                 (SWITCH, "identity switch (old->new ID)")):
                if swatch is None:
                    draw.ellipse((x, legend_y + 2, x + 12, legend_y + 14), outline=TRUTH)
                elif swatch == SWITCH:
                    draw.ellipse((x, legend_y + 1, x + 14, legend_y + 15), outline=SWITCH, width=2)
                else:
                    draw.ellipse((x + 2, legend_y + 4, x + 10, legend_y + 12), fill=swatch)
                draw.text((x + 20, legend_y), text, font=F_SMALL, fill=TEXT_2)
                x += 40 + draw.textlength(text, font=F_SMALL)
            draw.text((width - MARGIN - draw.textlength(f"{(t - times[0]) / 1e9:4.1f} s", font=F_SMALL), legend_y),
                      f"{(t - times[0]) / 1e9:4.1f} s", font=F_SMALL, fill=TEXT_2)
            for line_no, line in enumerate(args.footer):
                draw.text((MARGIN, legend_y + 26 + 18 * line_no), line, font=F_SMALL, fill=TEXT_2)
            img.save(Path(tmp) / f"{n:04d}.png")
        for k in range(int(round(args.end_hold_s * args.fps))):
            img.save(Path(tmp) / f"{len(times) + k:04d}.png")
        encode_gif(Path(tmp), args.out, args.fps, None, args.colors, "none")
    print(f"{args.out}: {len(times)} frames, ID switches {counts[0]} vs {counts[1]}")


if __name__ == "__main__":
    main()
