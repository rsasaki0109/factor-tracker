#!/usr/bin/env python3
"""Aggregates the per-window TSVs written by scripts/coda/run_window.sh.

usage: summarize_windows.py <results-dir> [--detections real] [--baseline base]

Prints, per configuration and zone, the ID switches summed over windows, the
change against the baseline configuration, and in how many windows the
configuration was better / equal / worse than the baseline.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ZONES = ["near5", "front8", "around8", "all"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--detections", default="real")
    parser.add_argument("--baseline", default="base")
    args = parser.parse_args()

    # data[window][config][zone] = (switches, coverage, gt)
    data: dict = defaultdict(lambda: defaultdict(dict))
    order: list[str] = []
    for path in sorted(args.results.glob("*.tsv")):
        for row in csv.DictReader(open(path), delimiter="\t"):
            if row["detections"] != args.detections:
                continue
            data[path.stem][row["config"]][row["zone"]] = (
                int(row["id_switches"]), float(row["coverage"]), int(row["gt_object_frames"]))
            if row["config"] not in order:
                order.append(row["config"])
    windows = sorted(data)
    print(f"{args.detections} detections, {len(windows)} windows: {', '.join(windows)}\n")
    print("per window, ID switches (front8 / all):")
    for config in order:
        cells = [f"{data[w][config]['front8'][0]:>3}/{data[w][config]['all'][0]:<4}" for w in windows]
        print(f"  {config:16s} " + "  ".join(cells))
    print("\nsummed over windows (change vs baseline; windows better/equal/worse):")
    for zone in ZONES:
        base = sum(data[w][args.baseline][zone][0] for w in windows)
        frames = sum(data[w][args.baseline][zone][2] for w in windows)
        print(f"  {zone} ({frames} truth object-frames):")
        for config in order:
            total = sum(data[w][config][zone][0] for w in windows)
            coverage = sum(data[w][config][zone][1] * data[w][config][zone][2] for w in windows) / max(frames, 1)
            better = sum(data[w][config][zone][0] < data[w][args.baseline][zone][0] for w in windows)
            worse = sum(data[w][config][zone][0] > data[w][args.baseline][zone][0] for w in windows)
            change = (total - base) / base * 100 if base else 0.0
            print(f"    {config:16s} {total:5d} ({change:+6.1f} %)  coverage {coverage:.3f}  "
                  f"{better}/{len(windows) - better - worse}/{worse}")


if __name__ == "__main__":
    main()
