#!/usr/bin/env python3
"""Aggregates per-sequence metrics written by evaluate_sequences.sh.

usage: summarize.py <results-dir> <mode> [<mode> ...]

Counts are summed over sequences. Coverage and class accuracy are pooled
ratios, the RMSE is pooled over all matched object-timestamps. The metric
definitions are those of docs/evaluation.md; none of them is MOTA/HOTA/IDF1.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    results = Path(sys.argv[1])
    modes = sys.argv[2:]
    sequences = sorted(p.name for p in results.iterdir() if (p / f"{modes[0]}.metrics.json").exists())

    summary = {"sequences": sequences, "modes": {}}
    for mode in modes:
        total = dict(
            gt=0, matched=0, squared_error=0.0, id_switches=0, class_correct=0, class_total=0,
            camera_matches=0, camera_unmatched=0, association_rejected=0, transform_unavailable=0,
        )
        for seq in sequences:
            metrics = json.loads((results / seq / f"{mode}.metrics.json").read_text())["modes"][0]
            counters = json.loads((results / seq / f"{mode}.diagnostics.json").read_text())["counters"]["engine"]
            total["gt"] += metrics["gt_object_times"]
            total["matched"] += metrics["matched_object_times"]
            if metrics["position_rmse"] is not None:
                total["squared_error"] += metrics["position_rmse"] ** 2 * metrics["matched_object_times"]
            total["id_switches"] += metrics["id_switches"]
            total["class_correct"] += metrics["class_correct"]
            total["class_total"] += metrics["class_total"]
            for key in ("camera_matches", "camera_unmatched", "association_rejected", "transform_unavailable"):
                total[key] += counters[key]
        total["track_coverage"] = ratio(total["matched"], total["gt"])
        mean_square = ratio(total["squared_error"], total["matched"])
        total["position_rmse"] = math.sqrt(mean_square) if mean_square is not None else None
        total["class_accuracy"] = ratio(total["class_correct"], total["class_total"])
        total["camera_match_rate"] = (
            ratio(total["camera_matches"], total["camera_matches"] + total["camera_unmatched"])
            if mode != "lidar_only"
            else None
        )
        summary["modes"][mode] = total

    (results / "summary.json").write_text(json.dumps(summary, indent=2))
    rows = [
        "| mode | coverage | position RMSE (m) | id switches | class accuracy | camera match rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for mode, total in summary["modes"].items():
        rows.append(
            f"| {mode} | {fmt(total['track_coverage'])} | {fmt(total['position_rmse'])} | "
            f"{total['id_switches']} | {fmt(total['class_accuracy'])} | {fmt(total['camera_match_rate'])} |"
        )
    table = "\n".join(rows)
    (results / "summary.md").write_text(f"{len(sequences)} sequences: {' '.join(sequences)}\n\n{table}\n")
    print(table)


if __name__ == "__main__":
    main()
