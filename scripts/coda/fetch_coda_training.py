#!/usr/bin/env python3
"""Fetches CODa scans with 3D labels for detector training, leaving out the
tracking evaluation windows.

For every sequence: every `--stride`-th scan that has a `3d_bbox` label,
except scans within `--margin` of an evaluation window (the windows that
scripts/eval/coda_tracker_eval.sh scores), over all of CODa's own splits.
Fetches the ego-compensated OS1 scan and the label of each, and writes the
chosen frames to `<coda-root>/training_frames.txt` (`<seq> <frame>` per line).

usage: fetch_coda_training.py --coda-root CODa [--stride 3] [--margin 100]
           [--sequences 0 1 ...] [--limit N]

CODa is CC BY-NC-SA 4.0 with additional dataset terms; read
docs/DATA_REPORT.md in https://github.com/ut-amrl/coda-devkit first.
"""

from __future__ import annotations

import argparse
import http.client
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fetch_coda import BASE, extract, load_index

# (sequence, first, last) of the tracking evaluation windows; keep in sync
# with scripts/eval/coda_tracker_eval.sh.
EVAL_WINDOWS = [
    ("13", 9700, 9860), ("20", 12700, 12860), ("20", 4820, 4980), ("20", 8640, 8800), ("20", 9330, 9490),
    ("3", 1800, 1959), ("4", 6870, 7029), ("9", 8515, 8674),
]


def excluded(seq: str, frame: int, margin: int) -> bool:
    return any(s == seq and first - margin <= frame <= last + margin for s, first, last in EVAL_WINDOWS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coda-root", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3, help="keep every n-th labelled scan")
    parser.add_argument("--margin", type=int, default=100, help="scans left out around each evaluation window")
    parser.add_argument("--sequences", nargs="*", default=[str(s) for s in range(23)])
    parser.add_argument("--limit", type=int, help="at most this many scans in total (for a quick test)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.coda_root.mkdir(parents=True, exist_ok=True)

    chosen: list[tuple[str, int]] = []
    jobs: list[tuple[str, str]] = []
    for seq in args.sequences:
        url = f"{BASE}/{seq}.zip"
        try:
            index = load_index(url, args.coda_root / f".index_{seq}.json")
        except (OSError, http.client.HTTPException) as error:
            print(f"sequence {seq}: index not readable ({error}), skipped", flush=True)
            continue
        pattern = re.compile(rf"3d_bbox/os1/{seq}/3d_bbox_os1_{seq}_(\d+)\.json$")
        labelled = sorted(int(m.group(1)) for name in index if (m := pattern.match(name)))
        frames = [f for f in labelled[::args.stride] if not excluded(seq, f, args.margin)]
        frames = [f for f in frames if f"3d_comp/os1/{seq}/3d_comp_os1_{seq}_{f}.bin" in index]
        print(f"sequence {seq}: {len(labelled)} labelled scans, {len(frames)} chosen", flush=True)
        for f in frames:
            chosen.append((seq, f))
            for name in (f"3d_comp/os1/{seq}/3d_comp_os1_{seq}_{f}.bin", f"3d_bbox/os1/{seq}/3d_bbox_os1_{seq}_{f}.json"):
                if not (args.coda_root / name).exists():
                    jobs.append((url, name, index[name]))
        if args.limit and len(chosen) >= args.limit:
            break
    if args.limit:
        keep = {f"{s}_{f}" for s, f in chosen[:args.limit]}
        chosen = chosen[:args.limit]
        jobs = [j for j in jobs if re.search(r"_(\d+_\d+)\.(bin|json)$", j[1]).group(1) in keep]
    print(f"{len(chosen)} scans, {len(jobs)} files to fetch ({sum(j[2][1] for j in jobs) / 1e9:.1f} GB)", flush=True)

    for round_ in range(3):
        failed = []
        with ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(extract, url, entry, args.coda_root / name): (url, name, entry)
                       for url, name, entry in jobs}
            for done, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except (OSError, http.client.HTTPException, ValueError) as error:
                    failed.append(futures[future])
                    print(f"  failed {futures[future][1]}: {error}", flush=True)
                if done % 500 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)}", flush=True)
        if not failed:
            break
        jobs = failed
        print(f"retrying {len(jobs)} files", flush=True)
    else:
        raise SystemExit(f"{len(jobs)} files could not be fetched")
    (args.coda_root / "training_frames.txt").write_text("".join(f"{s} {f}\n" for s, f in chosen))
    print(f"wrote {args.coda_root / 'training_frames.txt'}", flush=True)


if __name__ == "__main__":
    main()
