#!/usr/bin/env python3
"""Extracts a frame range of one CODa sequence from the per-sequence ZIP on
the TACC server (40-110 GB each) with parallel HTTP range requests.

usage: fetch_coda.py --coda-root CODa --sequence 20 --first 8640 --last 8800

Fetched: rectified cam0 images, ego-compensated OS1 scans and 3D boxes for the
frame range, plus the sequence's calibrations, timestamps, dense_global poses
and metadata. CODa is CC BY-NC-SA 4.0 with additional dataset terms; read
docs/DATA_REPORT.md in https://github.com/ut-amrl/coda-devkit first.
The ZIP index is cached as <coda-root>/.index_<seq>.json.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kitti"))

from fetch_frames import HttpRangeFile  # noqa: E402

BASE = "https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/sequences"
LOCAL_HEADER = 30


def fetch(url: str, start: int, end: int, attempts: int = 6) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if len(data) == end - start:
                return data
        except (OSError, http.client.HTTPException):
            if attempt == attempts - 1:
                raise
        time.sleep(2**attempt)
    raise OSError(f"short read for bytes {start}-{end - 1}")


def load_index(url: str, cache: Path) -> dict[str, list[int]]:
    if cache.exists():
        return json.loads(cache.read_text())
    archive = zipfile.ZipFile(HttpRangeFile(url))
    index = {i.filename: [i.header_offset, i.compress_size, i.compress_type, i.file_size] for i in archive.infolist()}
    cache.write_text(json.dumps(index))
    return index


def extract(url: str, entry: list[int], target: Path) -> None:
    offset, compressed, method, size = entry
    # Local header (30 bytes) + name + extra field; the extra field may differ
    # from the central directory one, so read the header first.
    header = fetch(url, offset, offset + LOCAL_HEADER)
    name_length = int.from_bytes(header[26:28], "little")
    extra_length = int.from_bytes(header[28:30], "little")
    start = offset + LOCAL_HEADER + name_length + extra_length
    data = fetch(url, start, start + compressed)
    if method == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif method != zipfile.ZIP_STORED:
        raise ValueError(f"unsupported compression {method} for {target}")
    if len(data) != size:
        raise OSError(f"size mismatch for {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_bytes(data)
    partial.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coda-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--lidar-only", action="store_true", help="skip the camera images")
    args = parser.parse_args()

    seq = args.sequence
    url = f"{BASE}/{seq}.zip"
    args.coda_root.mkdir(parents=True, exist_ok=True)
    index = load_index(url, args.coda_root / f".index_{seq}.json")
    prefixes = (f"calibrations/{seq}/", f"metadata/{seq}.", f"timestamps/{seq}.", f"poses/dense_global/{seq}.")
    wanted = [name for name in index if name.startswith(prefixes)]
    for frame in range(args.first, args.last + 1):
        wanted += [
            f"3d_comp/os1/{seq}/3d_comp_os1_{seq}_{frame}.bin",
            f"3d_bbox/os1/{seq}/3d_bbox_os1_{seq}_{frame}.json",
        ]
        if not args.lidar_only:
            wanted.append(f"2d_rect/cam0/{seq}/2d_rect_cam0_{seq}_{frame}.png")
    jobs = [name for name in wanted if name in index and not (args.coda_root / name).exists()]
    print(f"{len(jobs)} files to fetch ({sum(index[n][1] for n in jobs) / 1e6:.0f} MB)", flush=True)
    for round_ in range(3):
        failed = []
        with ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(extract, url, index[name], args.coda_root / name): name for name in jobs}
            for done, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except (OSError, http.client.HTTPException, ValueError) as error:
                    failed.append(futures[future])
                    print(f"  failed {futures[future]}: {error}", flush=True)
                if done % 50 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)}", flush=True)
        if not failed:
            return
        jobs = failed
        print(f"retrying {len(jobs)} files", flush=True)
    raise SystemExit(f"{len(jobs)} files could not be fetched")


if __name__ == "__main__":
    main()
