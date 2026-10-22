#!/usr/bin/env python3
"""Extracts a few frames of one KITTI tracking sequence from the official ZIP
archives over HTTP range requests, without downloading the whole archive
(image_2 is ~16 GB, velodyne ~37 GB).

usage: fetch_frames.py --kitti-root <dir> --sequence 0019 --first 160 --last 280 [--parts image_2 velodyne]

Accept the KITTI terms at https://www.cvlibs.net/datasets/kitti/ first.
Files land in the usual layout, e.g. <dir>/training/image_02/0019/000160.png.
"""

from __future__ import annotations

import argparse
import http.client
import io
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
MEMBER_DIR = {"image_2": "image_02", "velodyne": "velodyne"}
SUFFIX = {"image_2": ".png", "velodyne": ".bin"}


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, with a small block cache."""

    BLOCK = 1 << 20

    def __init__(self, url: str):
        self.url = url
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            self.size = int(response.headers["Content-Length"])
        self.position = 0
        self.cache: dict[int, bytes] = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def _fetch(self, start: int, end: int, attempts: int = 5) -> bytes:
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read()
                if len(data) == end - start:
                    return data
            except (OSError, http.client.HTTPException):
                if attempt == attempts - 1:
                    raise
            time.sleep(2**attempt)
        raise OSError(f"short read for bytes {start}-{end - 1} of {self.url}")

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.position
        size = max(0, min(size, self.size - self.position))
        if size == 0:
            return b""
        if size > self.BLOCK:
            data = self._fetch(self.position, self.position + size)
        else:
            chunks = []
            remaining, position = size, self.position
            while remaining:
                block = position // self.BLOCK
                if block not in self.cache:
                    if len(self.cache) > 64:
                        self.cache.clear()
                    start = block * self.BLOCK
                    self.cache[block] = self._fetch(start, min(start + self.BLOCK, self.size))
                offset = position - block * self.BLOCK
                piece = self.cache[block][offset : offset + remaining]
                chunks.append(piece)
                remaining -= len(piece)
                position += len(piece)
            data = b"".join(chunks)
        self.position += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--split", default="training")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--parts", nargs="+", default=["image_2", "velodyne"], choices=sorted(MEMBER_DIR))
    args = parser.parse_args()

    for part in args.parts:
        archive = zipfile.ZipFile(HttpRangeFile(f"{BASE}/data_tracking_{part}.zip"))
        wanted = {
            f"{args.split}/{MEMBER_DIR[part]}/{args.sequence}/{frame:06d}{SUFFIX[part]}"
            for frame in range(args.first, args.last + 1)
        }
        members = [info for info in archive.infolist() if info.filename in wanted]
        for info in members:
            target = args.kitti_root / info.filename
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
        print(f"{part}: {len(members)} of {len(wanted)} frames in {args.kitti_root / args.split / MEMBER_DIR[part] / args.sequence}")


if __name__ == "__main__":
    main()
