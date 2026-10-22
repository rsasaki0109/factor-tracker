#!/usr/bin/env python3
"""Re-implementations of two standard 3D trackers on the engine's detections.

Both read the detections (`observations`, tracking frame) of an engine output
JSONL and write the same format, so every tracker sees identical inputs and is
scored by the same tools. They are re-implementations for comparison, not the
authors' code:

  ab3dmot      Weng et al., AB3DMOT (IROS 2020): constant-velocity Kalman
               filter, Hungarian assignment with a gate, a track is reported
               once it has `min_hits` hits and only in scans where it was
               updated, deleted after `max_age` missed scans. Boxes are not
               available, so the 3D IoU gate is replaced by a centre-distance
               gate (as CenterPoint does for pedestrians).
  simpletrack  Pang et al., SimpleTrack (ECCVW 2022): the same filter and
               assignment with a Mahalanobis gate, longer life, and
               predicted states reported while a confirmed track coasts
               ("prediction during missing").

usage: baselines.py ab3dmot|simpletrack engine_output.jsonl out.jsonl [options]
"""

import argparse
import json

import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, tid, z, t, a):
        self.id, self.t = tid, t
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.p = np.diag([a.sigma ** 2, a.sigma ** 2, a.velocity_variance, a.velocity_variance])
        self.z = z[2]
        self.hits, self.miss, self.updated = 1, 0, True

    def predict(self, t, q):
        dt = t - self.t
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        g = np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0], [0, dt ** 3 / 3, 0, dt ** 2 / 2],
                      [dt ** 2 / 2, 0, dt, 0], [0, dt ** 2 / 2, 0, dt]])
        self.x, self.p, self.t = f @ self.x, f @ self.p @ f.T + q * g, t

    def innovation(self, z, r):
        s = self.p[:2, :2] + r * np.eye(2)
        nu = np.asarray(z[:2]) - self.x[:2]
        return nu, s

    def update(self, z, r):
        nu, s = self.innovation(z, r)
        k = self.p[:, :2] @ np.linalg.inv(s)
        self.x = self.x + k @ nu
        self.p = (np.eye(4) - k @ np.hstack([np.eye(2), np.zeros((2, 2))])) @ self.p
        self.z = 0.8 * self.z + 0.2 * z[2]
        self.hits += 1
        self.miss = 0
        self.updated = True


def run(kind, frames, a):
    tracks, next_id, out = [], 1, []
    r = a.sigma ** 2
    for frame in frames:
        t = frame["timestamp_ns"] * 1e-9
        dets = [np.array(o) for o in frame["observations"]]
        for tr in tracks:
            tr.predict(t, a.accel)
            tr.updated = False
        cost = np.full((len(tracks), len(dets)), 1e6)
        for i, tr in enumerate(tracks):
            for j, z in enumerate(dets):
                nu, s = tr.innovation(z, r)
                if kind == "ab3dmot":
                    d = np.linalg.norm(nu)
                    if d <= a.gate:
                        cost[i, j] = d
                else:
                    m = nu @ np.linalg.inv(s) @ nu
                    if m <= a.gate:
                        cost[i, j] = m
        used = {}
        if tracks and dets:
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < 1e6:
                    tracks[i].update(dets[j], r)
                    used[j] = tracks[i].id
        for tr in tracks:
            if not tr.updated:
                tr.miss += 1
        tracks = [tr for tr in tracks if tr.miss <= a.max_age]
        associations = []
        for j, z in enumerate(dets):
            if j in used:
                associations.append({"observation_index": j, "track_id": used[j], "outcome": "updated", "cost": None})
            else:
                tracks.append(Track(next_id, z, t, a))
                associations.append({"observation_index": j, "track_id": next_id, "outcome": "spawned", "cost": None})
                next_id += 1
        reported = []
        for tr in tracks:
            confirmed = tr.hits >= a.min_hits
            if kind == "ab3dmot" and not tr.updated:
                continue
            ns = frame["timestamp_ns"]
            reported.append({
                "track_id": tr.id, "lifecycle": "confirmed" if confirmed else "tentative",
                "state_timestamp_ns": ns, "last_spatial_update_ns": ns, "last_camera_support_ns": None,
                "age_since_spatial_update_ns": 0 if tr.updated else int(tr.miss * 1e8),
                "prediction_only": not tr.updated, "position": [tr.x[0], tr.x[1], tr.z],
                "velocity": [tr.x[2], tr.x[3], 0.0], "covariance": [0.0] * 36, "class_label": 2,
                "class_prior": None, "contributing_sensors": [1], "observation_times_count": tr.hits,
            })
        out.append({"type": "lidar_frame", "mode": "lidar_only", "timestamp_ns": frame["timestamp_ns"],
                    "tracks": reported, "observations": frame["observations"], "associations": associations,
                    "unmatched_track_ids": []})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["ab3dmot", "simpletrack"])
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--gate", type=float, help="m (ab3dmot) or squared Mahalanobis (simpletrack)")
    ap.add_argument("--max-age", type=int)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--accel", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--velocity-variance", type=float, default=1.0)
    a = ap.parse_args()
    if a.gate is None:
        a.gate = 1.0 if a.kind == "ab3dmot" else 11.8
    if a.max_age is None:
        a.max_age = 2 if a.kind == "ab3dmot" else 10
    frames = [json.loads(l) for l in open(a.input) if '"lidar_frame"' in l]
    with open(a.output, "w") as h:
        for r in run(a.kind, frames, a):
            h.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
