# GrandTour ETH-2: Mid-360 pedestrian identities

A small identity benchmark for a Livox Mid-360 on a legged robot in a crowd:
the ANYmal of [GrandTour](https://grand-tour.leggedrobotics.com) mission
`2024-10-01-11-47-44` (ETH Zurich main hall), LiDAR scans 3700-3900 (20 s,
10 Hz), pedestrians within 8 m of the robot. 68 people, 2,855 labelled boxes
(window 1, scans 3700-3800: 43 people, 1,832 boxes; window 2, scans
3801-3900: 44 people, 1,023 boxes, some of them continuing from window 1),
up to 30 detections per scan.

**Annotated by Claude (an AI model), not verified by a person.** Read
`annotation_notes.md` for every decision and its confidence before relying on
the numbers.

## Files

| file | content |
| --- | --- |
| `inputs.jsonl.gz` | engine input: Mid-360 pedestrian detections (CODa PV-RCNN 32-beam checkpoint on three accumulated sweeps, run on CPU) and front-camera YOLO detections; see `docs/mid360.md` |
| `setup.json.gz` | calibration and odometry poses |
| `truth.jsonl.gz` | identity ground truth in the world frame (`truth3d` records, one per scan) |
| `labels.json` | the same labels as `[scan, detection index, person id]` |
| `edits.json`, `annotation_notes.md` | the corrections applied to the pre-labels, and why |
| `evaluate.sh` | replays the detections with a few tracker settings and scores them |

## How the labels were made

1. Pre-labels: offline linking of the detections in the world frame
   (`scripts/grandtour/labeling/link2.py`: frame-to-frame Hungarian
   matching, then stitching of track ends to later starts). This is not the
   engine tracker.
2. Review with all five Alphasense cameras (front, front-left, front-right,
   left, right; only for labelling, the tracker uses the front camera at
   most): per-track crop sheets (`sheet.py`), crops either side of every
   stitched gap (`stitch.py`), end/start merge candidates (`pairs.py`) and
   full camera frames with every box and ID (`frameview.py`). Behind the
   robot there is no camera; those tracks were judged from the point cloud
   and the trajectory only.
3. `apply.py` applies `edits.json` (one window, 3700-3900, so people crossing
   scan 3800 keep their id): 24 splits, 17 merges, and segments that could
   not be decided are excluded rather than guessed. Pieces shorter
   than 5 detections that were not merged into a person are excluded too.

## What the truth does and does not cover

- The boxes are the detector's boxes. People the detector never found are
  not in the truth, so coverage measures how well tracks are kept on
  detected people, not detection recall; false positives that were
  excluded are not penalised.
- ID switches are counted against the person identities above.
- Excluded segments are mostly people walking in groups, where the
  detector box moves between people. These are also hard for the tracker,
  so the scores are optimistic; window 2 has more such exclusions than
  window 1, which is part of why it scores better.
- 8 m radius and 20 s only; the rest of the recording is unlabelled.

## Results

`benchmarks/grandtour_eth2/evaluate.sh` (lidar_only, pedestrian motion prior,
1 m match gate), ID switches / coverage:

| setting | all (3700-3900) | window 1 | window 2 |
| --- | --- | --- | --- |
| base | 22 / 0.990 | 21 / 0.985 | 1 / 0.998 |
| birth suppression 1.0 m (for good) | 24 / 0.969 | 20 / 0.956 | 4 / 0.991 |
| birth suppression 1.5 m, persistence 3 | 25 / 0.982 | 19 / 0.975 | 5 / 0.996 |

With DINOv2 embeddings from the front camera
(`scripts/reid/embed_detections.py`, not included here to keep the files
small): appearance weight 10 alone 27 / 0.990, with persistent birth
suppression 25 / 0.982.

On window 1 alone birth suppression looked slightly better than base; with
window 2 added the order reverses. The differences are a few switches on a
small, AI-labelled sample, so none of these settings is clearly better.

## Licence

The recording is GrandTour data (MIT licence, Frey et al.). The detections,
calibration, poses, labels and notes in this directory are derived from it and
are redistributed under that MIT licence.
