# Data format

`factor-tracker` reads and writes JSONL: one JSON object per line. Every record
contains

- `schema_version` — currently `1`. A record with a different version is
  rejected rather than guessed at.
- `type` — a discriminator string.

Blank lines are ignored. Times are integer nanoseconds (`timestamp_ns`) on one
shared, pre-synchronized epoch. Arrays whose lengths are specified below are
validated on read.

## Units and conventions

| Quantity | Unit | Ordering |
| --- | --- | --- |
| Timestamp | nanoseconds (`i64`) | — |
| 2D bbox | pixels | `[x_min, y_min, x_max, y_max]` |
| 2D measurement vector | pixels | `[cx, cy, w, h]` |
| 2D covariance | pixels² | row-major `4x4`, length 16 |
| 2D velocity | pixels / second | `[vx, vy]` |
| 2D state | mixed | `[cx, cy, w, h, vcx, vcy, vw, vh]` |
| 2D track covariance | mixed | row-major `8x8`, length 64 |
| 3D position | meters | `[x, y, z]` |
| 3D covariance | meters² | row-major `3x3`, length 9 |
| 3D velocity | meters / second | `[vx, vy, vz]` |
| 3D state | mixed | `[x, y, z, vx, vy, vz]` |
| 3D track covariance | mixed | row-major `6x6`, length 36 |

Row-major means the concatenation of rows: `m[0][0], m[0][1], ..., m[n-1][n-1]`.

## Observation records

### `bbox2d_batch`

One sensor's 2D detections at one timestamp.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | must be `1` |
| `sensor_id` | integer | opaque sensor id |
| `sequence` | integer | per-sensor counter; must strictly increase over accepted batches |
| `timestamp_ns` | integer | observation time |
| `frame_id` | string | image frame; must equal the tracker's frame |
| `observations` | array | detections below |

Each detection:

| Field | Type | Meaning |
| --- | --- | --- |
| `bbox` | 4 numbers | `[x_min, y_min, x_max, y_max]` pixels, width/height > 0 |
| `score` | number | detector confidence in `[0, 1]` |
| `class` | integer or absent | optional class id |
| `covariance` | 16 numbers | row-major `4x4` for `[cx, cy, w, h]`, positive definite |

Example:

```json
{"type":"bbox2d_batch","schema_version":1,"sensor_id":1,"sequence":0,"timestamp_ns":0,"frame_id":"camera","observations":[{"bbox":[100.0,50.0,160.0,130.0],"score":0.95,"class":1,"covariance":[4.0,0.0,0.0,0.0,0.0,4.0,0.0,0.0,0.0,0.0,4.0,0.0,0.0,0.0,0.0,4.0]}]}
```

### `position3d_batch`

One sensor's 3D positions at one timestamp. Positions must already be expressed
in the tracker's tracking frame.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | must be `1` |
| `sensor_id` | integer | opaque sensor id |
| `sequence` | integer | per-sensor counter |
| `timestamp_ns` | integer | observation time |
| `frame_id` | string | coordinate frame; must equal the tracker's frame |
| `observations` | array | detections below |

Each detection:

| Field | Type | Meaning |
| --- | --- | --- |
| `position` | 3 numbers | `[x, y, z]` meters |
| `class` | integer or absent | optional class id |
| `covariance` | 9 numbers | row-major `3x3` in meters², positive definite |

## Track records

### `bbox2d_tracks`

One line per tracked timestamp.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | `1` |
| `timestamp_ns` | integer | state time |
| `tracks` | array | snapshots below |

Each track:

| Field | Type | Meaning |
| --- | --- | --- |
| `track_id` | integer | never reused by the tracker |
| `lifecycle` | string | `tentative`, `confirmed` or `lost` |
| `state_timestamp_ns` | integer | time the state estimate refers to |
| `last_observation_timestamp_ns` | integer or null | last observation time |
| `last_update_sensor` | integer or null | sensor of the last observation |
| `contributing_sensors` | array of integers | distinct sensors (bounded) |
| `class` | integer or null | known class |
| `updated` | bool | updated by the last processed batch |
| `prediction_only` | bool | estimate is a prediction (`state_timestamp != last_observation_timestamp`) |
| `observation_times_count` | integer | distinct observation timestamps seen |
| `bbox` | 4 numbers | `[x_min, y_min, x_max, y_max]` pixels |
| `velocity` | 2 numbers | `[vx, vy]` pixels / second |
| `state` | 8 numbers | full 2D state |
| `covariance` | 64 numbers | row-major `8x8` |

### `position3d_tracks`

One line per tracked timestamp.

| Field | Type | Meaning |
| --- | --- | --- |
| `track_id` | integer | never reused |
| `lifecycle` | string | `tentative`, `confirmed` or `lost` |
| `state_timestamp_ns` | integer | state time |
| `last_observation_timestamp_ns` | integer or null | last observation time |
| `last_update_sensor` | integer or null | sensor of the last observation |
| `contributing_sensors` | array of integers | distinct sensors (bounded) |
| `class` | integer or null | known class |
| `updated` | bool | updated by the last processed batch |
| `prediction_only` | bool | estimate is a prediction |
| `observation_times_count` | integer | distinct observation timestamps |
| `position` | 3 numbers | meters |
| `velocity` | 3 numbers | meters / second |
| `state` | 6 numbers | `[x, y, z, vx, vy, vz]` |
| `covariance` | 36 numbers | row-major `6x6` |

`score` is never promoted to a track existence probability. There is no
existence-probability field because none is estimated.

## Ground-truth records

Ground truth is written by the scenario generator and is **never** given to a
tracker; it is only used for evaluation and visualization. Truth ids are
independent from track ids.

### `truth2d`

```json
{"type":"truth2d","schema_version":1,"timestamp_ns":0,"objects":[{"truth_id":1,"class":1,"bbox":[62.4,147.4,182.6,216.9],"present":true,"detected":true}]}
```

| Field | Meaning |
| --- | --- |
| `truth_id` | ground-truth object id |
| `class` | class id |
| `bbox` | `[x_min, y_min, x_max, y_max]` pixels |
| `present` | object existed at this time |
| `detected` | the synthetic sensor detected it (a miss is `false`) |

### `truth3d`

```json
{"type":"truth3d","schema_version":1,"timestamp_ns":0,"objects":[{"truth_id":1,"class":1,"position":[0.0,0.0,1.0],"present":true,"detected":true}]}
```

## Schema versioning

The schema version is bumped whenever the meaning or ordering of a field
changes. Readers reject unknown versions instead of guessing. The version is
present on every record so mixed or truncated files are detected.
