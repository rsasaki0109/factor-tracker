# Fixed-lag identity refinement (prototype)

`refine.py` re-decides, after the fact, which pieces of the engine's tracks
belong to the same person: tracks are cut at missed frames and at the start
and end of close encounters (< 1 m), and the pieces before and after are
linked by one global assignment. A link's cost is the change in the
trajectory's factor-graph cost (negative log-likelihood under a
constant-velocity model; for this linear-Gaussian graph computed exactly by a
Kalman pass) against starting a new person, with a bonus for the base
tracker's own links. Pieces inside an encounter take the identity of the
piece after it.

`switch_types.py` sorts a tracker's ID switches by cause; `dev_eval.sh` and
`grid.sh` score the refiner against the base tracker on five CODa windows
(real detections) and GrandTour ETH-2.

## Status

The usable version is the Rust `FixedLagRefiner` (`--fgo-lag`, see
`docs/evaluation.md`, "Fixed-lag re-association"). These scripts are the
prototypes it came from:

- `refine.py` (piece linking across gaps and encounters) barely helps: the
  base tracker's switches are not fragments.
- `window.py` (multi-frame re-association, the method the Rust refiner
  implements) found the cost settings. Its `--lag` mode is optimistic: it
  keeps each engine track's later detections in view.
