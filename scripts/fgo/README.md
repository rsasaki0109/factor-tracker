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

## Status (not a usable result yet)

Base tracker switches are almost never fragments (a lost track restarted):
on CODa about 48 % are steals (a track moves to another person; in about half
of them both people are detected) and 30 % duplicates (two tracks on one
person, the match alternates).

| setting | CODa all range | CODa within 8 m | ETH-2 | mean coverage |
| --- | --- | --- | --- | --- |
| base tracker | 1091 | 147 | 22 | 0.962 |
| refine, `--max-gap 30 --birth-cost 8 --miss-cost 0.3 --keep-bonus 2` | 1039 | 132 | 37 | 0.951 |

The refiner helps a little on CODa and hurts ETH-2 and coverage. Open
problems: the observations inside an encounter are left out of the decision,
and a duplicate track is never removed.
