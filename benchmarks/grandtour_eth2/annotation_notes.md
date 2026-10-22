# ETH-2 labelling notes (frames 3700-3800, within 8 m)
Decisions per pre-label track (v2.json). "ok" = one person throughout.
- 1 ok (dark jacket, light jeans; standing, walks off ~3785)
- 3 ok (walks out of view on the left, short)
- 4 ok, low confidence: behind the robot, no camera; point cloud stays a person-shaped static cluster, leaves ~3822
- 6 ok (beige jacket, crossbody bag, walks from ~3745, ends 3773)
- 5 ok (static at ~7 m, woman with phone next to a tall man; position constant)
- 7 ok, medium (dark, static at ~3 m, starts walking ~3790; geometry smooth)
- 8 SPLIT at 3774: 3700-3773 = woman in long blue down coat; 3774+ = the beige-jacket walker of track 6 (6 ends 3773) -> merge 8[3774:] into 6
- 9 ok (man, glasses, light jacket, crossbody bag, standing ~6 m)
- 10 ok, medium (woman, dark down jacket, walks away 3.7 -> 7 m into the crowd; crops crowded after 3730)
- 11 ok, medium (blond woman with cup, standing ~2.8 m; people pass in front 3763-3769)
- 12 ok (man, dark down jacket, phone, standing ~6 m)
- 13 ok (man with bag walking towards the robot, left camera)
- 15 ok, low (beige jacket at ~4 m, static, fragmented; different from 6 by position)
- 16 ok, low (blue quilted coat, light trousers, ~2.7 m, static; 0.85 m from track 8's blue-coat woman)
- 17 ok (man, dark coat, phone, standing ~2.9 m)
- 21 SPLIT at 3789: 3703-3775 = long-haired woman walking away; 3789 single detection = a man in a light jacket (different person)
## Stitch checks (gaps > 2 frames inside a track)
- 4 (6 gaps): no camera; static cluster, jumps <= 0.6 m: keep
- 7, 9, 10, 12, 15 (2), 16 (2), 22 (2), 27: same person on both sides: keep
- 8 3760->3774: split (see above)
- 11 3745->3757, 3770->3783, 3794->3805: keep. Crops before 3770 and 3794 show people walking in front of her (beige walker of track 6 at 3766, frame view confirms 6 is the closer box); the detection stays at her distance.
- 21 3758->3773: IGNORE 3773-3775 (crop shows another walker in jeans; cannot tell), 3789 split as above
- 25 3742->3747: keep, medium (dark, left camera)
- 38, 41 (2), 43, 49, 56, 64 (3), 65, 71 (2), 77, 100 (2), 116 3787->3790: same person on both sides: keep
- 45 (3 gaps), 68, 69, 96, 121: no camera, small jumps: keep
- 52: IGNORE (next to a pillar, 1.0 m jump across a 15-frame gap, crops unclear)
- 74 3740->3754: different man after the gap (light jacket, bag) -> 3754-3758 IGNORE; 3773+ another person again -> new person
- 76 3746->3751, 95 3770->3774: the beige walker passes in front; detection stays at the standing person's distance: keep
- 81 3750->3765: after the gap a different-looking person, 0.8 m jump -> 3765+ IGNORE
- 107 3790->3799: over-exposed/dark crops: keep, low
- 116 3793->3801: different person after the gap (0.7 m jump) -> 3801+ new person
## Merge candidates (A ends, B starts nearby)
- merge 2+40 (man with phone), 40+66, 66+73 (same man walking away), 14+26, 22+131 (walking woman), 25+96 (dark, stairs, medium),
  27+67 (woman with printed tote bag), 67+95 (same tote bag), 35+43, 35+57 (man in white/black jacket; 57 sits in 43's gap), 91+94, 98+129 (medium), 112+116
- no merge: 3/69, 10/106, 16/76 (76 is a man with a backpack and cap; 16 a blue quilted coat), 21/147, 21/155, 22/176, 27/78, 28/98, 38/98, 38/103, 44/68, 44/69, 73/96, 78/95, 95/130, 106/117
- 103 IGNORE (short duplicate near 22's woman); 154 NOT A PERSON (pole)
## Full-sheet checks of the remaining long tracks
- 22 ok (woman in patterned long coat, standing, starts walking ~3770), 23 ok, 24 ok (medium), 25 ok, 27 ok, 38 ok, 41 ok, 56 ok, 64 ok, 65 ok, 71 ok, 78 ok (blue down coat at 3.6 m; not merged), 81 ok up to 3750, 82 ok, 94 ok, 95 ok (tote-bag woman; the beige walker passes in front 3768-3770), 98 ok, 107 ok (low: box at image edge), 113 ok, 116 ok
- 28: SPLIT at 3735, 3735+ IGNORE (box covers a pair walking to the door; switches between them at 3734->3735, frame views 3730/3738)
- 100: SPLIT at 3797, 3797+ IGNORE (walking man before the gap, a static detection after it)
- 101: IGNORE (two people walking side by side; red jacket 3773-3777, long-haired woman 3778-3782)
- 44, 45, 69: no camera, kept as single people (low)

# Second window (frames 3801-3900)
Same process; edits.json now covers 3700-3900 as one window, so people crossing 3800 keep their id (window-1 labels unchanged).
More of this window is walking groups, so more segments were excluded than in the first.
## Tracks
- ok: 1, 4 (no camera), 5 (box sometimes includes the tall man next to her; position static), 7 (to 3834), 8, 9, 11, 41, 64, 99 (man in white shirt walking towards the robot), 107 (low), 128 (to 3830), 130 (low), 131 (as part of 22, medium), 142, 164, 167, 170 (medium), 171 (no camera, low), 176, 191, 192, 196 (person by the sign), 199, 201 (low), 203 (low)
- 12: SPLIT at 3842 (walking man before; different legs after a 0.3 m jump across 11 frames) -> 3842+ new person, low
- 24: SPLIT at 3857 (box in a group by the wall; after the gap a person by the sign) -> 3857+ new person, low
- 113: standing man 3800-3839; SPLIT at 3854 -> 3854-3893 new person (legs, consistent), 3899+ IGNORE
- 173: 3822-3856 the box moves over several people at 8 m -> IGNORE; 3857+ woman in patterned vest, standing -> new person
- 206+214+218: one walker (long hair, backpack) towards the robot -> merge, medium
- 185+189 merge (189 split at 3861)
## Excluded
- 7@3845, 71@3821, 94@3820 (walker with backpack before, someone in jeans after), 116@3813 (woman before, a walker after), 128@3845, 129@3811 (dark walker before, grey coat after), 137@3809 (red-shirt man before, light jacket after), 140@3830 (no camera, 0.8 m jump), 189@3861, 195@3871
- 153, 165, 169: whole tracks; the box moves between people in a walking group
- 154 stays excluded: confirmed a pole (the white-shirt man of 99 passes in front of it at 3823-3832)
- no merge: 7/198, 8/176, 22/176, 24/206, 94/198, 94/199, 100/199, 154/196, 197/203
