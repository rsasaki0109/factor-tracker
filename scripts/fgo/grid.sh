#!/usr/bin/env bash
# Grid over refine.py options on the dev sets; one line per setting:
# CODa all-range switches (sum), CODa within 8 m (sum), ETH-2 switches, mean coverage.
# usage: grid.sh <data-dir> <work-dir> "<options>" ["<options>" ...]
set -euo pipefail
data=$1; work=$2; shift 2
here=$(cd "$(dirname "$0")" && pwd); mkdir -p "$work"
for options in "$@"; do
  w=$work/$(echo "$options" | tr -c 'a-zA-Z0-9.' '_')
  "$here/dev_eval.sh" "$data" "$w" $options > "$w.txt" 2>&1 || { echo "FAILED $options"; continue; }
  # reuse base runs
  python3 - "$w.txt" "$options" <<'PY'
import sys
rows = [l.split() for l in open(sys.argv[1]) if l.strip() and not l.startswith("set")]
def tot(pred, col):
    return sum(int(r[col]) for r in rows if pred(r))
coda_all = lambda r: r[0].startswith("coda") and r[1] == "all"
coda_8 = lambda r: r[0].startswith("coda") and r[1] == "around8"
eth = lambda r: r[0] == "eth2"
cov_b = sum(float(r[3]) for r in rows) / len(rows); cov_r = sum(float(r[5]) for r in rows) / len(rows)
print(f"{sys.argv[2]:60s} coda_all {tot(coda_all,2):5d}->{tot(coda_all,4):5d}  coda_8m {tot(coda_8,2):4d}->{tot(coda_8,4):4d}  eth2 {tot(eth,2):3d}->{tot(eth,4):3d}  cov {cov_b:.3f}->{cov_r:.3f}")
PY
done
