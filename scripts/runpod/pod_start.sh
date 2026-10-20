#!/usr/bin/env bash
# Start command for an unattended CODa PV-RCNN pod: runs coda_pvrcnn.sh once
# and serves only its log and result, read-only, on port 8000
# (/run.log, /coda_pvrcnn_dets.tar.gz). Nothing is executed on request.
repo=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p /workspace/www
(cd /workspace/www && nohup python3 -m http.server 8000 > /dev/null 2>&1 &)
bash "$repo/scripts/runpod/${RUN_SCRIPT:-coda_pvrcnn.sh}" > /workspace/www/run.log 2>&1
echo "EXIT $?" >> /workspace/www/run.log
cp /workspace/coda_pvrcnn_dets.tar.gz /workspace/www/ 2> /dev/null
sleep infinity
