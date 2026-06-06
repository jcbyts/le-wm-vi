#!/usr/bin/env bash
set -u

cd /home/jake/repos/le-wm-vi
mkdir -p logs
status_log=logs/overnight_hourly_status.log

while true; do
  {
    echo "===== hourly status $(date -Is) ====="
    echo "-- screens --"
    screen -ls || true
    echo "-- gpu compute --"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
    echo "-- active overnight train processes --"
    ps -u jake -o pid,etime,pcpu,pmem,cmd | grep -E 'train.py .*overnight_' | grep -v grep || true
    echo "-- latest checkpoints/configs --"
    find /home/jake/.stable_worldmodel/checkpoints -maxdepth 2 -type f -path '*overnight_*' -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort || true
    echo "-- recent watchdog kills --"
    find logs -maxdepth 1 -name 'overnight_*.killed' -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort || true
    echo
  } >> "${status_log}" 2>&1
  sleep 3600
done
