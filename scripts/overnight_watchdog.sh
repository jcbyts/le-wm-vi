#!/usr/bin/env bash
set -u

cd /home/jake/repos/le-wm-vi
mkdir -p logs
monitor_log=logs/overnight_watchdog.log
bad_pattern='Traceback|CUDA out of memory|RuntimeError|Exception|loss[[:space:]/_A-Za-z0-9.-]*[:=][[:space:]]*nan|(^|[^[:alpha:]])nan([^[:alpha:]]|$)'

while true; do
  echo "===== $(date -Is) =====" >> "${monitor_log}"
  screen -ls >> "${monitor_log}" 2>&1 || true
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits >> "${monitor_log}" 2>&1 || true
  for log in logs/overnight_*.screen.log; do
    [ -e "${log}" ] || continue
    name=$(basename "${log}" .screen.log)
    killed="logs/${name}.killed"
    if [ ! -e "${killed}" ] && tail -n 300 "${log}" | grep -Eiq "${bad_pattern}"; then
      echo "KILL $(date -Is) ${name}: detected failure pattern in ${log}" | tee -a "${monitor_log}"
      pkill -TERM -f "output_model_name=${name}" || true
      touch "${killed}"
    fi
  done
  sleep 900
done
