#!/usr/bin/env bash
set -u
cd /home/jake/repos/le-wm-vi || exit 1
RUN="lewm_baseline_20260604_1418"
TARGET="/home/jake/.stable_worldmodel/checkpoints/${RUN}/weights_epoch_10.pt"
LOG="logs/${RUN}_stop_at_epoch10.log"
echo "$(date -Is) waiting for ${TARGET}" >> "${LOG}"
while [ ! -f "${TARGET}" ]; do
  sleep 300
  echo "$(date -Is) still waiting for epoch 10" >> "${LOG}"
done
pid="$(pgrep -f "train.py .*output_model_name=${RUN}" | head -n 1 || true)"
if [ -n "${pid}" ]; then
  echo "$(date -Is) found epoch 10; sending SIGINT to pid ${pid}" >> "${LOG}"
  kill -INT "${pid}"
  sleep 60
  if kill -0 "${pid}" 2>/dev/null; then
    echo "$(date -Is) pid ${pid} still alive after SIGINT; sending SIGTERM" >> "${LOG}"
    kill -TERM "${pid}"
  fi
else
  echo "$(date -Is) epoch 10 exists and no active ${RUN} train process found" >> "${LOG}"
fi
