#!/usr/bin/env bash
set -u
cd /home/jake/repos/le-wm-vi || exit 1
RUN="poiswm_rate1_embed384_20260604"
LOG="logs/${RUN}_continue_to10_manager.log"
BASE="/home/jake/.stable_worldmodel/checkpoints/${RUN}/weights_epoch_5.pt"
FINAL="/home/jake/.stable_worldmodel/checkpoints/${RUN}/weights_epoch_10.pt"
echo "$(date -Is) waiting for current ${RUN} run to finish epoch 5" >> "${LOG}"
while pgrep -af "train.py .*output_model_name=${RUN}" >/dev/null; do
  echo "$(date -Is) current ${RUN} process still running" >> "${LOG}"
  sleep 300
done
if [ ! -f "${BASE}" ]; then
  echo "$(date -Is) missing ${BASE}; cannot continue" >> "${LOG}"
  exit 1
fi
if [ -f "${FINAL}" ]; then
  echo "$(date -Is) ${FINAL} already exists; nothing to do" >> "${LOG}"
  exit 0
fi
echo "$(date -Is) launching continuation from ${BASE}" >> "${LOG}"
CUDA_VISIBLE_DEVICES=0 conda run -n lewm python train.py \
  model=poiswm \
  model.encoder.size=small \
  embed_dim=384 \
  model.target_rate=1.0 \
  loss.beta=1.0 \
  loader.batch_size=64 \
  monitor.enabled=false \
  trainer.devices=1 \
  trainer.max_epochs=5 \
  +init_weights="${RUN}/weights_epoch_5.pt" \
  +ckpt_epoch_offset=5 \
  output_model_name="${RUN}" \
  subdir="${RUN}" \
  wandb.enabled=true \
  wandb.config.name="${RUN}" \
  wandb.config.id="${RUN}" \
  wandb.config.resume=allow >> "logs/${RUN}_continue_to10.log" 2>&1
status="$?"
echo "$(date -Is) continuation finished with status ${status}" >> "${LOG}"
exit "${status}"
