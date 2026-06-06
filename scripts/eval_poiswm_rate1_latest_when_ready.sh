#!/usr/bin/env bash
set -u

cd /home/jake/repos/le-wm-vi || exit 1
mkdir -p logs

RUN="poiswm_rate1_embed384_20260604"
GPU="0"
WAIT_LOG="logs/eval_${RUN}_latest_wait.log"

echo "$(date -Is) waiting to evaluate ${RUN}" >> "${WAIT_LOG}"

while pgrep -af "train.py .*output_model_name=${RUN}" >/dev/null; do
  echo "$(date -Is) ${RUN} is still training; waiting 10 minutes" >> "${WAIT_LOG}"
  sleep 600
done

latest="$(find "/home/jake/.stable_worldmodel/checkpoints/${RUN}" -maxdepth 1 -type f -name 'weights_epoch_*.pt' | sort -V | tail -n 1)"
if [ -z "${latest}" ]; then
  echo "$(date -Is) no checkpoint found for ${RUN}" >> "${WAIT_LOG}"
  exit 1
fi

epoch="$(basename "${latest}" .pt | sed 's/weights_epoch_//')"
ckpt="${RUN}/$(basename "${latest}")"
out="pusht_eval_requested_${RUN}_epoch${epoch}.txt"
eval_log="logs/eval_requested_${RUN}_epoch${epoch}.log"

{
  echo "$(date -Is) starting eval"
  echo "checkpoint=${ckpt}"
  echo "output=${out}"
} >> "${eval_log}"

CUDA_VISIBLE_DEVICES="${GPU}" conda run -n lewm python eval.py --config-name=pusht \
  policy="${ckpt}" \
  output.filename="${out}" \
  eval.img_size=224 \
  +eval.normalize_img=true \
  eval.num_eval=50 \
  eval.eval_budget=50 >> "${eval_log}" 2>&1

status="$?"
echo "$(date -Is) eval finished with status ${status}" >> "${eval_log}"
exit "${status}"
