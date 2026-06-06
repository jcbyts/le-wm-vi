#!/usr/bin/env bash
set -u

if [ "$#" -ne 4 ]; then
  echo "usage: $0 RUN_NAME GPU WAIT_PATTERN OUT_PREFIX" >&2
  exit 2
fi

RUN="$1"
GPU="$2"
WAIT_PATTERN="$3"
OUT_PREFIX="$4"

cd /home/jake/repos/le-wm-vi || exit 1
mkdir -p logs

WAIT_LOG="logs/eval_${RUN}_with_videos_wait.log"
echo "$(date -Is) waiting to evaluate ${RUN} on GPU ${GPU}" >> "${WAIT_LOG}"

while pgrep -af "${WAIT_PATTERN}" >/dev/null; do
  echo "$(date -Is) ${RUN} training still active; waiting 10 minutes" >> "${WAIT_LOG}"
  sleep 600
done

latest="$(find "/home/jake/.stable_worldmodel/checkpoints/${RUN}" -maxdepth 1 -type f -name 'weights_epoch_*.pt' | sort -V | tail -n 1)"
if [ -z "${latest}" ]; then
  echo "$(date -Is) no checkpoint found for ${RUN}" >> "${WAIT_LOG}"
  exit 1
fi

epoch="$(basename "${latest}" .pt | sed 's/weights_epoch_//')"
ckpt="${RUN}/$(basename "${latest}")"
out="${OUT_PREFIX}_${RUN}_epoch${epoch}_with_videos.txt"
eval_log="logs/${OUT_PREFIX}_${RUN}_epoch${epoch}_with_videos.log"

{
  echo "$(date -Is) starting eval with planning videos"
  echo "checkpoint=${ckpt}"
  echo "output=${out}"
  echo "videos_dir=/home/jake/.stable_worldmodel/${RUN}"
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
