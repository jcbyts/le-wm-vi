#!/usr/bin/env bash
set -u

cd /home/jake/repos/le-wm-vi || exit 1
mkdir -p logs /home/jake/.stable_worldmodel/overnight_fond_gaussian_b1_224norm

RUN="overnight_fond_gaussian_b1_224norm"
EPOCH="5"
GPU="${1:-0}"
CKPT="${RUN}/weights_epoch_${EPOCH}.pt"
OUT="pusht_eval_epoch${EPOCH}_with_recon_stack.txt"
FULL_LOG="logs/eval_${RUN}_epoch${EPOCH}_full_stack.log"
MEDIA_LOG="logs/posthoc_${RUN}_epoch${EPOCH}_recon.log"

{
  echo "$(date -Is) starting full PushT eval"
  echo "gpu=${GPU}"
  echo "checkpoint=${CKPT}"
  echo "output=/home/jake/.stable_worldmodel/${RUN}/${OUT}"
  echo "videos=/home/jake/.stable_worldmodel/${RUN}/env_*.mp4"
} >> "${FULL_LOG}"

CUDA_VISIBLE_DEVICES="${GPU}" conda run -n lewm python eval.py --config-name=pusht \
  policy="${CKPT}" \
  output.filename="${OUT}" \
  eval.img_size=224 \
  +eval.normalize_img=true \
  eval.num_eval=50 \
  eval.eval_budget=50 >> "${FULL_LOG}" 2>&1

status="$?"
echo "$(date -Is) full PushT eval finished with status ${status}" >> "${FULL_LOG}"
if [ "${status}" -ne 0 ]; then
  exit "${status}"
fi

{
  echo "$(date -Is) starting FOND reconstruction media eval"
  echo "gpu=${GPU}"
  echo "run=${RUN}"
  echo "epoch=${EPOCH}"
  echo "outputs=logs/posthoc_fond_media_eval"
} >> "${MEDIA_LOG}"

CUDA_VISIBLE_DEVICES="${GPU}" conda run -n lewm python posthoc_fond_media_eval.py \
  --run "${RUN}" \
  --epoch "${EPOCH}" \
  --step "$((EPOCH * 13933))" \
  --device cuda:0 \
  --num-eval 4 \
  --skip-planning >> "${MEDIA_LOG}" 2>&1

status="$?"
echo "$(date -Is) FOND reconstruction media eval finished with status ${status}" >> "${MEDIA_LOG}"
exit "${status}"
