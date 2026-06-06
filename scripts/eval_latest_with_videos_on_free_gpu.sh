#!/usr/bin/env bash
set -u

if [ "$#" -ne 2 ]; then
  echo "usage: $0 RUN_NAME OUT_PREFIX" >&2
  exit 2
fi

RUN="$1"
OUT_PREFIX="$2"

cd /home/jake/repos/le-wm-vi || exit 1
mkdir -p logs "/home/jake/.stable_worldmodel/${RUN}"

WAIT_LOG="logs/eval_${RUN}_with_videos_free_gpu_wait.log"
echo "$(date -Is) waiting for a free GPU to evaluate ${RUN}" >> "${WAIT_LOG}"

pick_free_gpu() {
  for gpu in 0 1 2 3; do
    if ! nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader -i "${gpu}" 2>/dev/null | grep -q .; then
      echo "${gpu}"
      return 0
    fi
  done
  return 1
}

GPU=""
while [ -z "${GPU}" ]; do
  GPU="$(pick_free_gpu || true)"
  if [ -z "${GPU}" ]; then
    echo "$(date -Is) no free GPU yet; waiting 5 minutes" >> "${WAIT_LOG}"
    sleep 300
  fi
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
  echo "gpu=${GPU}"
  echo "checkpoint=${ckpt}"
  echo "output=${out}"
  echo "results_dir=/home/jake/.stable_worldmodel/${RUN}"
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
