#!/usr/bin/env bash
set -u

cd /home/jake/repos/le-wm-vi
mkdir -p logs

RUNS=(
  overnight_fond_poisson_b0p1_224norm
  overnight_fond_poisson_b1_224norm
  overnight_fond_poisson_b3_224norm
  overnight_fond_gaussian_b0p1_224norm
  overnight_fond_gaussian_b1_224norm
  overnight_fond_gaussian_b3_224norm
  overnight_poiswm_rate5_embed384
  overnight_poiswm_rate10_embed384
)

CACHE=/home/jake/.stable_worldmodel/checkpoints
WAIT_LOG=logs/eval_overnight_latest_wait.log

latest_rel_ckpt() {
  local run="$1"
  local latest
  latest=$(find "${CACHE}/${run}" -maxdepth 1 -type f -name 'weights_epoch_*.pt' 2>/dev/null | sort -V | tail -n 1 || true)
  if [ -z "${latest}" ]; then
    return 1
  fi
  basename "${latest}" | sed "s#^#${run}/#"
}

wait_for_training_done() {
  while pgrep -af 'train.py .*overnight_' >/dev/null; do
    {
      echo "$(date -Is) waiting for overnight training to finish"
      pgrep -af 'train.py .*overnight_' || true
      echo
    } >> "${WAIT_LOG}"
    sleep 1800
  done
  echo "$(date -Is) overnight training appears finished; starting evals" >> "${WAIT_LOG}"
}

run_eval() {
  local gpu="$1"
  local run="$2"
  local ckpt="$3"
  local epoch
  epoch=$(basename "${ckpt}" .pt | sed 's/weights_epoch_//')
  local log="logs/eval_${run}_latest_epoch${epoch}.log"
  local out="pusht_eval_latest_epoch${epoch}.txt"
  {
    echo "START $(date -Is) ${run} ${ckpt} GPU=${gpu}"
    echo "CMD CUDA_VISIBLE_DEVICES=${gpu} conda run -n lewm python eval.py --config-name=pusht policy=${ckpt} output.filename=${out} eval.img_size=224 +eval.normalize_img=true eval.num_eval=50 eval.eval_budget=50"
  } >> "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" conda run -n lewm python eval.py --config-name=pusht \
    policy="${ckpt}" \
    output.filename="${out}" \
    eval.img_size=224 \
    +eval.normalize_img=true \
    eval.num_eval=50 \
    eval.eval_budget=50 \
    >> "${log}" 2>&1
  local status=$?
  echo "END $(date -Is) ${run} status=${status}" >> "${log}"
  return ${status}
}

worker() {
  local gpu="$1"
  shift
  for run in "$@"; do
    local ckpt
    if ! ckpt=$(latest_rel_ckpt "${run}"); then
      echo "SKIP $(date -Is) ${run}: no weights found" >> "logs/eval_overnight_latest_missing.log"
      continue
    fi
    run_eval "${gpu}" "${run}" "${ckpt}"
  done
}

case "${1:-main}" in
  main)
    wait_for_training_done
    screen -dmS eval_overnight_gpu0 bash scripts/eval_overnight_latest.sh worker 0 "${RUNS[0]}" "${RUNS[4]}"
    screen -dmS eval_overnight_gpu1 bash scripts/eval_overnight_latest.sh worker 1 "${RUNS[1]}" "${RUNS[5]}"
    screen -dmS eval_overnight_gpu2 bash scripts/eval_overnight_latest.sh worker 2 "${RUNS[2]}" "${RUNS[6]}"
    screen -dmS eval_overnight_gpu3 bash scripts/eval_overnight_latest.sh worker 3 "${RUNS[3]}" "${RUNS[7]}"
    ;;
  worker)
    shift
    gpu="$1"
    shift
    worker "${gpu}" "$@"
    ;;
  *)
    echo "usage: $0 [main|worker GPU RUN...]" >&2
    exit 2
    ;;
esac
