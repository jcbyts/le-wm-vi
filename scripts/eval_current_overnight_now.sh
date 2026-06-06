#!/usr/bin/env bash
set -u

cd /home/jake/repos/le-wm-vi
mkdir -p logs
CACHE=/home/jake/.stable_worldmodel/checkpoints
GPU=3

RUNS=(
  overnight_fond_poisson_b0p1_224norm
  overnight_fond_poisson_b3_224norm
  overnight_fond_gaussian_b1_224norm
  overnight_poiswm_rate5_embed384
  overnight_poiswm_rate10_embed384
)

latest_rel_ckpt() {
  local run="$1"
  local latest
  latest=$(find "${CACHE}/${run}" -maxdepth 1 -type f -name 'weights_epoch_*.pt' 2>/dev/null | sort -V | tail -n 1 || true)
  if [ -z "${latest}" ]; then
    return 1
  fi
  basename "${latest}" | sed "s#^#${run}/#"
}

for run in "${RUNS[@]}"; do
  if ! ckpt=$(latest_rel_ckpt "${run}"); then
    echo "SKIP $(date -Is) ${run}: no weights found" | tee -a logs/eval_current_overnight_now.log
    continue
  fi
  epoch=$(basename "${ckpt}" .pt | sed 's/weights_epoch_//')
  out="pusht_eval_current_epoch${epoch}.txt"
  log="logs/eval_current_${run}_epoch${epoch}.log"
  {
    echo "START $(date -Is) ${run} ${ckpt} GPU=${GPU}"
    echo "CMD CUDA_VISIBLE_DEVICES=${GPU} conda run -n lewm python eval.py --config-name=pusht policy=${ckpt} output.filename=${out} eval.img_size=224 +eval.normalize_img=true eval.num_eval=50 eval.eval_budget=50"
  } >> "${log}"
  CUDA_VISIBLE_DEVICES="${GPU}" conda run -n lewm python eval.py --config-name=pusht \
    policy="${ckpt}" \
    output.filename="${out}" \
    eval.img_size=224 \
    +eval.normalize_img=true \
    eval.num_eval=50 \
    eval.eval_budget=50 \
    >> "${log}" 2>&1
  status=$?
  echo "END $(date -Is) ${run} status=${status}" >> "${log}"
  echo "DONE $(date -Is) ${run} ${ckpt} status=${status}" | tee -a logs/eval_current_overnight_now.log
  if [ "${status}" -ne 0 ]; then
    echo "WARN ${run} eval failed; continuing with remaining evals" | tee -a logs/eval_current_overnight_now.log
  fi
done
