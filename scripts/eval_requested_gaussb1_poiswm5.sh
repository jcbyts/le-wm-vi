#!/usr/bin/env bash
set -u

cd /home/jake/repos/le-wm-vi
mkdir -p logs

run_eval() {
  local gpu="$1"
  local run="$2"
  local ckpt="$3"
  local tag="$4"
  local log="logs/eval_${tag}.log"
  local out="pusht_eval_${tag}.txt"
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
  status=$?
  echo "END $(date -Is) ${run} status=${status}" >> "${log}"
  return ${status}
}

case "${1:-}" in
  gauss)
    run_eval 0 overnight_fond_gaussian_b1_224norm overnight_fond_gaussian_b1_224norm/weights_epoch_2.pt requested_gaussian_b1_epoch2
    ;;
  poiswm)
    run_eval 3 overnight_poiswm_rate5_embed384 overnight_poiswm_rate5_embed384/weights_epoch_5.pt requested_poiswm_rate5_epoch5
    ;;
  *)
    echo "usage: $0 gauss|poiswm" >&2
    exit 2
    ;;
esac
