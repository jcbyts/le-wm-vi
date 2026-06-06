#!/usr/bin/env bash
set -u
cd /home/jake/repos/le-wm-vi || exit 1
mkdir -p logs /home/jake/.stable_worldmodel/lewm_baseline_20260604_1418 /home/jake/.stable_worldmodel/poiswm_rate1_embed384_20260604

run_eval() {
  local gpu="$1"
  local run="$2"
  local ckpt="$3"
  local tag="$4"
  local out="pusht_eval_${tag}.txt"
  local log="logs/eval_${tag}.log"
  {
    echo "$(date -Is) starting eval"
    echo "gpu=${gpu}"
    echo "checkpoint=${ckpt}"
    echo "output=/home/jake/.stable_worldmodel/${run}/${out}"
    echo "videos=/home/jake/.stable_worldmodel/${run}/env_*.mp4"
  } >> "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" conda run -n lewm python eval.py --config-name=pusht \
    policy="${ckpt}" \
    output.filename="${out}" \
    eval.img_size=224 \
    +eval.normalize_img=true \
    eval.num_eval=50 \
    eval.eval_budget=50 >> "${log}" 2>&1
  local status="$?"
  echo "$(date -Is) eval finished with status ${status}" >> "${log}"
  return "${status}"
}

run_eval 0 lewm_baseline_20260604_1418 lewm_baseline_20260604_1418/weights_epoch_10.pt lewm_baseline_20260604_1418_epoch10 &
pid1="$!"
run_eval 1 poiswm_rate1_embed384_20260604 poiswm_rate1_embed384_20260604/weights_epoch_5.pt poiswm_rate1_embed384_20260604_epoch5 &
pid2="$!"

wait "${pid1}"
s1="$?"
wait "${pid2}"
s2="$?"

if [ "${s1}" -ne 0 ] || [ "${s2}" -ne 0 ]; then
  exit 1
fi
