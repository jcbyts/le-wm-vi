#!/usr/bin/env bash
set -u

cd /home/jake/repos/le-wm-vi
mkdir -p logs

run_job() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log="logs/${name}.screen.log"
  {
    echo "START $(date -Is) ${name} GPU=${gpu}"
    echo "CMD CUDA_VISIBLE_DEVICES=${gpu} conda run -n lewm python train.py $* output_model_name=${name} subdir=${name}"
  } >> "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" conda run -n lewm python train.py "$@" \
    output_model_name="${name}" \
    subdir="${name}" \
    trainer.devices=1 \
    trainer.max_epochs=5 \
    wandb.enabled=true \
    wandb.config.name="${name}" \
    wandb.config.id="${name}" \
    wandb.config.resume=allow \
    >> "${log}" 2>&1
  local status=$?
  echo "END $(date -Is) ${name} status=${status}" >> "${log}"
  return ${status}
}

fond_common=(
  --config-name fond
  loss.infer_backprop=true
  loss.target_scheme=online_filtering
  loss.infer_objective=free_energy
  loss.k_bptt=1
  loss.log_viz=false
  model.k_inner=2
  model.decoder.patch_size=16
  model.decoder.output_activation=identity
  img_size=224
  normalize_img=true
  loader.batch_size=32
  monitor.enabled=false
)

poiswm_common=(
  model=poiswm
  model.encoder.size=small
  embed_dim=384
  loss.beta=1.0
  loader.batch_size=64
  monitor.enabled=false
)

case "${1:-}" in
  worker0)
    run_job 0 overnight_fond_poisson_b0p1_224norm "${fond_common[@]}" model=fond_poisson loss.pred_loss=exact_kl loss.beta=0.1
    run_job 0 overnight_fond_poisson_b1_224norm "${fond_common[@]}" model=fond_poisson loss.pred_loss=exact_kl loss.beta=1.0
    ;;
  worker1)
    run_job 1 overnight_fond_poisson_b3_224norm "${fond_common[@]}" model=fond_poisson loss.pred_loss=exact_kl loss.beta=3.0
    run_job 1 overnight_fond_gaussian_b0p1_224norm "${fond_common[@]}" model=fond_gaussian loss.pred_loss=quadratic_fisher loss.beta=0.1
    ;;
  worker2)
    run_job 2 overnight_fond_gaussian_b1_224norm "${fond_common[@]}" model=fond_gaussian loss.pred_loss=quadratic_fisher loss.beta=1.0
    run_job 2 overnight_fond_gaussian_b3_224norm "${fond_common[@]}" model=fond_gaussian loss.pred_loss=quadratic_fisher loss.beta=3.0
    ;;
  worker3)
    run_job 3 overnight_poiswm_rate5_embed384 "${poiswm_common[@]}" model.target_rate=5.0
    run_job 3 overnight_poiswm_rate10_embed384 "${poiswm_common[@]}" model.target_rate=10.0
    ;;
  *)
    echo "usage: $0 worker0|worker1|worker2|worker3" >&2
    exit 2
    ;;
esac
