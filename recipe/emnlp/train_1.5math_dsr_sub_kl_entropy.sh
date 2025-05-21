#!/usr/bin/env bash
set -euxo pipefail

project_name='self_distillation_emnlp'

adv_estimator=grpo

kl_coef=0.001
kl_loss_coef=0.001
entropy_coef=0.001
kl_loss_type=low_var_kl
val_temperature=0.6
train_temperature=0.6

clip_ratio_low=0.2
clip_ratio_high=0.28

enable_overlong_buffer=False
overlong_buffer_len=512
overlong_penalty_factor=1.0

enable_filter_groups=True
filter_groups_metric=seq_final_reward
fill_to_train_bsz=True
train_prompt_bsz=256 # 512 works for 7B n = 8
multiplier=3 # 3 works for 7B n = 8
gen_prompt_bsz=$((train_prompt_bsz * multiplier))
train_prompt_mini_bsz=128
train_micro_batch_size=128
val_batch_size=530

#num_epochs=10
num_epochs=200

num_rollouts=8
val_kwargs_n=${num_rollouts}
n_resp_per_prompt=${num_rollouts}
n_advantage_tracking=${num_rollouts}

use_token_level_loss=False # TRUE for DAPO token level loss

# Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
#NNODES=${NNODES:-4}
NNODES=1

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"/fast/pmayilvahanan/"}
MODEL_PATH=Qwen/Qwen2.5-Math-1.5B
dataset_name='dsr_sub'

exp_name="${MODEL_PATH}_${dataset_name}_n_${n_resp_per_prompt}_bsz_${train_prompt_bsz}_epochs_${num_epochs}_kl_coef_${kl_coef}_kl_loss_coef_${kl_loss_coef}_entropy_coef_${entropy_coef}"

#MODEL_PATH=/fast/pmayilvahanan/post_training/verl_checkpoints/dapo/DAPO-Qwen2.5-7B-Math-DAPO/global_step_8
resume_mode=auto
resume_from_path=False
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/post_training/verl_checkpoints/self_distillation_emnlp/${exp_name}"}
#TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/datasets/${dataset_name}/train.parquet"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/datasets/rl_training_one_example/dsr_sub.parquet"} # change this to above


# Algorithm
## Train
max_prompt_length=$((1024))
max_response_length=$((1024 * 3))
## Validation
val_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout


# Mathematically equivalent
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null
offload=False
n_gpus_per_node=8

# ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
#     --working-dir "${WORKING_DIR}" \
python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files=[/fast/pmayilvahanan/datasets/openai_math/test.parquet,/fast/pmayilvahanan/datasets/aime_2024/test.parquet,/fast/pmayilvahanan/datasets/olympiad_bench/test.parquet,/fast/pmayilvahanan/datasets/minervamath/test.parquet,/fast/pmayilvahanan/datasets/amc23/test.parquet,/fast/pmayilvahanan/datasets/aime_2025/test.parquet] \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_batch_size} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.n_advantage_tracking=${n_advantage_tracking} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_kwargs_n} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.fill_to_train_bsz=${fill_to_train_bsz} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coef} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_token_level_loss=${use_token_level_loss} \
    actor_rollout_ref.actor.use_token_level_loss=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.temperature=${train_temperature} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.val_kwargs.top_k="${val_top_k}" \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0\
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes="${NNODES}" \
    +trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=25 \
    trainer.track_advantages=False \
    trainer.track_advantages_freq=5 \
    trainer.total_epochs=${num_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=${resume_mode} \
    trainer.resume_from_path=${resume_from_path} 