#!/usr/bin/env bash
set -euxo pipefail

# =====================
# 1. Project and Algorithm Settings
# =====================
kl_coef=0.001  # KL divergence coefficient for regularization
kl_loss_coef=0.001  # KL coefficient for GRPO
project_name='training_llm_rl_exploration'  # Name of the project
adv_estimator=grpo  # Advantage estimator type (e.g., grpo, ppo, etc.)
clip_ratio_low=0.2  # PPO lower clip ratio
clip_ratio_high=0.28  # PPO upper clip ratio
enable_overlong_buffer=False  # Enable buffer for overlong sequences
overlong_buffer_len=512  # Buffer length for overlong sequences
overlong_penalty_factor=1.0  # Penalty factor for overlong sequences
enable_filter_groups=True  # Enable filter groups for training
filter_groups_metric=seq_final_reward  # Metric for filtering groups
fill_to_train_bsz=True  # Fill to training batch size
remove_previous_ckpt_in_save=True

# =====================
# 2. Batch Sizes and Training Parameters
# =====================
train_prompt_bsz=256  # Training batch size (e.g., 512 works for 7B n=8)
multiplier=3  # Multiplier for generation batch size (e.g., 3 for 7B n=8)
gen_prompt_bsz=$((train_prompt_bsz * multiplier))  # Generation batch size
ppo_mini_batch_size=128  # PPO mini-batch size
ppo_micro_batch_size=64  # PPO micro-batch size
val_batch_size=530  # Validation batch size
num_epochs=10  # Number of training epochs
num_rollouts=8  # Number of rollouts per prompt (k in pass@k)
val_kwargs_n=${num_rollouts}  # Validation: number of responses per prompt
n_resp_per_prompt=${num_rollouts}  # Number of responses per prompt
n_advantage_tracking=${num_rollouts}  # Number of advantage tracking samples
use_token_level_loss=False  # TRUE for DAPO token level loss
NNODES=1  # Number of nodes for distributed training

# =====================
# 3. Paths and Dataset
# =====================
RAY_DATA_HOME=${RAY_DATA_HOME:-"/u/rfechner"}  # Base directory for data and checkpoints
MODEL_PATH=Qwen/Qwen2.5-1.5B  # Model name or path
dataset_name='math'  # Name of the dataset
exp_name="${MODEL_PATH}_${dataset_name}_n_${n_resp_per_prompt}_bsz_${train_prompt_bsz}_epochs_${num_epochs}_kl_loss_coef_${kl_loss_coef}"  # Experiment name string
timestamp=$(date +"%Y%m%d_%H%M%S")  # Timestamp for unique checkpointing
resume_mode=auto  # Resume mode for training
resume_from_path=False  # Whether to resume from a specific checkpoint path
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/post_training/verl_checkpoints/${project_name}/${exp_name}_${timestamp}"}  # Checkpoint directory
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/${dataset_name}/train.parquet"}  # Training data file
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/${dataset_name}/test.parquet"}  # Test/validation data file

# =====================
# 4. Model and Validation Settings
# =====================
max_prompt_length=$((1024))  # Maximum prompt length
max_response_length=$((1024 * 3))  # Maximum response length
val_top_k=-1
val_temperature=0.6  # Temperature for validation generation
train_temperature=1.0  # Temperature for training generation

# =====================
# 5. Hardware and Miscellaneous
# =====================
use_dynamic_bsz=True  # Enable dynamic batch size
infer_micro_batch_size=null  # Micro batch size for inference (null = auto)
ppo_micro_batch_size=null  # Micro batch size for training (null = auto)
offload=False  # Enable parameter/optimizer offloading
n_gpus_per_node=4  # Number of GPUs per node

# =====================
# 6. Training Command
# =====================
echo "Qwen 1.5B GRPO Training with KL-loss-coefficient set to ${kl_loss_coef}. Setting up..."

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
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
    actor_rollout_ref.actor.use_kl_loss=True \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${ppo_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_token_level_loss=${use_token_level_loss} \
    actor_rollout_ref.actor.use_token_level_loss=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${train_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_k="${val_top_k}" \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0\
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes="${NNODES}" \
    +trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=25 \
    trainer.track_advantages=True \
    trainer.track_advantages_freq=5 \
    trainer.total_epochs=${num_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=${resume_mode} \
    trainer.resume_from_path=${resume_from_path} \
    trainer.remove_previous_ckpt_in_save=${remove_previous_ckpt_in_save}
