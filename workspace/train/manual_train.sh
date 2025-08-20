#!/bin/bash
# Manual training script used to run on interactive nodes.
# Set fixed values and defaults
LEARNING_RATE=1e-6
TOTAL_EPOCHS=2
TRAIN_BATCH_SIZE=128
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=512
SAVE_FREQ=5
TEST_FREQ=2
NNODES=1
RAY_DATA_HOME=${RAY_DATA_HOME:-"/u/rfechner"}  # Base directory for data and checkpoints
ENTROPY_COEF=0.0
KL_LOSS_COEF=0.0
MODEL_PATH="Qwen/Qwen2.5-1.5B"
PROJECT_NAME="entropy_logging"
DATA_SEED=42

EXPNAME="${MODEL_PATH}_entropy_${ENTROPY_COEF}_kl_${KL_LOSS_COEF}"
CHECKPOINT_DIR="${RAY_DATA_HOME}/out/${PROJECT_NAME}/${EXPNAME}"  # Checkpoint directory
TRAIN_FILES="data/math/train.parquet"
VAL_FILES="data/math500/test.parquet"

echo "Configuration:"
echo "  Train files: $TRAIN_FILES"
echo "  Val files: $VAL_FILES"
echo "  Model: $MODEL_PATH"
echo "  Learning rate: $LEARNING_RATE (fixed)"
echo "  Entropy coefficient: $ENTROPY_COEF"
echo "  KL loss coefficient: $KL_LOSS_COEF"
echo "  Total epochs: $TOTAL_EPOCHS (fixed)"
echo "  Project name: $PROJECT_NAME"
echo "  Nodes: $NNODES (fixed)"
echo "========================================================"

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load modules and activate your Conda env
# ─────────────────────────────────────────────────────────────────────────────
echo "Loading environment modules..."
module purge
module load anaconda/3/2023.03
module load cuda/12.6 

# Set PyTorch memory management for better fragmentation handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# VLLM-specific environment variables for stability
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Network Settings for Multi-Node Training (NCCL configuration)
export NCCL_DEBUG=${NCCL_DEBUG:-"INFO"}
export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9"
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-"3"}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-"0"}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-"1"}

# >>> conda initialize >>>
__conda_setup="$('/mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh" ]; then
        . "/mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh"
    else
        export PATH="/mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Activate your environment
conda activate verl

# ─────────────────────────────────────────────────────────────────────────────
# 2) Start Training
# ─────────────────────────────────────────────────────────────────────────────
echo "Starting training..."

python -u -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="$TRAIN_FILES" \
        data.val_files=$VAL_FILES \
        data.train_batch_size=$TRAIN_BATCH_SIZE \
        data.max_prompt_length=$MAX_PROMPT_LENGTH \
        data.max_response_length=$MAX_RESPONSE_LENGTH \
        data.truncation=left \
        +data.seed=$DATA_SEED \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
        +actor_rollout_ref.actor.DEV_ESTIMATE_ENTROPY_DELTA=True \
        +actor_rollout_ref.actor.log_prob_max_token_len_per_gpu=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((1 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
        actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEF \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((6 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
        +algorithm.use_kl_in_reward=False \
        trainer.default_local_dir="${CHECKPOINT_DIR}" \
        trainer.project_name="$PROJECT_NAME" \
        trainer.logger=console \
        +trainer.val_before_train=False \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=$NNODES \
        trainer.save_freq=$SAVE_FREQ \
        trainer.test_freq=$TEST_FREQ \
        trainer.default_local_dir="${CHECKPOINT_DIR}" \
        trainer.remove_previous_ckpt_in_save=False \
        trainer.total_epochs=$TOTAL_EPOCHS \
        +trainer.early_stopping_enabled=True \
        +trainer.early_stopping_patience=20 \
        +trainer.early_stopping_min_delta=0.001 \
        +trainer.save_best_checkpoint=True