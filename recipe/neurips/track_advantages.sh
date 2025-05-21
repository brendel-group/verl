#!/usr/bin/env bash
# Usage:
#   bash recipe/neurips/track_advantages.sh /path/to/experiment_folder [start_step]
#
# This script performs advantage tracking for Hugging Face model checkpoints
# found within the specified /path/to/experiment_folder.
# It looks for subdirectories named "global_step_X".
#
# Arguments:
#   /path/to/experiment_folder: (Required) The main directory containing
#                               global_step_* checkpoint subdirectories.
#                               If a path to a specific global_step_X directory
#                               is given, only that checkpoint is processed.
#   start_step:                 (Optional) An integer. If provided, only checkpoints
#                               with a step number (X) greater than or equal to
#                               start_step will be processed.
#
# Advantage tracking results will be saved in an 'advantage_tracking' subdirectory
# within /path/to/experiment_folder (e.g., .../advantage_tracking/advantages_train_step_X.pt).
set -euxo pipefail

project_name='self_distillation_neurips' # Or make this configurable

# Default values, similar to eval.sh, adjust if necessary for advantage tracking
adv_estimator=grpo
kl_coef=0.0
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
enable_overlong_buffer=False
overlong_buffer_len=512
overlong_penalty_factor=1.0
enable_filter_groups=True # Usually enabled for training, might be relevant for how model was trained
filter_groups_metric=seq_final_reward
fill_to_train_bsz=True
train_prompt_bsz=64
multiplier=2
gen_prompt_bsz=$((train_prompt_bsz * multiplier))
train_prompt_mini_bsz=128
# train_micro_batch_size needs to be determined based on GPU memory for generation/logprob, not training updates
# For advantage tracking, actual training updates are skipped.
# However, logprob computation still happens.
# Let's keep train_micro_batch_size, but it might be less critical than for actual training.
train_micro_batch_size=64
val_batch_size=512 # Used for creating val dataloader for advantage tracking

num_epochs=1 # Not used for advantage_tracking_only, but script expects it

num_rollouts=8 # This is actor_rollout_ref.rollout.n
val_kwargs_n=${num_rollouts} # n for validation generation part of advantage tracking
n_resp_per_prompt=${num_rollouts}
# n_advantage_tracking should be set in config if different from num_rollouts
# For the script, ensure actor_rollout_ref.rollout.n_advantage_tracking can be overridden if needed
n_advantage_tracking=${num_rollouts} # Defaulting to num_rollouts, can be overridden by config

use_token_level_loss=False # DAPO specific, usually False for PPO reward shaping

# Ray (mostly not critical for advantage_tracking_only on a single node setup, but kept for consistency)
NNODES=1
n_gpus_per_node=4 # Adjust based on your setup for generation/logprob compute

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"/fast/pmayilvahanan/"}
INPUT_PATH=$1
START_STEP_ARG=${2:-""}

declare -a CHECKPOINT_FULL_PATHS_TO_PROCESS
declare -a TEMP_CHECKPOINT_PATHS_TO_PROCESS
MAIN_EXPERIMENT_DIR=""

if [[ ! -e "$INPUT_PATH" ]]; then
    echo "Error: Input path $INPUT_PATH does not exist."
    exit 1
fi

if [[ "$INPUT_PATH" == *"global_step_"* ]] && [ -d "$INPUT_PATH" ]; then
    TEMP_CHECKPOINT_PATHS_TO_PROCESS=("$INPUT_PATH")
    MAIN_EXPERIMENT_DIR=$(dirname "$INPUT_PATH")
elif [ -d "$INPUT_PATH" ]; then
    MAIN_EXPERIMENT_DIR="$INPUT_PATH"
    mapfile -t TEMP_CHECKPOINT_PATHS_TO_PROCESS < <(find "$MAIN_EXPERIMENT_DIR" -maxdepth 1 -type d -name "global_step_*" | sort -V)
    if [ ${#TEMP_CHECKPOINT_PATHS_TO_PROCESS[@]} -eq 0 ]; then
        echo "No global_step_* directories found in $MAIN_EXPERIMENT_DIR."
        if [[ "$MAIN_EXPERIMENT_DIR" == *"global_step_"* ]]; then
             echo "Treating $MAIN_EXPERIMENT_DIR as a single checkpoint directory."
             TEMP_CHECKPOINT_PATHS_TO_PROCESS=("$MAIN_EXPERIMENT_DIR")
        else
            echo "Exiting."
            exit 1
        fi
    fi
else
    echo "Error: $INPUT_PATH is not a valid directory."
    exit 1
fi

if [[ -n "$START_STEP_ARG" ]]; then
    echo "Filtering checkpoints to start from step: $START_STEP_ARG"
    for ckpt_path_candidate in "${TEMP_CHECKPOINT_PATHS_TO_PROCESS[@]}"; do
        step_num_from_path=$(basename "$ckpt_path_candidate" | sed 's/global_step_//')
        if [[ "$step_num_from_path" =~ ^[0-9]+$ ]] && [ "$step_num_from_path" -ge "$START_STEP_ARG" ]; then
            CHECKPOINT_FULL_PATHS_TO_PROCESS+=("$ckpt_path_candidate")
        fi
    done
    if [ ${#CHECKPOINT_FULL_PATHS_TO_PROCESS[@]} -eq 0 ]; then
        echo "No checkpoints found at or after step $START_STEP_ARG in $MAIN_EXPERIMENT_DIR (from initial list of ${#TEMP_CHECKPOINT_PATHS_TO_PROCESS[@]} checkpoints)."
        exit 0
    fi
else
    echo "No start step specified, processing all found checkpoints."
    CHECKPOINT_FULL_PATHS_TO_PROCESS=("${TEMP_CHECKPOINT_PATHS_TO_PROCESS[@]}")
fi

DEFAULT_LOCAL_DIR_FOR_PYTHON="${MAIN_EXPERIMENT_DIR}"
EXPERIMENT_NAME_FOR_PYTHON=$(basename "${MAIN_EXPERIMENT_DIR}")
#dataset_name=${dataset_name:-'openai_math'}
#TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/datasets/${dataset_name}/train.parquet"}
dataset_name='dsr_sub'
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/datasets/rl_training_one_example/dsr_sub.parquet"}

# For advantage tracking, we might want to use the same val files as eval for consistency
VAL_FILES_STR="[/fast/pmayilvahanan/datasets/openai_math/test.parquet]" # Define your val files for advantage tracking

# Loop through each checkpoint directory and run advantage tracking
for CHECKPOINT_PATH_FOR_PYTHON in "${CHECKPOINT_FULL_PATHS_TO_PROCESS[@]}"; do
    echo "----------------------------------------------------"
    echo "Processing checkpoint for advantage tracking: $CHECKPOINT_PATH_FOR_PYTHON"
    echo "Output advantage files will be in: ${DEFAULT_LOCAL_DIR_FOR_PYTHON}/advantage_tracking/"
    echo "Experiment name for logging context: ${EXPERIMENT_NAME_FOR_PYTHON}"
    echo "----------------------------------------------------"

    STEP_NUM_FOR_LOGGING=""
    if [[ "$CHECKPOINT_PATH_FOR_PYTHON" == *"global_step_"* ]]; then
        STEP_NUM_FOR_LOGGING=$(basename "$CHECKPOINT_PATH_FOR_PYTHON" | sed 's/global_step_//')
    else
        echo "Warning: Could not determine step number from $CHECKPOINT_PATH_FOR_PYTHON. Using 0."
        STEP_NUM_FOR_LOGGING="0"
    fi
    echo "Step number for advantage file naming: ${STEP_NUM_FOR_LOGGING}"

    max_prompt_length=$((1024 * 1))
    max_response_length=$((1024 * 3))
    val_top_k=-1 # Affects generation if _compute_and_save_dataset_advantages performs generation

    use_dynamic_bsz=True # Recommended
    infer_micro_batch_size=null # Set to null if use_dynamic_bsz=True
    # train_micro_batch_size is already set above, but might not be used if dynamic_bsz=True
    offload=False # Usually for large models

    # Note: Many parameters below are for full PPO. Advantage tracking mode should bypass most of them.
    # However, data loading, model loading, and generation/logprob computation parameters are relevant.
    python3 -m verl.trainer.main_ppo \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${VAL_FILES_STR}" \
        data.prompt_key=prompt \
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
        actor_rollout_ref.model.path="${CHECKPOINT_PATH_FOR_PYTHON}" \
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
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.use_token_level_loss=${use_token_level_loss} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.val_kwargs.top_k="${val_top_k}" \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
        actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        trainer.logger=['console'] \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${EXPERIMENT_NAME_FOR_PYTHON}" \
        trainer.n_gpus_per_node=${n_gpus_per_node} \
        trainer.nnodes="${NNODES}" \
        trainer.total_epochs=${num_epochs} \
        trainer.default_local_dir="${DEFAULT_LOCAL_DIR_FOR_PYTHON}" \
        trainer.resume_mode="disable" \
        trainer.resume_from_path=False  \
        +trainer.advantage_tracking_only=True \
        trainer.track_advantages=True \
        +trainer.evaluation_step=${STEP_NUM_FOR_LOGGING} \
        +trainer.val_only=False \
        +trainer.val_before_train=False

done # End of loop for CHECKPOINT_PATH_FOR_PYTHON

echo "All specified advantage tracking tasks complete." 