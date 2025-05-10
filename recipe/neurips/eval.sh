#!/usr/bin/env bash
# Usage:
#   bash recipe/neurips/eval.sh /path/to/experiment_folder [start_step]
#
# This script evaluates Hugging Face model checkpoints found within the specified
# /path/to/experiment_folder. It looks for subdirectories named "global_step_X".
#
# Arguments:
#   /path/to/experiment_folder: (Required) The main directory containing
#                               global_step_* checkpoint subdirectories.
#                               If a path to a specific global_step_X directory
#                               is given, only that checkpoint is evaluated.
#   start_step:                 (Optional) An integer. If provided, only checkpoints
#                               with a step number (X) greater than or equal to
#                               start_step will be evaluated.
#
# The script iterates through each qualifying checkpoint, loads it as a base model
# (not resuming training state), and runs validation using verl.trainer.main_ppo.
# Evaluation results (metrics) are appended to an 'eval.jsonl' file located
# in /path/to/experiment_folder. Each entry in eval.jsonl will be tagged
# with the corresponding step number.
set -euxo pipefail

project_name='self_distillation_neurips'

adv_estimator=grpo

kl_coef=0.0
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

enable_overlong_buffer=False
overlong_buffer_len=512
overlong_penalty_factor=1.0

enable_filter_groups=True
filter_groups_metric=seq_final_reward
fill_to_train_bsz=True
train_prompt_bsz=512 # 512 works for 7B n = 8
multiplier=3 # 3 works for 7B n = 8
gen_prompt_bsz=$((train_prompt_bsz * multiplier))
train_prompt_mini_bsz=128
train_micro_batch_size=64
val_batch_size=512

num_epochs=10

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
# INPUT_PATH is the first argument to the script.
# It can be /path/to/exp_dir or /path/to/exp_dir/global_step_X
INPUT_PATH=$1
# START_STEP_ARG is the optional second argument.
START_STEP_ARG=${2:-""} # Default to empty string if not provided

# Determine the main experiment directory and specific checkpoint paths to evaluate
declare -a CHECKPOINT_FULL_PATHS_TO_EVAL
declare -a TEMP_CHECKPOINT_PATHS_TO_EVAL # Used for initial find before filtering
MAIN_EXPERIMENT_DIR=""

if [[ ! -e "$INPUT_PATH" ]]; then
    echo "Error: Input path $INPUT_PATH does not exist."
    exit 1
fi

if [[ "$INPUT_PATH" == *"global_step_"* ]] && [ -d "$INPUT_PATH" ]; then
    # Input is a specific checkpoint directory
    TEMP_CHECKPOINT_PATHS_TO_EVAL=("$INPUT_PATH")
    MAIN_EXPERIMENT_DIR=$(dirname "$INPUT_PATH")
elif [ -d "$INPUT_PATH" ]; then
    # Input is a parent directory, find all global_step_* subdirectories
    MAIN_EXPERIMENT_DIR="$INPUT_PATH"
    # Use find and map to an array, sorting by step number
    mapfile -t TEMP_CHECKPOINT_PATHS_TO_EVAL < <(find "$MAIN_EXPERIMENT_DIR" -maxdepth 1 -type d -name "global_step_*" | sort -V)
    if [ ${#TEMP_CHECKPOINT_PATHS_TO_EVAL[@]} -eq 0 ]; then
        echo "No global_step_* directories found in $MAIN_EXPERIMENT_DIR."
        # Check if MAIN_EXPERIMENT_DIR itself might be a checkpoint (e.g. downloaded from elsewhere)
        if [[ "$MAIN_EXPERIMENT_DIR" == *"global_step_"* ]]; then
             echo "Treating $MAIN_EXPERIMENT_DIR as a single checkpoint directory."
             TEMP_CHECKPOINT_PATHS_TO_EVAL=("$MAIN_EXPERIMENT_DIR")
        else
            echo "Exiting."
            exit 1
        fi
    fi
else
    echo "Error: $INPUT_PATH is not a valid directory."
    exit 1
fi

# Filter checkpoints if START_STEP_ARG is provided
if [[ -n "$START_STEP_ARG" ]]; then
    echo "Filtering checkpoints to start from step: $START_STEP_ARG"
    for ckpt_path_candidate in "${TEMP_CHECKPOINT_PATHS_TO_EVAL[@]}"; do
        # Extract step number from path (e.g., global_step_100 -> 100)
        step_num_from_path=$(basename "$ckpt_path_candidate" | sed 's/global_step_//')
        # Check if step_num_from_path is a valid number before comparison
        if [[ "$step_num_from_path" =~ ^[0-9]+$ ]] && [ "$step_num_from_path" -ge "$START_STEP_ARG" ]; then
            CHECKPOINT_FULL_PATHS_TO_EVAL+=("$ckpt_path_candidate")
        fi
    done
    if [ ${#CHECKPOINT_FULL_PATHS_TO_EVAL[@]} -eq 0 ]; then
        echo "No checkpoints found at or after step $START_STEP_ARG in $MAIN_EXPERIMENT_DIR (from initial list of ${#TEMP_CHECKPOINT_PATHS_TO_EVAL[@]} checkpoints)."
        exit 0 # Exit gracefully if no checkpoints match the criteria
    fi
else
    echo "No start step specified, evaluating all found checkpoints."
    CHECKPOINT_FULL_PATHS_TO_EVAL=("${TEMP_CHECKPOINT_PATHS_TO_EVAL[@]}")
fi

# Common settings for python script invocation
# trainer.default_local_dir will be MAIN_EXPERIMENT_DIR for storing eval.jsonl
DEFAULT_LOCAL_DIR_FOR_PYTHON="${MAIN_EXPERIMENT_DIR}"
# trainer.experiment_name can be the basename of the main experiment dir
EXPERIMENT_NAME_FOR_PYTHON=$(basename "${MAIN_EXPERIMENT_DIR}")
# trainer.resume_from_path should be False to use resume_mode path
RESUME_FROM_PATH_FOR_PYTHON=False
# Ensure validation only mode
VAL_ONLY_FOR_PYTHON=True
VAL_BEFORE_TRAIN_FOR_PYTHON=True # Ensures validation runs

dataset_name=${dataset_name:-'openai_math'} # Ensure dataset_name is set
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/datasets/${dataset_name}/train.parquet"} # dataset_name is defined below, or use a default

# Loop through each checkpoint directory and run evaluation
for CHECKPOINT_PATH_FOR_PYTHON in "${CHECKPOINT_FULL_PATHS_TO_EVAL[@]}"; do
    echo "----------------------------------------------------"
    echo "Evaluating checkpoint: $CHECKPOINT_PATH_FOR_PYTHON"
    echo "Output eval.jsonl will be in: ${DEFAULT_LOCAL_DIR_FOR_PYTHON}/eval.jsonl"
    echo "Experiment name for logging: ${EXPERIMENT_NAME_FOR_PYTHON}"
    echo "----------------------------------------------------"

    # Extract step number from checkpoint path
    STEP_NUM_FOR_LOGGING=""
    if [[ "$CHECKPOINT_PATH_FOR_PYTHON" == *"global_step_"* ]]; then
        STEP_NUM_FOR_LOGGING=$(basename "$CHECKPOINT_PATH_FOR_PYTHON" | sed 's/global_step_//')
    else
        # Fallback or error if step number can't be determined, though find should ensure format
        echo "Warning: Could not determine step number from $CHECKPOINT_PATH_FOR_PYTHON. Using 0."
        STEP_NUM_FOR_LOGGING="0"
    fi
    echo "Step number for logging: ${STEP_NUM_FOR_LOGGING}"

    # MODEL_PATH_FOR_PYTHON for actor_rollout_ref.model.path is the specific checkpoint
    # RESUME_MODE_FOR_PYTHON will be "disable" to load as a base model
    # EVALUATION_STEP_FOR_PYTHON will pass the extracted step number for logging

    # The MODEL_PATH variable in the original script is now CHECKPOINT_PATH_FOR_PYTHON for actor path
    # The exp_name variable is EXPERIMENT_NAME_FOR_PYTHON
    # The CKPTS_DIR variable is DEFAULT_LOCAL_DIR_FOR_PYTHON
    # resume_mode is now 'disable'
    # resume_from_path variable is RESUME_FROM_PATH_FOR_PYTHON (still False)

    # Algorithm
    ## Train
    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 3))
    ## Validation
    val_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz=True
    infer_micro_batch_size=null
    train_micro_batch_size=null
    offload=False
    #    data.val_files=[/fast/pmayilvahanan/datasets/openai_math/test.parquet,/fast/pmayilvahanan/datasets/aime_2024/test.parquet,/fast/pmayilvahanan/datasets/olympiad_bench/test.parquet,/fast/pmayilvahanan/datasets/gpqa/test.parquet,/fast/pmayilvahanan/datasets/minervamath/test.parquet,/fast/pmayilvahanan/datasets/amc23/test.parquet,/fast/pmayilvahanan/datasets/aime_2025/test.parquet] \

    # ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    #     --working-dir "${WORKING_DIR}" \
    python3 -m verl.trainer.main_ppo \
        data.train_files="${TRAIN_FILE}" \
        data.val_files=[/fast/pmayilvahanan/datasets/openai_math/test.parquet,/fast/pmayilvahanan/datasets/aime_2024/test.parquet,/fast/pmayilvahanan/datasets/olympiad_bench/test.parquet,/fast/pmayilvahanan/datasets/gpqa/test.parquet,/fast/pmayilvahanan/datasets/minervamath/test.parquet,/fast/pmayilvahanan/datasets/amc23/test.parquet,/fast/pmayilvahanan/datasets/aime_2025/test.parquet] \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.gen_batch_size=${gen_prompt_bsz} \
        data.train_batch_size=${train_prompt_bsz} \
        data.val_batch_size=${val_batch_size} \
        data.truncation='left' \
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
        actor_rollout_ref.actor.use_token_level_loss=True \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.val_kwargs.top_k="${val_top_k}" \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0\
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
        actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        trainer.logger=['console'] \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${EXPERIMENT_NAME_FOR_PYTHON}" \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes="${NNODES}" \
        +trainer.val_before_train=${VAL_BEFORE_TRAIN_FOR_PYTHON} \
        trainer.test_freq=1 \
        trainer.save_freq=5 \
        trainer.track_advantages=False \
        trainer.track_advantages_freq=5 \
        trainer.total_epochs=${num_epochs} \
        trainer.default_local_dir="${DEFAULT_LOCAL_DIR_FOR_PYTHON}" \
        trainer.resume_mode="disable" \
        trainer.resume_from_path=${RESUME_FROM_PATH_FOR_PYTHON}  \
        +trainer.val_only=${VAL_ONLY_FOR_PYTHON} \
        +trainer.evaluation_step=${STEP_NUM_FOR_LOGGING} \
        +trainer.current_val_files_config="[/fast/pmayilvahanan/datasets/openai_math/test.parquet,/fast/pmayilvahanan/datasets/aime_2024/test.parquet,/fast/pmayilvahanan/datasets/olympiad_bench/test.parquet,/fast/pmayilvahanan/datasets/gpqa/test.parquet,/fast/pmayilvahanan/datasets/minervamath/test.parquet,/fast/pmayilvahanan/datasets/amc23/test.parquet,/fast/pmayilvahanan/datasets/aime_2025/test.parquet]"
done # End of loop for CHECKPOINT_PATH_FOR_PYTHON

echo "All specified evaluations complete."