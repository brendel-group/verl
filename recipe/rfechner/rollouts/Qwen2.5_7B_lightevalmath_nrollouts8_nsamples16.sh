#!/usr/bin/env bash
set -euxo pipefail

# ------------------------------------------------------------------------------
# Script: generate_rfechner_test.sh
# Purpose: Run LLM rollout generation for RL exploration using verl/trainer.
#   - Configures experiment, dataset, model, and hardware settings
#   - Launches generation with specified parameters and saves results
# ------------------------------------------------------------------------------

# =====================
# 1. Project, Experiment, and Paths
# =====================
project_name='rollouts'  # Name of the project
RAY_DATA_HOME=${RAY_DATA_HOME:-"/u/rfechner"}  # Base directory for data and checkpoints
MODEL_PATH=Qwen/Qwen2.5-7B  # Model name or path
dataset_name='math'  # Name of the dataset
num_rollouts=8  # Number of rollouts per prompt
num_questions=16  # Number of questions to take from the dataset
exp_name="${MODEL_PATH}_${dataset_name}_nquestions_${num_questions}_nrollouts_${num_rollouts}"  # Experiment name string
timestamp=$(date +"%Y%m%d_%H%M%S")  # Timestamp for unique checkpointing
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/out/${project_name}/${exp_name}_${timestamp}"}  # Checkpoint directory
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/${dataset_name}/train.parquet"}  # Training data file
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/${dataset_name}/test.parquet"}  # Test/validation data file

# =====================
# 2. Training and Hardware Parameters
# =====================
NNODES=1  # Number of nodes for distributed training
n_gpus_per_node=4  # Number of GPUs per node

# =====================
# 3. Model and Generation Settings
# =====================
max_prompt_length=1024  # Maximum prompt length
max_response_length=$((1024 * 3))  # Maximum response length
val_top_k=-1  # Top-k for validation generation
val_temperature=0.6  # Temperature for validation generation

# =====================
# 4. Launch Generation
# =====================
python3 -m verl.trainer.main_generation \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    +data.compute_scores=True \
    data.path="${TEST_FILE}" \
    data.output_path="${CKPTS_DIR}/results.parquet" \
    data.n_samples=${num_rollouts} \
    +data.take_first_n=${num_questions} \
    data.batch_size=8 \
    model.path="${MODEL_PATH}" \
    rollout.response_length=${max_response_length} \
    rollout.temperature=${val_temperature} \
    rollout.top_k=${val_top_k} \
    rollout.prompt_length=${max_prompt_length} \
    rollout.response_length=${max_response_length} \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8 \