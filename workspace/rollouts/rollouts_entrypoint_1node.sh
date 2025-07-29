#!/bin/bash -l

#SBATCH --job-name=chunk
#SBATCH --output=/u/rfechner/jobs/rollouts_%j.out # Standard output file (%j expands to job ID)
#SBATCH --error=/u/rfechner/jobs/rollouts_%j.err  # Standard error file (%j expands to job ID)
#SBATCH --nodes=1                        # Request 1 node (as per NNODES=1 in your script)
#SBATCH --ntasks-per-node=1              # Run the shell script as a single task on this node
#SBATCH --cpus-per-task=72               # Allocate all CPUs on the node (Raven GPU nodes often have 72 cores, adjust if different)
#SBATCH --gres=gpu:a100:4                # Request 4 A100 GPUs
#SBATCH --time=23:00:00                  # Maximum wall clock time (e.g., 24 hours). Adjust as needed.
#SBATCH --mem=500000                     # Request all available memory on the node. Can be specific e.g., "200G".
#SBATCH --chdir=./                       # Set working directory to submission directory.

echo "========================================================"
echo "Starting Slurm Job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Submitted from: $(pwd)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated GPUs: $SLURM_JOB_GPUS (CUDA_VISIBLE_DEVICES likely set to $CUDA_VISIBLE_DEVICES)"
echo "========================================================"
echo "Loading environment modules..."
module purge
module load anaconda/3/2023.03
module load cuda/12.6 

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
module list
echo "========================================================"
echo "Start time: $(date)"
echo "========================================================"


# Check for required environment variables
if [ -z "${DATASET_NAME}" ]; then
    echo "ERROR: DATASET_NAME is not set. Please export it before running this script."
    exit 1

fi

if [ -z "${DATASET_PATH}" ]; then
    echo "ERROR: DATASET_PATH is not set. Please export it before running this script."
    exit 1
fi

if [ -z "${CHUNK_SIZE}" ]; then
    echo "ERROR: CHUNK_SIZE is not set. Please export it before running this script."
    exit 1
fi

if [ -z "${ROLLOUTS}" ]; then
    echo "ERROR: ROLLOUTS is not set. Please export it before running this script."
    exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH is not set. Please export it before running this script."
    exit 1
fi

if [ -z "${CHECKPOINT}" ]; then
    # CHECKPOINT is unset or empty
    CHECKPOINT=null
fi

if [ -z "${TIMESTAMP}" ]; then
    echo "ERROR: TIMESTAMP is not set. Please export it before running this script."
    exit 1
fi

if [ -z "${IDENTIFIER}" ]; then
    # IDENTIFIER is unset or empty
    IDENTIFIER=""
fi

# =====================
# 1. Project, Experiment, and Paths
# =====================

# Set PyTorch memory management for better fragmentation handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

project_name='rollouts'  # Name of the project
RAY_DATA_HOME=${RAY_DATA_HOME:-"/u/rfechner"}  # Base directory for data and checkpoints

# Use timestamp passed from environment variable
# Build experiment name with optional identifier
if [ -n "${IDENTIFIER}" ]; then
    EXPERIMENT_NAME="${MODEL_PATH}_${DATASET_NAME}_nrollouts_${ROLLOUTS}_${TIMESTAMP}_${IDENTIFIER}/chunk${SLURM_IDX}"
else
    EXPERIMENT_NAME="${MODEL_PATH}_${DATASET_NAME}_nrollouts_${ROLLOUTS}_${TIMESTAMP}/chunk${SLURM_IDX}"
fi
CHECKPOINTS_DIR="${RAY_DATA_HOME}/out/${project_name}/${EXPERIMENT_NAME}"  # Checkpoint directory
FILE="${RAY_DATA_HOME}/data/${DATASET_PATH}"

# =====================
# 2. Model and Generation Settings
# =====================
max_prompt_length=1024  # Maximum prompt length
max_response_length=2048  # Reduced from 3072 to save memory (like in train script)
val_top_k=-1  # Top-k for validation generation
val_temperature=0.6  # Temperature for validation generation


python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    data.path="${FILE}" \
    data.output_path="${CHECKPOINTS_DIR}/results.parquet" \
    data.n_samples=${ROLLOUTS} \
    data.batch_size=12 \
    +data.compute_scores=True \
    +data.CHUNK_SIZE=${CHUNK_SIZE} \
    +data.SLURM_IDX=${SLURM_IDX} \
    +data.dump_parts=True \
    model.path=${MODEL_PATH} \
    +model.checkpoint=${CHECKPOINT} \
    rollout.response_length=${max_response_length} \
    rollout.temperature=${val_temperature} \
    rollout.top_k=${val_top_k} \
    rollout.prompt_length=${max_prompt_length} \
    rollout.response_length=${max_response_length} \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.65 \
    rollout.enable_chunked_prefill=True \
    rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \

echo "========================================================"
echo "Script execution finished with exit code: $?"
echo "End time: $(date)"
echo "Slurm job finished."
echo "========================================================"
