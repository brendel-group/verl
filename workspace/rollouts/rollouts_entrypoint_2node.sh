#!/bin/bash -l

#SBATCH --job-name=chunk-2node
#SBATCH --output=/u/rfechner/jobs/rollouts_2node_%j.out # Standard output file (%j expands to job ID)
#SBATCH --error=/u/rfechner/jobs/rollouts_2node_%j.err  # Standard error file (%j expands to job ID)
#SBATCH --nodes=2                        # Request 2 nodes for distributed rollouts
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1              # Run the shell script as a single task on this node
#SBATCH --cpus-per-task=72               # Allocate all CPUs on the node (Raven GPU nodes often have 72 cores, adjust if different)
#SBATCH --gres=gpu:a100:4                # Request 4 A100 GPUs per node (8 total)
#SBATCH --time=23:00:00                  # Maximum wall clock time (e.g., 24 hours). Adjust as needed.
#SBATCH --mem=500000                     # Request all available memory on the node. Can be specific e.g., "200G".
#SBATCH --chdir=./                       # Set working directory to submission directory.

echo "========================================================"
echo "Starting 2-Node Slurm Job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
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
# 2. Ray Cluster Setup (2 nodes)
# =====================

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
else
    head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
export RAY_ADDRESS="$ip_head"

echo "IP Head: $ip_head"
echo "RAY_ADDRESS: $RAY_ADDRESS"

# Set number of GPUs using SLURM environment variable
num_gpus=${SLURM_GPUS_PER_NODE:-4}  # Use SLURM detection, fallback to 4

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --dashboard-host 0.0.0.0 --dashboard-port=8265 \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "$num_gpus" --block &

# Give head node more time to initialize
sleep 30

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "$num_gpus" --block &
    sleep 5
done

# Give cluster time to fully connect
sleep 15
echo "Ray cluster should be ready, testing connection..."

# =====================
# 3. Model and Generation Settings
# =====================
max_prompt_length=1024  # Maximum prompt length
max_response_length=2048  # Reduced from 3072 to save memory (like in train script)
val_top_k=-1  # Top-k for validation generation
val_temperature=0.6  # Temperature for validation generation

# =====================
# 4. Run Generation on 2-node Ray cluster
# =====================

python3 -m verl.trainer.main_generation \
    trainer.nnodes=2 \
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
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.65 \
    rollout.enable_chunked_prefill=True \
    rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \

echo "========================================================"
echo "Script execution finished with exit code: $?"
echo "End time: $(date)"
echo "Slurm job finished."
echo "========================================================"
