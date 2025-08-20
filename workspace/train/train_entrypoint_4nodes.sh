#!/bin/bash
#SBATCH --job-name=verl-train4                               # Descriptive job name
#SBATCH --output=/u/rfechner/jobs/slurm_train4_%j.out       # Standard output file (%j expands to job ID)
#SBATCH --error=/u/rfechner/jobs/slurm_train4_%j.err        # Standard error file (%j expands to job ID)
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1                                 # Run the shell script as a single task on this node
#SBATCH --cpus-per-task=72                                  # Allocate all CPUs on the node (Raven GPU nodes often have 72 cores, adjust if different)
#SBATCH --gres=gpu:a100:4                                   
#SBATCH --time=23:59:00                                     # Maximum wall clock time (e.g., 24 hours). Adjust as needed.
#SBATCH --mem=500000                                        # Request all available memory on the node. Can be specific e.g., "200G".
#SBATCH --chdir=./                                          # Set working directory to submission directory.

echo "========================================================"
echo "Starting VERL Training Job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated GPUs: $SLURM_JOB_GPUS"
echo "========================================================"

# Check for required environment variables
required_vars=("TRAIN_FILES" "MODEL_PATH" "ENTROPY_COEF" "KL_LOSS_COEF" "PROJECT_NAME" "DATA_SEED")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "ERROR: $var is not set. Please export it before running this script."
        exit 1
    fi
done

# Handle optional variables
if [ -z "${TIMESTAMP}" ]; then
    echo "ERROR: TIMESTAMP is not set. Please export it before running this script."
    exit 1
fi

if [ -z "${IDENTIFIER}" ]; then
    # IDENTIFIER is unset or empty
    IDENTIFIER=""
fi

# Set fixed values and defaults
LEARNING_RATE=1e-6
TOTAL_EPOCHS=20
TRAIN_BATCH_SIZE=1024  # Increased for 4 nodes (16 GPUs total)
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=2048  # Reduced from 3072 to save memory
SAVE_FREQ=5
TEST_FREQ=2
NNODES=${SLURM_JOB_NUM_NODES:-4}  # Use dynamic node detection from SLURM

RAY_DATA_HOME=${RAY_DATA_HOME:-"/u/rfechner"}  # Base directory for data and checkpoints
# Use timestamp passed from environment variable
# Build experiment name with optional identifier
if [ -n "${IDENTIFIER}" ]; then
    EXPNAME="${MODEL_PATH}_entropy_${ENTROPY_COEF}_kl_${KL_LOSS_COEF}_${TIMESTAMP}_${IDENTIFIER}"
else
    EXPNAME="${MODEL_PATH}_entropy_${ENTROPY_COEF}_kl_${KL_LOSS_COEF}_${TIMESTAMP}"
fi
CHECKPOINT_DIR="${RAY_DATA_HOME}/out/${PROJECT_NAME}/${EXPNAME}"  # Checkpoint directory
VAL_FILES=[data/math500/test.parquet]

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
echo "  Timestamp: $TIMESTAMP"
echo "  Data seed: $DATA_SEED"
if [ -n "${IDENTIFIER}" ]; then
    echo "  Identifier: $IDENTIFIER"
fi
echo "========================================================"

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load modules and activate your Conda env
# ─────────────────────────────────────────────────────────────────────────────
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

# Set PyTorch memory management for better fragmentation handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Network Settings for Multi-Node Training (NCCL configuration)
export NCCL_DEBUG=${NCCL_DEBUG:-"INFO"}
export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9"
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-"3"}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-"0"}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-"1"}

# Activate your environment
conda activate verl

# ─────────────────────────────────────────────────────────────────────────────
# 2) Set up Ray cluster (only if using multiple nodes)
# ─────────────────────────────────────────────────────────────────────────────
echo "Setting up Ray cluster for multi-node training..."

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

# ─────────────────────────────────────────────────────────────────────────────
# 3) Start Training
# ─────────────────────────────────────────────────────────────────────────────

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
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.model.path=$MODEL_PATH \
        +actor_rollout_ref.model.override_config.attention_dropout=0. \
        +actor_rollout_ref.model.override_config.embd_pdrop=0. \
        +actor_rollout_ref.model.override_config.resid_pdrop=0. \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        +actor_rollout_ref.ref.ppo_mini_batch_size=64 \
        +actor_rollout_ref.ref.ppo_micro_batch_size_per_gpu=2 \
        +actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
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
        trainer.resume_mode=auto \
        trainer.resume_from_path=False \
        trainer.remove_previous_ckpt_in_save=True \
        trainer.total_epochs=$TOTAL_EPOCHS \
        +trainer.early_stopping_enabled=True \
        +trainer.early_stopping_patience=20 \
        +trainer.early_stopping_min_delta=0.001 \
        +trainer.save_best_checkpoint=True \
        +trainer.DEV_ESTIMATE_ENTROPY_DELTA=True

echo "========================================================"
echo "Training completed with exit code: $?"
echo "End time: $(date)"
echo "Training job finished."
echo "========================================================"
