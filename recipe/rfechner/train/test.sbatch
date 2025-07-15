#!/bin/bash
#SBATCH --job-name=slurm-ray-test                           # Descriptive job name
#SBATCH --output=/u/rfechner/jobs/slurm_verl_train_%j.out   # Standard output file (%j expands to job ID)
#SBATCH --error=/u/rfechner/jobs/slurm_verl_train_%j.err    # Standard error file (%j expands to job ID)
#SBATCH --nodes=1                                           # Request 1 node (as per NNODES=1 in your script)
#SBATCH --ntasks-per-node=1                                 # Run the shell script as a single task on this node
#SBATCH --cpus-per-task=72                                  # Allocate all CPUs on the node (Raven GPU nodes often have 72 cores, adjust if different)
#SBATCH --gres=gpu:a100:4                                   # Request 8 A100 GPUs (based on n_gpus_per_node=8 in your script)
                                                            # IMPORTANT: Verify Raven node configuration for 8 GPUs/node. Examples often show 4.
#SBATCH --time=23:59:00                                     # Maximum wall clock time (e.g., 24 hours). Adjust as needed.
#SBATCH --mem=500000                                        # Request all available memory on the node. Can be specific e.g., "200G".
#SBATCH --chdir=./                                          # Set working directory to submission directory.

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

# Activate your environment
conda activate verl

echo "Activated conda environment: verl"
echo "Modules currently loaded:"
module list

# ─────────────────────────────────────────────────────────────────────────────
# 2) Paths & settings
# ─────────────────────────────────────────────────────────────────────────────
train_files=/u/rfechner/data/math/train.parquet
val_files=/u/rfechner/data/math/test.parquet

# Ray port & head node config
port=6379

# ─────────────────────────────────────────────────────────────────────────────
# 3) Discover head‐node IP address
# ─────────────────────────────────────────────────────────────────────────────
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes[0]}

# Ask srun for the head's IP; pick the IPv4 if both are returned
raw_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
if [[ "$raw_ip" =~ " " ]]; then
  # split on space, pick the shorter (IPv4)
  for ip in $raw_ip; do
    if [[ ${#ip} -le 15 ]]; then ipv4=$ip; fi
  done
  head_node_ip=$ipv4
else
  head_node_ip=$raw_ip
fi
echo "Head node IP: $head_node_ip"

export ip_head="$head_node_ip:$port"

# ─────────────────────────────────────────────────────────────────────────────
# 4) Start Ray HEAD
# ─────────────────────────────────────────────────────────────────────────────
echo "Starting Ray head on ${head_node}..."
srun --nodes=1 --ntasks=1 -w "$head_node" \
     ray start --head \
               --node-ip-address="$head_node_ip" \
               --port="$port" \
               --num-cpus="${SLURM_CPUS_PER_TASK}" \
               --num-gpus="${SLURM_GPUS_PER_NODE}" \
               --block &

sleep 10   # give head time to spin up

# ─────────────────────────────────────────────────────────────────────────────
# 5) Start Ray WORKERS
# ─────────────────────────────────────────────────────────────────────────────
worker_count=$(( SLURM_JOB_NUM_NODES - 1 ))
for (( i=1; i<=worker_count; i++ )); do
  node=${nodes[i]}
  echo "Starting Ray worker #$i on $node..."
  srun --nodes=1 --ntasks=1 -w "$node" \
       ray start --address="$ip_head" \
                 --num-cpus="${SLURM_CPUS_PER_TASK}" \
                 --num-gpus="${SLURM_GPUS_PER_NODE}" \
                 --block &
  sleep 5
done

# ─────────────────────────────────────────────────────────────────────────────
# 6) Launch your training script (via Conda env)
# ─────────────────────────────────────────────────────────────────────────────
echo "Launching training on head node..."
srun --nodes=1 --ntasks=1 -w "$head_node" \
     python -m verl.trainer.main_ppo \
       algorithm.adv_estimator=gae \
       data.train_files="$train_files" \
       data.val_files="$val_files" \
       data.train_batch_size=256 \
       data.max_prompt_length=512 \
       data.max_response_length=256 \
       actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B \
       actor_rollout_ref.actor.optim.lr=1e-6 \
       actor_rollout_ref.actor.ppo_mini_batch_size=64 \
       actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
       actor_rollout_ref.actor.use_kl_loss=False \
       actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
       actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
       actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
       critic.optim.lr=1e-5 \
       critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
       critic.ppo_micro_batch_size_per_gpu=4 \
       algorithm.use_kl_in_reward=False \
       trainer.logger=console \
       trainer.val_before_train=False \
       trainer.n_gpus_per_node="${SLURM_GPUS_PER_NODE}" \
       trainer.nnodes="${SLURM_NNODES}" \
       trainer.save_freq=10 \
       trainer.test_freq=10 \
       trainer.total_epochs=2 2>&1 | tee verl_demo_slurm.log