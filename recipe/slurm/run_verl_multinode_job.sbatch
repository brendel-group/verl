#!/bin/bash -l

#SBATCH --job-name=verl_multinode_train_job # Descriptive job name
#SBATCH --output=/u/prama/jobs/slurm_verl_multinode_train_%j.out # Standard output file
#SBATCH --error=/u/prama/jobs/slurm_verl_multinode_train_%j.err  # Standard error file
#SBATCH --nodes=2                        # Number of nodes to request (adjust as needed)
#SBATCH --ntasks-per-node=1              # Run one primary task (the script itself) per node
#SBATCH --gpus-per-node=4                # Request 8 GPUs per node (adjust based on your hardware and trainer.n_gpus_per_node in train script)
#SBATCH --cpus-per-task=72               # Request all CPUs for the main task on the node (e.g., 72 on Raven)
#SBATCH --time=23:00:00               # Maximum wall clock time (e.g., 30 days)
#SBATCH --mem=500000                     # Memory per node (e.g., 500GB)
#SBATCH --chdir=./                       # Set working directory to submission directory.

# Optional: Specify a partition if required by your cluster setup
# #SBATCH --partition=your_gpu_partition

echo "========================================================"
echo "Starting Slurm Job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Nodes List: $SLURM_JOB_NODELIST"
echo "GPUs per Node: $SLURM_GPUS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Running on host: $(hostname)"
echo "Submitted from: $(pwd)"
echo "========================================================"

# Load necessary environment modules
echo "Loading environment modules..."
module purge
module load anaconda/3/2021.11  # Or a newer/different Anaconda version
module load cuda/12.6           # Or a CUDA version compatible with your PyTorch/TensorFlow

# Initialize Conda for shell interaction and activate your environment
echo "Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda activate .verl # Replace with your actual Conda environment name

echo "Modules currently loaded:"
module list
echo "Conda environment:"
conda info --envs | grep '*'
echo "========================================================"

# Network Settings for Multi-Node Training (adjust if necessary for your cluster)
export NCCL_DEBUG=${NCCL_DEBUG:-"INFO"} # Changed from TRACE to INFO to reduce verbosity
export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9" # Adjust to your InfiniBand devices if different
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-"3"}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-"0"}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-"1"}
# export NCCL_SOCKET_IFNAME=ib0 # Optional: specify network interface for NCCL if needed
# export NCCL_IB_DISABLE=0 # Ensure InfiniBand is enabled
# export NCCL_P2P_DISABLE=0 # Ensure P2P is enabled

echo "NCCL Environment Variables Set:"
env | grep NCCL
echo "========================================================"

# Ray Cluster Setup
echo "Setting up Ray cluster..."

# Get node hostnames
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\\n' ' '))
head_node=${nodes_array[0]}

# Get head node IP address
# Using srun to get IP on the head_node itself ensures correct resolution
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# If head_node_ip contains spaces (e.g. IPv6 and IPv4), try to pick IPv4
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 && ${#ADDR[1]} -lt 16 ]]; then # Basic check if first is IPv6-like and second is IPv4-like
        head_node_ip=${ADDR[1]}
    elif [[ ${#ADDR[0]} -lt 16 ]]; then
        head_node_ip=${ADDR[0]}
    else # Fallback, may need manual adjustment if complex multi-IP setup
        head_node_ip=${ADDR[0]}
    fi
    echo "Detected multiple IPs. Selected IP: $head_node_ip"
fi

# Define Ray head port and address
port=6379
ip_head=$head_node_ip:$port
export RAY_ADDRESS="http://$ip_head" # Export for trainer script if it uses ray.init(address="auto") or RAY_ADDRESS

echo "Head Node: $head_node"
echo "Head Node IP: $head_node_ip"
echo "Ray Head Address (ip_head): $ip_head"
echo "RAY_ADDRESS: $RAY_ADDRESS"

num_cpus_ray=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE * SLURM_GPUS_PER_NODE)) # Allocate CPUs for Ray based on total for task, ensures it's a multiple of GPUs for simpler division if Ray workers care
# Or simply use SLURM_CPUS_PER_TASK if Ray manages all resources on the node centrally
num_cpus_ray=${SLURM_CPUS_PER_TASK}

echo "Starting Ray HEAD on $head_node ($head_node_ip) with $num_cpus_ray CPUs and $SLURM_GPUS_PER_NODE GPUs"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --dashboard-host 0.0.0.0 --dashboard-port=8265 \
    --num-cpus "${num_cpus_ray}" --num-gpus "${SLURM_GPUS_PER_NODE}" \
    --block &
sleep 15 # Give head node time to start

worker_num=$(($SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    worker_node=${nodes_array[$i]}
    echo "Starting Ray WORKER $i on $worker_node with $num_cpus_ray CPUs and $SLURM_GPUS_PER_NODE GPUs"
    srun --nodes=1 --ntasks=1 -w "$worker_node" \
        ray start --address "$ip_head" \
        --num-cpus "${num_cpus_ray}" --num-gpus "${SLURM_GPUS_PER_NODE}" \
        --block &
    sleep 5 # Stagger worker starts
done
sleep 10 # Wait for all workers to connect
echo "Ray cluster setup initiated."
echo "========================================================"

# Ray Initialization Test
echo "Testing Ray cluster status..."
# srun on head node to execute the python test using the Ray cluster
srun --nodes=1 --ntasks=1 -w "$head_node" \
python3 -uc '
import ray
import os
try:
    # RAY_ADDRESS env var should be picked up by ray.init() if set,
    # otherwise "auto" for Slurm integration if Ray is new enough.
    # Explicitly using RAY_ADDRESS if you want to be sure.
    address = os.environ.get("RAY_ADDRESS", "auto")
    if address != "auto" and not address.startswith("http://"): # ray.init expects http:// for explicit address
        address = f"http://{address}"

    print(f"Attempting to connect to Ray at: {address}")
    ray.init(address=address)
    print("\\n=== Ray Cluster Status ===")
    print(f"Ray version: {ray.__version__}")
    print(f"Number of nodes: {len(ray.nodes())}")
    for node in ray.nodes():
        node_status = "Alive" if node["Alive"] else "Dead"
        print(f"  Node: {node.get(\\"NodeManagerAddress\\", node.get(\\"RayletSocketName\\"))}, "
              f"Hostname: {node.get(\\"NodeManagerHostname\\", \\"N/A\\")}, "
              f"Status: {node_status}, "
              f"CPUs: {node.get(\\"Resources\\", {}).get(\\"CPU\\")}, "
              f"GPUs: {node.get(\\"Resources\\", {}).get(\\"GPU\\")}")
    ray.shutdown()
    print("Ray initialization test successful!")
except Exception as e:
    print(f"Ray initialization test failed: {str(e)}")
    print("Please check Ray head and worker logs, and network configuration.")
'
echo "Ray test completed."
echo "========================================================"

# Define the path to your training script
TARGET_SCRIPT="recipe/slurm/train_multinode.sh"

echo "Executing training script: ${TARGET_SCRIPT}"
echo "Start time: $(date)"
echo "========================================================"

# Run the training script on the head node.
# The script itself should be responsible for distributing work via Ray.
# Pass Slurm environment variables that the script might need.
export SLURM_GPUS_PER_NODE # Make sure it's available to the script
export SLURM_JOB_NUM_NODES # Make sure it's available to the script

srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash "${TARGET_SCRIPT}"

echo "========================================================"
echo "Training script execution finished with exit code: $?"
echo "End time: $(date)"
echo "Slurm job finished."
echo "========================================================" 