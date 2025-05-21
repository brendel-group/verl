#!/bin/bash -l

#SBATCH --job-name=verl_train_job      # Descriptive job name
#SBATCH --output=/u/prama/jobs/slurm_verl_train_%j.out # Standard output file (%j expands to job ID)
#SBATCH --error=/u/prama/jobs/slurm_verl_train_%j.err  # Standard error file (%j expands to job ID)
#SBATCH --nodes=1                        # Request 1 node (as per NNODES=1 in your script)
#SBATCH --ntasks-per-node=1              # Run the shell script as a single task on this node
#SBATCH --cpus-per-task=72               # Allocate all CPUs on the node (Raven GPU nodes often have 72 cores, adjust if different)
#SBATCH --gres=gpu:a100:4                # Request 8 A100 GPUs (based on n_gpus_per_node=8 in your script)
                                         # IMPORTANT: Verify Raven node configuration for 8 GPUs/node. Examples often show 4.
#SBATCH --time=23:00:00                  # Maximum wall clock time (e.g., 24 hours). Adjust as needed.
#SBATCH --mem=500000                          # Request all available memory on the node. Can be specific e.g., "200G".
#SBATCH --chdir=./                       # Set working directory to submission directory.

# Optional: Specify a partition if required by your cluster setup (e.g., gpu_a100)
# #SBATCH --partition=your_gpu_partition

echo "========================================================"
echo "Starting Slurm Job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Submitted from: $(pwd)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated GPUs: $SLURM_JOB_GPUS (CUDA_VISIBLE_DEVICES likely set to $CUDA_VISIBLE_DEVICES)"
echo "========================================================"

# Load necessary environment modules
# IMPORTANT: Adjust these modules and their versions based on the specific
# requirements of your 'verl' application and the script being run.
echo "Loading environment modules..."
module purge
module load anaconda/3/2021.11  # Or a newer/different Anaconda version suitable for verl
module load cuda/12.6 
module list
# Initialize Conda for shell interaction and activate your specific environment
# Replace 'your_verl_conda_env_name' with the actual name of your Conda environment
eval "$(conda shell.bash hook)"
conda activate .verl
          # Or a CUDA version compatible with your PyTorch/TensorFlow and GPU drivers

# Add other modules if verl or its dependencies require them explicitly
# For example, if specific compilers or MPI libraries are needed:
# module load gcc/10 impi/2021.2

echo "Modules currently loaded:"
module list
echo "========================================================"

# Define the path to your shell script
# If you want to run a different script, change this path.
# This path is relative to the directory where you run `sbatch run_verl_job.sbatch`.
TARGET_SCRIPT="recipe/emnlp/train_1.5_dsr_sub.sh"

echo "Executing script: ${TARGET_SCRIPT}"
echo "Start time: $(date)"
echo "========================================================"

# Run the user's shell script using srun
# srun ensures proper Slurm integration for the main task.
srun bash "${TARGET_SCRIPT}"

echo "========================================================"
echo "Script execution finished with exit code: $?"
echo "End time: $(date)"
echo "Slurm job finished."
echo "========================================================"
