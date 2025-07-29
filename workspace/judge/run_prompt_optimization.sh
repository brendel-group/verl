#!/bin/bash -l

#SBATCH --job-name=prompt_optimization
#SBATCH --output=./prompt_opt_%j.out
#SBATCH --error=./prompt_opt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --chdir=/u/rfechner/verl/workspace/judge

echo "Starting prompt optimization experiment..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Load modules and activate environment
module purge
module load anaconda/3/2023.03
module load cuda/12.1

eval "$(conda shell.bash hook)"
conda activate /u/rfechner/conda-envs/verl

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

# Run the expanded prompt optimization experiment (100 samples, 10 responses each)
python prompt_optimizer.py --samples 50 --responses 10

echo "Experiment finished at: $(date)"
echo "Exit code: $?"
