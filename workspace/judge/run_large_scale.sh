#!/bin/bash -l

#SBATCH --job-name=large_scale_annotation
#SBATCH --output=/u/rfechner/jobs/large_scale_%j.out
#SBATCH --error=/u/rfechner/jobs/large_scale_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --chdir=/u/rfechner/verl/workspace/judge

echo "Starting large-scale annotation..."
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

# Run large-scale annotation focusing on negative reward samples
# (adjust --samples as needed - using 200 to capture more negative reward cases)
python large_scale_annotator.py --samples 200 --responses 10

echo "Annotation finished at: $(date)"
echo "Exit code: $?"
