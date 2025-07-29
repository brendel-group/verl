#!/bin/bash
#SBATCH --job-name=grade_rollouts
#SBATCH --output=grade_rollouts_%j.out
#SBATCH --error=grade_rollouts_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --partition=cpu

# Set up environment
echo "Starting grading job at $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"

# Change to the working directory
cd /u/rfechner/verl

# Create output directory
mkdir -p workspace/rollouts/graded_results


# Load modules and activate environment
module purge
module load anaconda/3/2023.03
module load cuda/12.1

eval "$(conda shell.bash hook)"
conda activate /u/rfechner/conda-envs/verl

# Run the grading script
echo "Starting grading process..."
python workspace/rollouts/grade_generic.py \
    --rollouts_dir /u/rfechner/out/rollouts/Qwen \
    --output_dir workspace/rollouts/graded_results

# Check exit status
if [ $? -eq 0 ]; then
    echo "Grading completed successfully at $(date)"
else
    echo "Grading failed at $(date)"
    exit 1
fi

# List the results
echo "Generated files:"
ls -la workspace/rollouts/graded_results/

echo "Job completed at $(date)"
