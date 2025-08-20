#!/bin/bash
dataset="aime25"
project_dir="fixed_rollouts"
tensor_parallel_size=2  # Default tensor parallel size, can be overridden

# Array to store all job IDs
ALL_JOB_IDS=()

# Function to extract job IDs from run_rollouts.py output
extract_job_ids() {
    local output="$1"
    local job_ids_line=$(echo "$output" | grep "^JOB_IDS:")
    if [ -n "$job_ids_line" ]; then
        echo "$job_ids_line" | sed 's/^JOB_IDS://'
    fi
}

conda info

echo "Starting rollout jobs with tensor_parallel_size=${tensor_parallel_size}..."

# First rollout: 7b-entropy0.01
echo "Running 7b-entropy0.01..."
output1=$(python run_rollouts.py \
    --file data/${dataset}/test.parquet \
    --checkpoint out/fixed/Qwen/Qwen2.5-7B_entropy_0.01_kl_0.001_20250808_130223/best_checkpoint/actor \
    --identifier 7b-entropy0.01 \
    --project-dir ${project_dir} \
    --tensor-parallel-size ${tensor_parallel_size})
echo "$output1"
job_ids1=$(extract_job_ids "$output1")
if [ -n "$job_ids1" ]; then
    ALL_JOB_IDS+=($job_ids1)
fi

# Second rollout: 7b-entropy0.0
echo "Running 7b-entropy0.0..."
output2=$(python run_rollouts.py \
    --file data/${dataset}/test.parquet \
    --checkpoint out/fixed/Qwen/Qwen2.5-7B_entropy_0.0_kl_0.001_20250808_130226/best_checkpoint/actor \
    --identifier 7b-entropy0.0 \
    --project-dir ${project_dir} \
    --tensor-parallel-size ${tensor_parallel_size})
echo "$output2"
job_ids2=$(extract_job_ids "$output2")
if [ -n "$job_ids2" ]; then
    ALL_JOB_IDS+=($job_ids2)
fi

# Third rollout: 1.5b-entropy0.01
echo "Running 1.5b-entropy0.01..."
output3=$(python run_rollouts.py \
    --file data/${dataset}/test.parquet \
    --checkpoint out/fixed/Qwen/Qwen2.5-1.5B_entropy_0.01_kl_0.001_20250808_130215/best_checkpoint/actor \
    --identifier 1.5b-entropy0.01 \
    --project-dir ${project_dir} \
    --tensor-parallel-size ${tensor_parallel_size})
echo "$output3"
job_ids3=$(extract_job_ids "$output3")
if [ -n "$job_ids3" ]; then
    ALL_JOB_IDS+=($job_ids3)
fi

# Fourth rollout: 1.5b-entropy0.0
echo "Running 1.5b-entropy0.0..."
output4=$(python run_rollouts.py \
    --file data/${dataset}/test.parquet \
    --checkpoint out/fixed/Qwen/Qwen2.5-1.5B_entropy_0.0_kl_0.001_20250808_130206/best_checkpoint/actor \
    --identifier 1.5b-entropy0.0 \
    --project-dir ${project_dir} \
    --tensor-parallel-size ${tensor_parallel_size})
echo "$output4"
job_ids4=$(extract_job_ids "$output4")
if [ -n "$job_ids4" ]; then
    ALL_JOB_IDS+=($job_ids4)
fi

# Export the job IDs as environment variable
export ROLLOUT_JOB_IDS="${ALL_JOB_IDS[*]}"

echo ""
echo "================================================"
echo "All rollout jobs have been submitted!"
echo "Tensor Parallel Size: ${tensor_parallel_size}"
echo "Job IDs: ${ROLLOUT_JOB_IDS}"
echo ""
echo "To cancel all these jobs, run:"
echo "scancel ${ROLLOUT_JOB_IDS}"
echo ""
echo "Or use the environment variable:"
echo "scancel \$ROLLOUT_JOB_IDS"
echo "================================================"
