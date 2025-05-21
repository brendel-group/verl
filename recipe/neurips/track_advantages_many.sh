#!/usr/bin/env bash
# This script calls track_advantages.sh for multiple models/experiments.
# Modify the list of experiments as needed.

# Example: Process all checkpoints in a specific experiment folder
# bash recipe/neurips/track_advantages.sh /fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/qwen_2.5_3b_base_sft_data_Qwen2.5-14B_openai_math_n_8_bsz_128_epochs_1_kl_coef_0.0_step_0_3b_7b_14b_epochs_5_lr_1e-6_bsz_256_micro_bsz_256_total_training_steps_90/

# Example: Process a specific checkpoint step (e.g., step 60)
# bash recipe/neurips/track_advantages.sh /fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/qwen_2.5_3b_base_sft_data_Qwen2.5-3B_openai_math_n_8_bsz_512_epochs_1_kl_coef_0.0_step_0_3b_7b_14b_epochs_5_lr_1e-6_bsz_256_micro_bsz_256_total_training_steps_90/global_step_60

# Add more calls to track_advantages.sh below for other experiments
# For example, if you have a list of experiments similar to eval_many.sh:

# Experiments from eval_many.sh (commented out, adapt as needed)
# echo "--- Tracking advantages for base models ---"
# bash recipe/neurips/track_advantages.sh Qwen/Qwen2.5-1.5B
# bash recipe/neurips/track_advantages.sh Qwen/Qwen2.5-7B
# ... and so on for other base models

echo "--- Tracking advantages for specific experiment checkpoints ---"

# Example from your eval_many.sh, adapt paths and uncomment if these are experiment folders containing global_step_* subfolders
# Make sure these paths point to the parent experiment folder, not individual global_step_* folders, unless you want to process only one.

# bash recipe/neurips/track_advantages.sh /fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/qwen_2.5_3b_base_sft_data_DeepSeek-R1-Distill-Qwen-7B_openai_math_n_8_bsz_512_epochs_1_kl_coef_0.0_step_0_epochs_5_lr_1e-6_bsz_512_micro_bsz_512/
# bash recipe/neurips/track_advantages.sh /fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/qwen_2.5_3b_base_sft_data_Qwen2.5-3B_openai_math_n_8_bsz_512_epochs_1_kl_coef_0.0_step_0_epochs_5_lr_1e-6_bsz_512_micro_bsz_512/

# If you want to process a specific checkpoint like in your example:
# /fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/Qwen2.5-Math-1.5B_dsr_sub_n_8_bsz_64_epochs_1_kl_coef_0.0/advantage_tracking
# The command would be (assuming Qwen2.5-Math-1.5B_dsr_sub_n_8_bsz_64_epochs_1_kl_coef_0.0 is the experiment folder):
# bash recipe/neurips/track_advantages.sh /fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/Qwen2.5-Math-1.5B_dsr_sub_n_8_bsz_64_epochs_1_kl_coef_0.0/

# To process a specific checkpoint (e.g. step 0 for the above example, if it exists as global_step_0):
# bash recipe/neurips/track_advantages.sh /fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/Qwen2.5-Math-1.5B_dsr_sub_n_8_bsz_64_epochs_1_kl_coef_0.0/global_step_0

# Add the specific lines from your eval_many.sh that you want to run advantage tracking for:
# For example, to track advantages for these specific checkpoints (assuming they are experiment folders or specific checkpoint folders):

# From your eval_many.sh, these look like specific checkpoint paths already.
# The track_advantages.sh script can handle these directly.

# Example: one specific checkpoint path from your list
CHECKPOINT_PATH_1="/fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/qwen_2.5_3b_base_sft_data_Qwen2.5-3B_openai_math_n_8_bsz_512_epochs_1_kl_coef_0.0_step_0_3b_7b_14b_epochs_5_lr_1e-6_bsz_256_micro_bsz_256_total_training_steps_90/global_step_60"
echo "Tracking advantages for: ${CHECKPOINT_PATH_1}"
bash recipe/neurips/track_advantages.sh "${CHECKPOINT_PATH_1}"

# Another example (if it's an experiment folder with multiple steps)
# EXP_PATH_2="/fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/qwen_2.5_math_1.5b_base_sft_data_Qwen2.5-Math-1.5B_pi1_r128_n_8_bsz_64_epochs_1_kl_coef_0.0_step_0_all_random_state_100_47x20_epochs_4_lr_1e-5_bsz_128_micro_bsz_128_total_training_steps_30/"
# echo "Tracking advantages for all steps in: ${EXP_PATH_2}"
# bash recipe/neurips/track_advantages.sh "${EXP_PATH_2}"


# Add all the paths from your `eval_many.sh` that you want to process for advantage tracking.
# Remember that `track_advantages.sh` will look for `global_step_*` subdirectories if you provide a parent directory.
# If you provide a direct path to a `global_step_X` folder, it will process only that one.

PATHS_TO_TRACK=(
    # Paste paths from your eval_many.sh here, for example:
    "/fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/qwen_2.5_3b_base_sft_data_DeepSeek-R1-Distill-Qwen-7B_openai_math_n_8_bsz_512_epochs_1_kl_coef_0.0_step_0_epochs_5_lr_1e-6_bsz_512_micro_bsz_512/global_step_30"
    "/fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/qwen_2.5_3b_base_sft_data_Qwen2.5-3B_openai_math_n_8_bsz_512_epochs_1_kl_coef_0.0_step_0_epochs_5_lr_1e-6_bsz_512_micro_bsz_512/global_step_50"
    # ... Add all other paths you need from eval_many.sh
    "/fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/qwen_2.5_3b_base_sft_data_Qwen2.5-3B_openai_math_n_8_bsz_512_epochs_1_kl_coef_0.0_step_0_3b_7b_14b_epochs_5_lr_1e-6_bsz_256_micro_bsz_256_total_training_steps_90/"
    "/fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/qwen_2.5_math_1.5b_base_sft_data_Qwen2.5-Math-1.5B_pi1_r128_n_8_bsz_64_epochs_1_kl_coef_0.0_step_0_all_random_state_100_47x20_epochs_4_lr_1e-5_bsz_128_micro_bsz_128_total_training_steps_30/"
    "/fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/qwen_2.5_math_1.5b_base_sft_data_Qwen2.5-Math-1.5B_pi1_r128_n_8_bsz_64_epochs_1_kl_coef_0.0_step_0_all_single_datapoint_124_random_state_42_epochs_4_lr_1e-5_bsz_128_micro_bsz_128_total_training_steps_30/"
)

for path_or_checkpoint in "${PATHS_TO_TRACK[@]}"; do
    echo "----------------------------------------------------"
    echo "Calling track_advantages.sh for: ${path_or_checkpoint}"
    bash recipe/neurips/track_advantages.sh "${path_or_checkpoint}"
    echo "Finished tracking for: ${path_or_checkpoint}"
    echo "----------------------------------------------------"
done

echo "All advantage tracking tasks from the list are complete." 