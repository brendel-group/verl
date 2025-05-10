set -x
# Check this is okay for SFT on self-distilled datasets
# weight_key  = 'weight' (None for standard GSM8K)
# prompt_key  = 'prompt' (extra_info for standard GSM8K)
# response_key = 'extra_info' (extra_info for standard GSM8K)
# prompt_dict_keys = ['content'] (question for standard GSM8K)
# response_dict_keys = ['answer'] (answer for standard GSM8K)

epochs=3
lr=1e-6
bsz=256 #for 7b even 512 works
#dir=qwen2.5_1.5b_grpo_gsm8k_epochs_2_rollouts_5_no_kl_step_0
# dir=qwen2.5_1.5b_grpo_gsm8k_epochs_2_rollouts_5_no_kl_step_0
# dir_val=qwen2.5_1.5b_grpo_gsm8k_epochs_2_rollouts_5_no_kl_step_0

#dir=Qwen2.5-1.5B_openai_math_n_32_bsz_512_epochs_1_kl_coef_0.0_step_0
#dir=Qwen2.5-7B_openai_math_n_8_bsz_512_epochs_10_kl_coef_0.0_step_0
#dir=Qwen2.5-7B_openai_math_n_8_bsz_512_epochs_10_kl_coef_0.0_step_0_single_datapoint_5645
dir_train=Qwen2.5-7B_dapo_math_17k_n_8_bsz_1024_epochs_1_kl_coef_0.0_step_0_all_neg_if_positive_ratio_all_seed_42
dir_val=Qwen2.5-7B_dapo_math_17k_n_8_bsz_1024_epochs_1_kl_coef_0.0_step_0_all
experiment_name=qwen_2.5_7b_base_sft_data_${dir_train}_epochs_${epochs}_lr_${lr}_bsz_${bsz}
save_path=/fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/$experiment_name
model_path=Qwen/Qwen2.5-7B

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/fast/pmayilvahanan/post_training/self_distilled_datasets_neurips/${dir_train}/train.parquet \
    data.val_files=/fast/pmayilvahanan/post_training/self_distilled_datasets_neurips/${dir_val}/test.parquet \
    data.prompt_key=prompt \
    data.response_key=extra_info \
    data.max_length=4096 \
    data.truncation=right \
    data.weight_key=weight \
    optim.lr=$lr \
    optim.use_likelihood_loss=False \
    +data.prompt_dict_keys=['content'] \
    +data.response_dict_keys=['answer'] \
    data.train_batch_size=$bsz \
    data.micro_batch_size=256 \
    model.partial_pretrain=$model_path \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=self_distillation_neurips \
    trainer.experiment_name=$experiment_name \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=$epochs \
    +trainer.save_checkpoint_steps=-1 \
    +trainer.validate_every_n_steps=1 \
    trainer.default_hdfs_dir=null \
    +trainer.save_config=True \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true $@ 