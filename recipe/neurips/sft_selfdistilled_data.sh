set -x
# Check this is okay for SFT on self-distilled datasets
# weight_key  = 'weight' (None for standard GSM8K)
# prompt_key  = 'prompt' (extra_info for standard GSM8K)
# response_key = 'extra_info' (extra_info for standard GSM8K)
# prompt_dict_keys = ['content'] (question for standard GSM8K)
# response_dict_keys = ['answer'] (answer for standard GSM8K)

epochs=2
lr=1e-5
bsz=256 #for 7b even 512 works
micro_bsz=256
total_training_steps=90
#dir=qwen2.5_1.5b_grpo_gsm8k_epochs_2_rollouts_5_no_kl_step_0
# dir=qwen2.5_1.5b_grpo_gsm8k_epochs_2_rollouts_5_no_kl_step_0
# dir_val=qwen2.5_1.5b_grpo_gsm8k_epochs_2_rollouts_5_no_kl_step_0

#dir=Qwen2.5-1.5B_openai_math_n_32_bsz_512_epochs_1_kl_coef_0.0_step_0
#dir=Qwen2.5-7B_openai_math_n_8_bsz_512_epochs_10_kl_coef_0.0_step_0
#dir=Qwen2.5-7B_openai_math_n_8_bsz_512_epochs_10_kl_coef_0.0_step_0_single_datapoint_5645
dir=$1
experiment_name=qwen_2.5_1.5b_base_sft_data_${dir}_epochs_${epochs}_lr_${lr}_bsz_${bsz}_micro_bsz_${micro_bsz}_total_training_steps_${total_training_steps}
save_path=/fast/pmayilvahanan/post_training/verl_checkpoints/self_distillation_neurips/Qwen/$experiment_name
model_path=Qwen/Qwen2.5-1.5B

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/fast/pmayilvahanan/post_training/self_distilled_datasets_neurips/${dir}/train.parquet \
    data.val_files=/fast/pmayilvahanan/post_training/self_distilled_datasets_neurips/openai_math/test.parquet \
    data.prompt_key=prompt \
    data.response_key=extra_info \
    data.max_length=5120 \
    data.truncation=right \
    optim.lr=$lr \
    optim.use_likelihood_loss=False \
    +data.prompt_dict_keys=['content'] \
    +data.response_dict_keys=['answer'] \
    data.train_batch_size=$bsz \
    data.micro_batch_size=$micro_bsz \
    model.partial_pretrain=$model_path \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=self_distillation_neurips \
    trainer.experiment_name=$experiment_name \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=$epochs \
    trainer.total_training_steps=$total_training_steps \
    +trainer.save_checkpoint_steps=6 \
    +trainer.validate_every_n_steps=6 \
    trainer.default_hdfs_dir=null \
    +trainer.save_config=True \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true 