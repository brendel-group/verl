cd ~/ && python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=/home/rfechner/data/gsm8k/test.parquet \
    data.prompt_key=prompt \
	data.n_prompts=3 \
    data.n_samples=2 \
    data.output_path=/home/rfechner/data/output_gen_test.parquet \
    model.path=/home/rfechner/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=2048 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8
