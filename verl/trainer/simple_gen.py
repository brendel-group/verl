import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl.utils.reward_score.math import compute_score as math_compute_score

def select_reward_fn(data_source):
    if data_source == 'DigitalLearningGmbH/MATH-lighteval' or data_source == 'lighteval/MATH':
        return math_compute_score
    else:
        raise NotImplementedError
    
def run_qwen_batched(local_dir='~/data/math', output_file='output.json', batch_size=64):
    # Load and prepare data
    local_dir = os.path.expanduser(local_dir)
    df = pd.read_parquet(os.path.join(local_dir, 'train.parquet')).head(50)
    num_samples = 128
    
    data_sources = df['data_source']
    if len(set(data_sources)) > 1:
        raise RuntimeError("Mixed data source. Currently not supported.")
    reward_fn = select_reward_fn(data_source=data_sources.iloc[0])

    # Load tokenizer and model with FlashAttention (if supported)
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # Speed boost (if supported)
    )
    model.eval()
    print("Loaded tokenizer, model. Starting to generate...")

    results = []
    prompts = df['prompt'].apply(lambda arr : arr[0]['content']) # arrs only have len 1
    ground_truths = df['reward_model'].apply(lambda d: d['ground_truth'])

    # Run batched inference
    for p_i, g_i in tqdm(zip(prompts, ground_truths)):

        p_i_buffer = []
        for i in range(0, num_samples, batch_size):
            qs = [p_i] * batch_size
            inputs = tokenizer(qs, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=1
                )
                
            # Remove prompt from generated outputs
            input_length = inputs['input_ids'].shape[1]
            trimmed_outputs = []
            for output in outputs:
                gen_tokens = output[input_length:]
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                trimmed_outputs.append(gen_text)
            p_i_buffer.extend(trimmed_outputs)

        ytrues = [g_i] * len(p_i_buffer)
        scores = list(map(reward_fn, p_i_buffer, ytrues))
        
        results.append(
            {
                'prompt' : p_i,
                'answers' : p_i_buffer,
                'scores' : scores
            }
        )

    # Save to JSON
    dataset_name = local_dir.split('/')[-1]
    output_file = dataset_name + output_file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved to {output_file}")

if __name__ == '__main__':
    run_qwen_batched()