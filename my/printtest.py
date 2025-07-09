import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_qwen_batched(local_dir='~/data/math', output_file='output.json', batch_size=8):
    # Load and prepare data
    local_dir = os.path.expanduser(local_dir)
    df = pd.read_parquet(os.path.join(local_dir, 'train.parquet')).head(1)  # Load only the first row

    # Extract the first prompt
    first_prompt = df.iloc[0]['prompt'][0]['content']

    # Repeat the first prompt 64 times
    prompts = [first_prompt] * 64

    # Load tokenizer and model with FlashAttention (if supported)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # Speed boost (if supported)
    )
    model.eval()

    results = []

    # Run batched inference
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_beams=1
            )
            
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Strip prompt from decoded outputs to get the model's continuation
        for prompt_text, full_text in zip(batch_prompts, decoded_outputs):
            result = {
                "prompt": prompt_text,
                "answer": full_text[len(prompt_text):].strip()
            }
            results.append(result)

    # Save to JSON
    dataset_name = local_dir.split('/')[-1]
    output_file = dataset_name + output_file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} results to {output_file}")

if __name__ == '__main__':
    run_qwen_batched()
