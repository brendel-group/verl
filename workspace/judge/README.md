# Simple Mathematical Error Categorization

Two focused scripts for mathematical error categorization:

## Files

1. **`prompt_optimizer.py`** - Find the best prompt template
2. **`large_scale_annotator.py`** - Run large-scale annotation with majority voting
3. **`parse.py`** - Response parsing utilities (unchanged)

## Usage

### 1. Optimize Prompts
```bash
# Quick test
python prompt_optimizer.py --samples 5 --responses 3

# Full optimization 
python prompt_optimizer.py --samples 10 --responses 10

# SLURM job
sbatch run_prompt_optimization.sbatch
```

### 2. Large-Scale Annotation
```bash
# Small run
python large_scale_annotator.py --samples 50 --responses 5

# Large run
python large_scale_annotator.py --samples 200 --responses 5

# SLURM job
sbatch run_large_scale.sbatch
```

## Error Categories

1. **Computational Error** - arithmetic/algebraic mistakes
2. **Conceptual Misunderstanding** - wrong approach
3. **Incomplete Solution** - didn't finish
4. **Wrong Method** - inappropriate technique  
5. **Logic Error** - flawed reasoning
6. **Mislabeling** - student was actually correct

## Output Files

- `prompt_optimization_results_*.json` - Prompt testing results
- `large_scale_results_*.json` - Full annotation results
- `large_scale_summary_*.csv` - Summary for analysis

## Key Features

- **Simple**: Just 2 main scripts
- **Fast**: Optimized for efficiency
- **Robust**: Majority voting for quality
- **Complete**: From optimization to large-scale deployment
