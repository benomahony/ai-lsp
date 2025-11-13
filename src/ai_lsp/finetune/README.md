# Fine-Tuning Pipeline

This module provides tools to fine-tune small language models (SLMs) on your OpenTelemetry traces from ai-lsp usage.

## Overview

The fine-tuning pipeline converts your otel traces into a training dataset and fine-tunes a model to better understand your codebase and coding patterns. It uses **MLX** for efficient training on Apple Silicon.

## Features

- **Dataset Conversion**: Extracts code analysis examples from otel traces
- **MLX-Optimized**: Uses Apple's MLX framework for fast, efficient training on Apple Silicon
- **LoRA Fine-Tuning**: Memory-efficient training with Low-Rank Adaptation
- **Flexible Models**: Supports any MLX-compatible model from Hugging Face
- **Simple CLI**: Three commands to go from traces to fine-tuned model

## Quick Start

### 1. Prepare Dataset

Convert your otel traces to a training dataset:

```bash
uv run ai-lsp finetune prepare-dataset ai-lsp-otel.jsonl
```

This creates `dataset.json` with training examples extracted from your traces.

### 2. Fine-Tune Model

Fine-tune a model on your dataset:

```bash
uv run ai-lsp finetune train dataset.json --model-name mlx-community/gemma-2-2b-it-4bit
```

### 3. Run Complete Pipeline

Or run everything in one command:

```bash
uv run ai-lsp finetune pipeline ai-lsp-otel.jsonl
```

## Supported Models

All MLX-compatible models from Hugging Face, including:

- `mlx-community/gemma-2-2b-it-4bit` - Gemma 2B 4-bit (default, recommended)
- `mlx-community/Phi-4-4bit` - Phi-4 14B 4-bit
- `mlx-community/Qwen2.5-7B-Instruct-4bit` - Qwen 2.5 7B 4-bit
- `mlx-community/Llama-3.2-3B-Instruct-4bit` - Llama 3.2 3B 4-bit

Browse more models at [mlx-community on Hugging Face](https://huggingface.co/mlx-community)

## Requirements

The fine-tuning process requires:
- **Apple Silicon Mac** (M1, M2, M3, M4)
- ~8GB RAM for smaller models (2B-3B parameters)
- ~16GB RAM for larger models (7B+ parameters)
- ~10GB disk space for model and checkpoints

MLX is optimized for Apple Silicon and runs efficiently on Mac hardware.

## Configuration

Customize training with options:

```bash
uv run ai-lsp finetune train dataset.json \
  --model-name mlx-community/gemma-2-2b-it-4bit \
  --output-dir ./my-finetuned-model \
  --num-epochs 5 \
  --batch-size 4 \
  --learning-rate 1e-5
```

## How It Works

### Dataset Creation

The pipeline extracts training examples from your otel traces:

1. Finds `agent run` spans from pydantic-ai
2. Extracts the code being analyzed
3. Extracts the issues identified by the AI
4. Creates training examples with system prompt, user query, and assistant response

### Training Process

Uses LoRA with MLX for efficient fine-tuning:

1. Loads base model (automatically quantized by MLX)
2. Adds LoRA adapter layers (only these are trained)
3. Fine-tunes on your dataset using MLX optimizations
4. Fuses adapters with base model
5. Saves the complete fine-tuned model

### Using the Fine-Tuned Model

After training, you can use the model with MLX:

```python
from mlx_lm import load, generate

model, tokenizer = load("./finetuned-model/model")
response = generate(model, tokenizer, prompt="Analyze this code...")
```

Or integrate directly into ai-lsp by updating the model path in settings.

## Example Output

After fine-tuning on 100+ examples, the model learns:
- Your code style and patterns
- Common issues in your codebase
- Project-specific semantic problems
- Better context about your domain

## Tips

- Collect at least 50-100 diverse examples for best results
- More epochs (3-5) work better with smaller datasets
- Use larger models (7B+) for complex codebases
- Fine-tune periodically as your codebase evolves

## Troubleshooting

**Out of memory errors:**
- Use a smaller model (2B instead of 7B)
- Reduce `--batch-size` (try 1)
- Close other applications to free RAM
- Consider a model with more aggressive quantization

**Poor results:**
- Collect more training examples (50-100 minimum)
- Increase `--num-epochs` (try 5-10)
- Try a larger base model
- Check dataset quality with `prepare-dataset`

**Slow training:**
- MLX is optimized for Apple Silicon - training on Mac should be fast
- Ensure you're running on Apple Silicon (not Intel)
- Try a smaller batch size if memory-constrained
