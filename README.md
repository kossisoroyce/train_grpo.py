# GRPO Trainer v2.0

<div align="center">

**Advanced GRPO/GSPO Training Framework for LLM Fine-tuning on Math & Reasoning Datasets**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/grpo_training_super.ipynb)

</div>

---

## Overview

GRPO Trainer is a production-ready framework for training Large Language Models using **Group Relative Policy Optimization (GRPO)** and **GSPO** (enhanced GRPO with sequence-level importance sampling). It provides a modular, configurable, and scalable solution for fine-tuning models on mathematical reasoning tasks.

### Key Features

- **GRPO & GSPO Support**: Standard GRPO plus dr_grpo, IPO, and SimPO loss variants
- **Multi-Dataset Support**: GSM8K, MATH, SVAMP, AQuA-RAT, and custom datasets
- **Modular Reward System**: 8 configurable reward functions with weighted combinations
- **VLM Support**: Gibberish penalty for vision-language models (Qwen2.5-VL fix)
- **Flexible Configuration**: YAML-based configs with CLI overrides
- **Memory Efficient**: 4-bit/8-bit quantization + 8-bit AdamW optimizer
- **Distributed Training**: DeepSpeed ZeRO-2/3 and FSDP support
- **Custom Delimiters**: Configurable reasoning/answer tags
- **Evaluation Pipeline**: Built-in evaluation with detailed metrics
- **Docker Ready**: Full containerization support
- **Modern CLI**: Rich terminal interface with progress tracking
- **Colab Notebook**: Ready-to-run super notebook for quick experiments

---

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/kossisoroyce/grpo-trainer.git
cd grpo-trainer
pip install -e ".[all]"
```

### Quick Install

```bash
pip install -e .
```

### With Optional Dependencies

```bash
# Development tools
pip install -e ".[dev]"

# DeepSpeed support
pip install -e ".[deepspeed]"

# Flash Attention (requires CUDA)
pip install flash-attn --no-build-isolation
```

---

## Quick Start

### Option 1: Use the Colab Notebook (Easiest)

Open the **[Super Notebook](notebooks/grpo_training_super.ipynb)** in Google Colab for a fully guided experience with form-based configuration.

### Option 2: Command Line

```bash
# 1. Generate a config
grpo-train init-config config.yaml --preset gsm8k

# 2. Start training
grpo-train train -c config.yaml

# 3. Evaluate
grpo-eval outputs/grpo-default -d gsm8k -s test
```

### Option 3: Python API

```python
from grpo_trainer import GRPOTrainerWrapper, Config

config = Config.from_yaml("configs/default.yaml")
trainer = GRPOTrainerWrapper(config)
trainer.setup()
trainer.train()
trainer.save()
```

---

## Configuration

### YAML Configuration

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  torch_dtype: "bfloat16"
  attn_implementation: "flash_attention_2"

lora:
  enabled: true
  r: 16
  lora_alpha: 64

data:
  name: "gsm8k"
  use_one_shot: true

reward:
  correctness_weight: 2.0
  format_weight: 0.5
  reasoning_quality_weight: 0.3
  gibberish_penalty_weight: 1.0  # For VLMs
  
  # Custom delimiters (optional)
  use_custom_delimiters: true
  reasoning_start: "<REASONING>"
  reasoning_end: "</REASONING>"
  answer_start: "<SOLUTION>"
  answer_end: "</SOLUTION>"

training:
  learning_rate: 5.0e-6
  per_device_train_batch_size: 2
  num_train_epochs: 1
  output_dir: "outputs/my-run"
  optim: "adamw_8bit"  # Memory-efficient optimizer
  
  # GSPO options
  loss_type: "dr_grpo"  # or "grpo", "ipo", "simpo"
  importance_sampling_level: "sequence"
  mask_truncated_completions: false
```

### CLI Options

```bash
grpo-train train --help

Options:
  -c, --config PATH       Path to YAML configuration file
  -m, --model TEXT        Model name or path
  -d, --dataset TEXT      Dataset name (gsm8k, math, svamp, etc.)
  -o, --output TEXT       Output directory
  --lr FLOAT              Learning rate
  -b, --batch-size INT    Batch size per device
  -e, --epochs INT        Number of epochs
  --lora/--no-lora        Enable/disable LoRA
  -r, --lora-rank INT     LoRA rank
  --resume PATH           Resume from checkpoint
```

---

## Supported Datasets

| Dataset | Description | Task |
|---------|-------------|------|
| **GSM8K** | Grade School Math 8K | Arithmetic word problems |
| **MATH** | Competition Mathematics | Olympiad-level math |
| **SVAMP** | Simple Variations on AMPs | Arithmetic word problems |
| **AQuA-RAT** | Algebra Question Answering | Algebraic reasoning |
| **Custom** | Your own data | Any format |

### Using Custom Datasets

```yaml
data:
  name: "custom"
  custom_path: "path/to/your/data.jsonl"
```

Your JSONL should have `question` and `answer` fields.

---

## Reward Functions

The framework includes 8 configurable reward functions:

| Reward | Default | Description |
|--------|---------|-------------|
| **Correctness** | 2.0 | Exact match with ground truth |
| **Format** | 0.5 | Follows XML/custom reasoning format |
| **Integer** | 0.5 | Answer is a valid number |
| **XML Count** | 0.5 | Proper XML tag structure |
| **Length Penalty** | 0.0 | Penalizes overly long responses |
| **Reasoning Quality** | 0.0 | Rewards step-by-step reasoning |
| **Gibberish Penalty** | 0.0 | Penalizes VLM artifacts (addCriterion fix) |
| **Custom Delimiter** | 0.0 | Rewards custom tag formats |

### Custom Rewards

```python
from grpo_trainer.rewards import BaseReward

class MyCustomReward(BaseReward):
    def compute(self, completions, **kwargs):
        # Your reward logic here
        return [1.0 for _ in completions]
```

---

## GSPO (Enhanced GRPO)

GSPO uses sequence-level importance sampling for more stable training:

```yaml
training:
  loss_type: "dr_grpo"           # GSPO variant
  importance_sampling_level: "sequence"
  mask_truncated_completions: false
  optim: "adamw_8bit"            # 50% memory savings
```

Or via CLI:

```bash
grpo-train train -c configs/gspo_vision.yaml
```

---

## Distributed Training

### Multi-GPU with DeepSpeed

```bash
# ZeRO Stage 2
./scripts/train_deepspeed.sh -c configs/default.yaml --ds-config configs/deepspeed_zero2.json

# ZeRO Stage 3 (for larger models)
./scripts/train_deepspeed.sh -c configs/gsm8k_7b.yaml --ds-config configs/deepspeed_zero3.json
```

### Multi-GPU with torchrun

```bash
torchrun --nproc_per_node=4 -m grpo_trainer.cli train -c config.yaml
```

---

## Docker

### Build and Run

```bash
# Build image
docker build -t grpo-trainer .

# Run training
docker-compose run train

# Interactive shell
docker-compose run grpo-trainer bash
```

### With GPU Support

```bash
docker run --gpus all -v $(pwd)/outputs:/app/outputs grpo-trainer \
    grpo-train train -c /app/configs/default.yaml
```

---

## Project Structure

```text
grpo-trainer/
├── src/grpo_trainer/
│   ├── __init__.py       # Package exports
│   ├── cli.py            # Command-line interface
│   ├── config.py         # Configuration classes
│   ├── datasets.py       # Dataset loading & preprocessing
│   ├── model.py          # Model loading utilities
│   ├── rewards.py        # Reward functions (8 types)
│   ├── trainer.py        # GRPO/GSPO trainer wrapper
│   └── evaluate.py       # Evaluation utilities
├── configs/
│   ├── default.yaml      # Default configuration
│   ├── gsm8k_7b.yaml     # Config for 7B models
│   ├── math_dataset.yaml # MATH dataset config
│   ├── gspo_vision.yaml  # GSPO for vision models
│   └── deepspeed_*.json  # DeepSpeed configs
├── notebooks/
│   └── grpo_training_super.ipynb  # 🚀 Colab notebook
├── scripts/
│   ├── train.sh          # Training script
│   ├── train_deepspeed.sh
│   └── evaluate.sh
├── tests/                # Unit tests
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
└── README.md
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [grpo_training_super.ipynb](notebooks/grpo_training_super.ipynb) | Full training pipeline with form-based config |

The notebook includes:
- One-click installation
- Form-based configuration (works in Colab)
- GRPO and GSPO support
- All 8 reward functions
- Evaluation and inference
- Model saving and Hub upload

---

## Legacy Script

The original single-file `train_grpo.py` is preserved for reference.

---

## System Requirements

- **Python**: 3.10+
- **PyTorch**: 2.1+
- **CUDA**: 11.8+ (recommended)
- **GPU Memory**:
  - 1.5B model: ~16GB
  - 3B model: ~24GB (or 16GB with 4-bit)
  - 7B model: ~40GB (or 24GB with 4-bit)

---

## Citation

If you use this framework, please cite:

```bibtex
@software{grpo_trainer,
  title = {GRPO Trainer: Advanced GRPO Training Framework},
  author = {kossisoroyce},
  year = {2024},
  url = {https://github.com/kossisoroyce/grpo-trainer}
}
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

---

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.
