"""Tests for configuration module."""

import pytest
import tempfile
from pathlib import Path

from grpo_trainer.config import (
    Config,
    ModelConfig,
    LoRAConfig,
    DataConfig,
    RewardConfig,
    TrainingConfig,
)


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_default_values(self):
        config = ModelConfig()
        assert config.name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert config.torch_dtype == "bfloat16"
        assert config.device_map == "auto"
    
    def test_custom_values(self):
        config = ModelConfig(
            name="meta-llama/Llama-2-7b",
            torch_dtype="float16",
            load_in_4bit=True,
        )
        assert config.name == "meta-llama/Llama-2-7b"
        assert config.torch_dtype == "float16"
        assert config.load_in_4bit is True


class TestLoRAConfig:
    """Tests for LoRAConfig."""
    
    def test_default_values(self):
        config = LoRAConfig()
        assert config.enabled is True
        assert config.r == 16
        assert config.lora_alpha == 64
        assert "q_proj" in config.target_modules
    
    def test_disabled(self):
        config = LoRAConfig(enabled=False)
        assert config.enabled is False


class TestDataConfig:
    """Tests for DataConfig."""
    
    def test_default_dataset(self):
        config = DataConfig()
        assert config.name == "gsm8k"
        assert config.split == "train"
        assert config.use_one_shot is True
    
    def test_custom_dataset(self):
        config = DataConfig(
            name="math",
            split="train",
            max_samples=1000,
        )
        assert config.name == "math"
        assert config.max_samples == 1000


class TestRewardConfig:
    """Tests for RewardConfig."""
    
    def test_default_weights(self):
        config = RewardConfig()
        assert config.correctness_weight == 2.0
        assert config.format_weight == 0.5
        assert config.integer_weight == 0.5
    
    def test_custom_weights(self):
        config = RewardConfig(
            correctness_weight=3.0,
            reasoning_quality_weight=0.5,
        )
        assert config.correctness_weight == 3.0
        assert config.reasoning_quality_weight == 0.5


class TestTrainingConfig:
    """Tests for TrainingConfig."""
    
    def test_default_values(self):
        config = TrainingConfig()
        assert config.learning_rate == 5e-6
        assert config.num_train_epochs == 1
        assert config.bf16 is True
    
    def test_custom_values(self):
        config = TrainingConfig(
            learning_rate=1e-5,
            num_train_epochs=3,
            output_dir="custom/output",
        )
        assert config.learning_rate == 1e-5
        assert config.num_train_epochs == 3
        assert config.output_dir == "custom/output"


class TestConfig:
    """Tests for main Config class."""
    
    def test_default_config(self):
        config = Config()
        assert config.model.name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert config.lora.enabled is True
        assert config.data.name == "gsm8k"
    
    def test_yaml_round_trip(self):
        config = Config()
        config.model.name = "test-model"
        config.training.learning_rate = 1e-4
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.to_yaml(f.name)
            loaded = Config.from_yaml(f.name)
        
        assert loaded.model.name == "test-model"
        assert loaded.training.learning_rate == 1e-4
    
    def test_merge_cli_args(self):
        config = Config()
        config.merge_cli_args(**{
            "model.name": "new-model",
            "training.learning_rate": 2e-5,
        })
        
        assert config.model.name == "new-model"
        assert config.training.learning_rate == 2e-5
