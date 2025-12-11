"""GRPO Trainer - Advanced GRPO Training Framework for LLM Fine-tuning."""

__version__ = "2.0.0"
__author__ = "kossisoroyce"

from grpo_trainer.trainer import GRPOTrainerWrapper
from grpo_trainer.config import TrainingConfig, ModelConfig, DataConfig
from grpo_trainer.rewards import RewardManager

__all__ = [
    "GRPOTrainerWrapper",
    "TrainingConfig",
    "ModelConfig", 
    "DataConfig",
    "RewardManager",
]
