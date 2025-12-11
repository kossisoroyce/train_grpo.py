"""GRPO Trainer wrapper with enhanced functionality."""

import logging
import os
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from grpo_trainer.config import Config
from grpo_trainer.model import load_model_and_tokenizer, count_parameters
from grpo_trainer.datasets import load_dataset_for_training, load_eval_dataset
from grpo_trainer.rewards import RewardManager

logger = logging.getLogger(__name__)


class GRPOTrainerWrapper:
    """Enhanced GRPO Trainer with configuration management and utilities."""
    
    def __init__(
        self,
        config: Config,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        peft_config: Optional[LoraConfig] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.peft_config = peft_config
        self.trainer: Optional[GRPOTrainer] = None
        self.reward_manager: Optional[RewardManager] = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for training."""
        log_level = logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
            ]
        )
        
        # Create output directory
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logging.getLogger().addHandler(file_handler)
    
    def setup(self):
        """Set up model, tokenizer, and trainer."""
        # Load model and tokenizer if not provided
        if self.model is None or self.tokenizer is None:
            logger.info("Loading model and tokenizer...")
            self.model, self.tokenizer, self.peft_config = load_model_and_tokenizer(self.config)
        
        # Log model info
        param_info = count_parameters(self.model)
        logger.info(
            f"Model parameters: {param_info['total']:,} total, "
            f"{param_info['trainable']:,} trainable ({param_info['trainable_percent']:.2f}%)"
        )
        
        # Load dataset
        logger.info("Loading training dataset...")
        train_dataset = load_dataset_for_training(self.config)
        logger.info(f"Training dataset size: {len(train_dataset)}")
        
        # Load eval dataset if specified
        eval_dataset = load_eval_dataset(self.config)
        if eval_dataset:
            logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        
        # Set up reward functions
        logger.info("Setting up reward functions...")
        self.reward_manager = RewardManager(self.config)
        reward_funcs = self.reward_manager.get_reward_functions()
        logger.info(f"Registered {len(reward_funcs)} reward functions")
        
        # Create training arguments
        training_args = self._create_training_args()
        
        # Initialize trainer
        logger.info("Initializing GRPOTrainer...")
        trainer_kwargs = {
            "model": self.model,
            "processing_class": self.tokenizer,
            "reward_funcs": reward_funcs,
            "args": training_args,
            "train_dataset": train_dataset,
        }
        
        if eval_dataset:
            trainer_kwargs["eval_dataset"] = eval_dataset
        
        if self.peft_config and self.config.lora.enabled:
            trainer_kwargs["peft_config"] = self.peft_config
        
        self.trainer = GRPOTrainer(**trainer_kwargs)
        
        logger.info("Setup complete!")
        return self
    
    def _create_training_args(self) -> GRPOConfig:
        """Create GRPOConfig from our configuration."""
        tc = self.config.training
        
        args_dict = {
            "output_dir": tc.output_dir,
            "run_name": tc.run_name,
            "learning_rate": tc.learning_rate,
            "adam_beta1": tc.adam_beta1,
            "adam_beta2": tc.adam_beta2,
            "weight_decay": tc.weight_decay,
            "warmup_ratio": tc.warmup_ratio,
            "lr_scheduler_type": tc.lr_scheduler_type,
            "max_grad_norm": tc.max_grad_norm,
            "optim": tc.optim,
            "per_device_train_batch_size": tc.per_device_train_batch_size,
            "per_device_eval_batch_size": tc.per_device_eval_batch_size,
            "gradient_accumulation_steps": tc.gradient_accumulation_steps,
            "num_generations": tc.num_generations,
            "max_prompt_length": tc.max_prompt_length,
            "max_completion_length": tc.max_completion_length,
            "num_train_epochs": tc.num_train_epochs,
            "logging_steps": tc.logging_steps,
            "log_completions": tc.log_completions,
            "save_steps": tc.save_steps,
            "save_total_limit": tc.save_total_limit,
            "bf16": tc.bf16,
            "fp16": tc.fp16,
            "report_to": tc.report_to,
            "log_on_each_node": tc.log_on_each_node,
            "seed": tc.seed,
        }
        
        # GSPO / Advanced GRPO options
        if tc.loss_type != "grpo":
            args_dict["loss_type"] = tc.loss_type
        if tc.importance_sampling_level != "token":
            args_dict["importance_sampling_level"] = tc.importance_sampling_level
        if not tc.mask_truncated_completions:
            args_dict["mask_truncated_completions"] = tc.mask_truncated_completions
        
        if tc.max_steps > 0:
            args_dict["max_steps"] = tc.max_steps
        
        if tc.deepspeed:
            args_dict["deepspeed"] = tc.deepspeed
        
        if tc.fsdp:
            args_dict["fsdp"] = tc.fsdp
            if tc.fsdp_config:
                args_dict["fsdp_config"] = tc.fsdp_config
        
        return GRPOConfig(**args_dict)
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Run training."""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")
        
        checkpoint = resume_from_checkpoint or self.config.training.resume_from_checkpoint
        
        logger.info("Starting training...")
        try:
            if checkpoint:
                logger.info(f"Resuming from checkpoint: {checkpoint}")
                self.trainer.train(resume_from_checkpoint=checkpoint)
            else:
                self.trainer.train()
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        return self
    
    def save(self, output_dir: Optional[str] = None):
        """Save the trained model."""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized.")
        
        save_dir = output_dir or self.config.training.output_dir
        
        logger.info(f"Saving model to {save_dir}")
        self.trainer.save_model(save_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        # Save config
        config_path = Path(save_dir) / "training_config.yaml"
        self.config.to_yaml(config_path)
        
        logger.info("Model saved successfully!")
        return self
    
    def evaluate(self):
        """Run evaluation if eval dataset is available."""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized.")
        
        logger.info("Running evaluation...")
        metrics = self.trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics


def train_grpo(config: Config) -> GRPOTrainerWrapper:
    """Convenience function to train a model with GRPO."""
    trainer = GRPOTrainerWrapper(config)
    trainer.setup()
    trainer.train()
    trainer.save()
    return trainer
