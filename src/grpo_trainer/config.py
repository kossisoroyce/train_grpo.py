"""Configuration classes for GRPO training."""

from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
import yaml
from omegaconf import OmegaConf, DictConfig


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"
    device_map: str = "auto"
    trust_remote_code: bool = True
    use_cache: bool = False
    
    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration for PEFT."""
    enabled: bool = True
    r: int = 16
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "up_proj", "down_proj", "gate_proj"
    ])
    task_type: str = "CAUSAL_LM"
    bias: str = "none"
    modules_to_save: Optional[list[str]] = None


@dataclass
class DataConfig:
    """Dataset configuration."""
    name: str = "gsm8k"
    subset: Optional[str] = "main"
    split: str = "train"
    eval_split: Optional[str] = "test"
    
    # Prompting
    use_one_shot: bool = True
    use_few_shot: bool = False
    num_few_shot_examples: int = 3
    
    # Processing
    max_samples: Optional[int] = None
    shuffle: bool = True
    seed: int = 42
    
    # Custom dataset path (for local datasets)
    custom_path: Optional[str] = None


@dataclass
class RewardConfig:
    """Reward function configuration."""
    # Reward weights
    correctness_weight: float = 2.0
    format_weight: float = 0.5
    integer_weight: float = 0.5
    xml_count_weight: float = 0.5
    
    # Additional rewards
    length_penalty_weight: float = 0.0
    max_length_for_penalty: int = 1024
    
    reasoning_quality_weight: float = 0.0
    
    # VLM gibberish penalty (for vision models)
    gibberish_penalty_weight: float = 0.0
    gibberish_threshold: float = 0.5
    
    # Custom delimiters (alternative to XML format)
    use_custom_delimiters: bool = False
    reasoning_start: str = "<REASONING>"
    reasoning_end: str = "</REASONING>"
    answer_start: str = "<SOLUTION>"
    answer_end: str = "</SOLUTION>"
    
    # Strict format checking
    strict_format: bool = False
    
    # Custom reward functions (module paths)
    custom_rewards: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Output
    output_dir: str = "outputs/grpo-run"
    run_name: str = "grpo-training"
    
    # Training hyperparameters
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.1
    optim: str = "adamw_torch"  # Use "adamw_8bit" for memory efficiency
    
    # Batch sizes
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    
    # GRPO specific
    num_generations: int = 8
    max_prompt_length: int = 256
    max_completion_length: int = 786
    temperature: float = 0.7
    top_p: float = 0.9
    
    # GSPO / Advanced GRPO options
    loss_type: str = "grpo"  # "grpo", "dr_grpo" (GSPO), "ipo", "simpo"
    importance_sampling_level: str = "token"  # "token" or "sequence"
    mask_truncated_completions: bool = True  # False for GSPO
    
    # Training duration
    num_train_epochs: int = 1
    max_steps: int = -1
    
    # Logging & saving
    logging_steps: int = 1
    log_completions: bool = False  # Log generated completions
    save_steps: int = 100
    save_total_limit: int = 3
    eval_steps: int = 100
    eval_strategy: str = "steps"
    
    # Precision
    bf16: bool = True
    fp16: bool = False
    
    # Distributed training
    deepspeed: Optional[str] = None
    fsdp: Optional[str] = None
    fsdp_config: Optional[str] = None
    
    # Reporting
    report_to: str = "wandb"
    log_on_each_node: bool = False
    
    # Resume
    resume_from_checkpoint: Optional[str] = None
    
    # Seed
    seed: int = 42


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            raw_config = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**raw_config.get("model", {})),
            lora=LoRAConfig(**raw_config.get("lora", {})),
            data=DataConfig(**raw_config.get("data", {})),
            reward=RewardConfig(**raw_config.get("reward", {})),
            training=TrainingConfig(**raw_config.get("training", {})),
        )
    
    @classmethod
    def from_omega(cls, cfg: DictConfig) -> "Config":
        """Load configuration from OmegaConf DictConfig."""
        return cls(
            model=ModelConfig(**OmegaConf.to_container(cfg.get("model", {}))),
            lora=LoRAConfig(**OmegaConf.to_container(cfg.get("lora", {}))),
            data=DataConfig(**OmegaConf.to_container(cfg.get("data", {}))),
            reward=RewardConfig(**OmegaConf.to_container(cfg.get("reward", {}))),
            training=TrainingConfig(**OmegaConf.to_container(cfg.get("training", {}))),
        )
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "model": self.model.__dict__,
            "lora": self.lora.__dict__,
            "data": self.data.__dict__,
            "reward": self.reward.__dict__,
            "training": self.training.__dict__,
        }
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def merge_cli_args(self, **kwargs) -> "Config":
        """Merge CLI arguments into configuration."""
        for key, value in kwargs.items():
            if value is None:
                continue
            
            # Parse dotted keys like "model.name" or "training.learning_rate"
            parts = key.split(".")
            if len(parts) == 2:
                section, attr = parts
                if hasattr(self, section):
                    section_obj = getattr(self, section)
                    if hasattr(section_obj, attr):
                        setattr(section_obj, attr, value)
            elif hasattr(self.training, key):
                setattr(self.training, key, value)
        
        return self
