"""Model loading and configuration for GRPO training."""

import logging
import torch
from typing import Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": "auto",
}


def get_quantization_config(model_config) -> Optional[BitsAndBytesConfig]:
    """Create quantization config if needed."""
    if model_config.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DTYPE_MAP.get(model_config.bnb_4bit_compute_dtype, torch.bfloat16),
            bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
        )
    elif model_config.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def get_lora_config(lora_config) -> Optional[LoraConfig]:
    """Create LoRA config if enabled."""
    if not lora_config.enabled:
        return None
    
    return LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        task_type=lora_config.task_type,
        bias=lora_config.bias,
        modules_to_save=lora_config.modules_to_save,
    )


def load_model_and_tokenizer(config) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Optional[LoraConfig]]:
    """Load model and tokenizer with all configurations applied."""
    model_config = config.model
    lora_config = config.lora
    
    logger.info(f"Loading model: {model_config.name}")
    
    # Determine dtype
    torch_dtype = DTYPE_MAP.get(model_config.torch_dtype, torch.bfloat16)
    
    # Get quantization config
    quant_config = get_quantization_config(model_config)
    
    # Determine attention implementation
    attn_impl = model_config.attn_implementation
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn
            logger.info("Using Flash Attention 2")
        except ImportError:
            logger.warning("flash-attn not installed, falling back to eager attention")
            attn_impl = "eager"
    
    # Load model
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": model_config.device_map,
        "trust_remote_code": model_config.trust_remote_code,
        "use_cache": model_config.use_cache,
    }
    
    if attn_impl and attn_impl != "auto":
        model_kwargs["attn_implementation"] = attn_impl
    
    if quant_config:
        model_kwargs["quantization_config"] = quant_config
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.name,
            **model_kwargs
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Fallback without flash attention
        if "flash" in str(e).lower():
            logger.info("Retrying without flash attention...")
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(
                model_config.name,
                **model_kwargs
            )
        else:
            raise
    
    # Prepare for k-bit training if using quantization
    if quant_config:
        model = prepare_model_for_kbit_training(model)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.name,
        trust_remote_code=model_config.trust_remote_code,
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Get LoRA config
    peft_config = get_lora_config(lora_config)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded: {total_params:,} total params, {trainable_params:,} trainable")
    
    return model, tokenizer, peft_config


def load_model_for_inference(
    model_path: str,
    device: str = "auto",
    dtype: str = "bfloat16",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a trained model for inference."""
    logger.info(f"Loading model for inference from: {model_path}")
    
    torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def count_parameters(model: PreTrainedModel) -> dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_percent": 100 * trainable / total if total > 0 else 0,
    }


def get_model_memory_footprint(model: PreTrainedModel) -> dict:
    """Get model memory usage information."""
    try:
        mem = model.get_memory_footprint()
        return {
            "total_bytes": mem,
            "total_gb": mem / (1024 ** 3),
        }
    except Exception:
        return {"total_bytes": 0, "total_gb": 0}
