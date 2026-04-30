from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

from src.training.config import TrainingConfig


def build_bnb_config(config: TrainingConfig) -> BitsAndBytesConfig:
    """Build 4-bit quantization config.

    bnb_4bit_compute_dtype controls the precision used during forward pass
    for quantized layers. float16 is optimal for T4 GPUs.
    """
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_lora_config(config: TrainingConfig) -> LoraConfig:
    """Build LoRA adapter config."""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )


def load_model_and_tokenizer(config: TrainingConfig):
    """Load 4-bit quantized model and tokenizer.

    Note: LoRA adapters are NOT applied here. When using SFTTrainer,
    pass peft_config directly to the trainer — it handles PEFT wrapping
    internally, which is the recommended approach.
    """
    bnb_config = build_bnb_config(config)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
