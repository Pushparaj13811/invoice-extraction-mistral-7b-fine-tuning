from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.training.config import TrainingConfig


def build_bnb_config(config: TrainingConfig) -> BitsAndBytesConfig:
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_lora_config(config: TrainingConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )


def _cast_all_to_fp16(model):
    """Cast all non-quantized parameters and buffers from bfloat16 to float16.

    Mistral's config.json declares torch_dtype="bfloat16", which causes
    non-quantized layers (embeddings, norms, lm_head) to load in bfloat16.
    fp16 training uses GradScaler which crashes on bfloat16 gradients.
    This function ensures zero bfloat16 tensors remain in the model.
    """
    count = 0
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
            count += 1
    for name, buf in model.named_buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.to(torch.float16)
            count += 1
    print(f"Cast {count} bfloat16 tensors to float16")
    return model


def load_model_and_tokenizer(config: TrainingConfig):
    """Load quantized model and tokenizer, apply LoRA adapters."""
    bnb_config = build_bnb_config(config)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model = prepare_model_for_kbit_training(model)

    lora_config = build_lora_config(config)
    model = get_peft_model(model, lora_config)

    # Critical: cast ALL remaining bfloat16 params/buffers to float16
    # This prevents GradScaler crash with bfloat16 gradients
    model = _cast_all_to_fp16(model)

    model.print_trainable_parameters()

    return model, tokenizer
