from __future__ import annotations

from src.training.config import TrainingConfig

# Try Unsloth first (2x less VRAM), fall back to standard HF
try:
    from unsloth import FastLanguageModel
    _HAS_UNSLOTH = True
except ImportError:
    _HAS_UNSLOTH = False

if not _HAS_UNSLOTH:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def build_bnb_config(config: TrainingConfig):
    """Build BitsAndBytesConfig (only used in non-Unsloth path)."""
    if _HAS_UNSLOTH:
        return None
    import torch
    from transformers import BitsAndBytesConfig
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_lora_config(config: TrainingConfig):
    """Build LoraConfig (only used in non-Unsloth path)."""
    if _HAS_UNSLOTH:
        return None
    from peft import LoraConfig
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )


def load_model_and_tokenizer(config: TrainingConfig):
    """Load model and tokenizer with LoRA adapters.

    Uses Unsloth if available (2x less VRAM), otherwise standard HF + PEFT.
    """
    if _HAS_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            dtype=None,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        model.print_trainable_parameters()
        return model, tokenizer

    # Standard HF path
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

    model = prepare_model_for_kbit_training(model)

    lora_config = build_lora_config(config)
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, tokenizer
