from __future__ import annotations

import os

import wandb
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from src.training.config import TrainingConfig
from src.training.lora_setup import load_model_and_tokenizer
from src.data.format import load_jsonl

# Check if newer trl with SFTConfig is available
try:
    from trl import SFTConfig
    _HAS_SFT_CONFIG = True
except ImportError:
    _HAS_SFT_CONFIG = False


def format_for_sft(example: dict) -> str:
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


def build_training_args(config: TrainingConfig, output_dir: str) -> TrainingArguments:
    base_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        optim=config.optim,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
    )

    if _HAS_SFT_CONFIG:
        import inspect
        sft_params = inspect.signature(SFTConfig.__init__).parameters
        if "max_seq_length" in sft_params:
            return SFTConfig(**base_kwargs, max_seq_length=config.max_seq_length)
        elif "max_length" in sft_params:
            return SFTConfig(**base_kwargs, max_length=config.max_seq_length)
        return SFTConfig(**base_kwargs)
    return TrainingArguments(**base_kwargs)


def train(
    config: TrainingConfig,
    train_path: str,
    eval_path: str,
    output_dir: str = "outputs/",
):
    wandb.init(
        project=config.wandb_project,
        config={
            "model": config.model_name,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "epochs": config.num_train_epochs,
            "batch_size": config.effective_batch_size,
            "learning_rate": config.learning_rate,
        },
    )

    train_data = load_jsonl(train_path)
    eval_data = load_jsonl(eval_path)

    # Pre-format into text strings for SFTTrainer
    train_texts = [format_for_sft(ex) for ex in train_data]
    eval_texts = [format_for_sft(ex) for ex in eval_data]

    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})

    model, tokenizer = load_model_and_tokenizer(config)

    training_args = build_training_args(config, output_dir)

    # Build SFTTrainer with version-appropriate args
    trainer_kwargs = dict(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    if _HAS_SFT_CONFIG:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
        trainer_kwargs["max_seq_length"] = config.max_seq_length

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()

    # Save adapter
    trainer.save_model(os.path.join(output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_adapter"))

    wandb.finish()

    return trainer
