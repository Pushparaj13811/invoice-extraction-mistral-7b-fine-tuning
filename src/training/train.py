from __future__ import annotations

import os

import wandb
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from src.training.config import TrainingConfig
from src.training.lora_setup import load_model_and_tokenizer
from src.data.format import load_jsonl


def format_for_sft(example: dict) -> str:
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


def build_training_args(config: TrainingConfig, output_dir: str) -> TrainingArguments:
    return TrainingArguments(
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
        fp16=True,
        gradient_checkpointing=True,
    )


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

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    model, tokenizer = load_model_and_tokenizer(config)

    training_args = build_training_args(config, output_dir)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=lambda x: [format_for_sft(x)],
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()

    trainer.save_model(os.path.join(output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_adapter"))

    wandb.finish()

    return trainer
