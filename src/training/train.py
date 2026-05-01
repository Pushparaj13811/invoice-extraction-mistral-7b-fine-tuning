from __future__ import annotations

import os

import wandb
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from src.training.config import TrainingConfig
from src.training.lora_setup import load_model_and_tokenizer, build_lora_config
from src.data.format import load_jsonl


class _NoOpGradScaler:
    """Drop-in replacement for torch.amp.GradScaler that does nothing.

    When fine-tuning quantized models (QLoRA) with fp16=True, the real
    GradScaler crashes because bitsandbytes produces bfloat16 gradients
    from Mistral's native bfloat16 non-quantized layers. This scaler
    keeps fp16 autocast for speed (3-5x faster on T4/P100) while
    bypassing gradient scaling entirely.

    Implements the complete GradScaler interface as called by
    accelerate, transformers Trainer, and PyTorch internals.
    """

    def __init__(self):
        self._device = "cuda"

    def scale(self, outputs):
        return outputs

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer, closure=None):
        if closure is not None:
            optimizer.step(closure)
        else:
            optimizer.step()

    def update(self, new_scale=None):
        pass

    def state_dict(self):
        return {
            "scale": 1.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "_growth_tracker": 0,
        }

    def load_state_dict(self, state_dict):
        pass

    def is_enabled(self):
        return False

    def get_scale(self):
        return 1.0

    def get_growth_factor(self):
        return 2.0

    def set_growth_factor(self, new_factor):
        pass

    def get_backoff_factor(self):
        return 0.5

    def set_backoff_factor(self, new_factor):
        pass

    def get_growth_interval(self):
        return 2000

    def set_growth_interval(self, new_interval):
        pass

    def _lazy_init_scale_growth_tracker(self, device):
        pass


class QLoRASFTTrainer(SFTTrainer):
    """SFTTrainer with GradScaler replaced for QLoRA compatibility.

    Replaces the real GradScaler with a no-op after accelerator setup.
    This allows fp16 autocast (for speed) without gradient scaling
    (which crashes on bfloat16 gradients from quantized models).
    """

    def create_accelerator_and_postprocess(self):
        super().create_accelerator_and_postprocess()
        if hasattr(self.accelerator, "scaler") and self.accelerator.scaler is not None:
            self.accelerator.scaler = _NoOpGradScaler()


def format_for_sft(example: dict) -> str:
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


def build_training_args(config: TrainingConfig, output_dir: str) -> SFTConfig:
    """Build SFTConfig with fp16 enabled for speed.

    fp16=True enables autocast (fast) + GradScaler (crashes with QLoRA).
    We keep fp16=True for the autocast speed benefit and replace the
    GradScaler with a no-op in QLoRASFTTrainer.
    """
    import inspect
    sft_params = inspect.signature(SFTConfig.__init__).parameters

    kwargs = dict(
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
        gradient_checkpointing=True,
        fp16=True,
        report_to="wandb",
    )

    if "max_seq_length" in sft_params:
        kwargs["max_seq_length"] = config.max_seq_length
    elif "max_length" in sft_params:
        kwargs["max_length"] = config.max_seq_length

    return SFTConfig(**kwargs)


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

    train_texts = [format_for_sft(ex) for ex in train_data]
    eval_texts = [format_for_sft(ex) for ex in eval_data]

    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})

    model, tokenizer = load_model_and_tokenizer(config)
    peft_config = build_lora_config(config)

    training_args = build_training_args(config, output_dir)

    trainer = QLoRASFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(os.path.join(output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_adapter"))

    wandb.finish()

    return trainer
