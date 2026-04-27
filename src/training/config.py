from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    model_name: str = "mistralai/Mistral-7B-v0.3"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 50
    max_seq_length: int = 2048
    optim: str = "paged_adamw_8bit"
    save_steps: int = 200
    logging_steps: int = 10
    eval_steps: int = 100
    wandb_project: str = "invoice-extraction-finetune"

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_json(cls, path: str) -> TrainingConfig:
        with open(path) as f:
            data = json.load(f)
        flat = {}
        for key, value in data.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in flat.items() if k in valid_keys}
        return cls(**filtered)
