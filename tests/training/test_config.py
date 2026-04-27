import json
import pytest
from src.training.config import TrainingConfig


def test_default_config():
    config = TrainingConfig()
    assert config.model_name == "mistralai/Mistral-7B-v0.3"
    assert config.lora_r == 64
    assert config.lora_alpha == 128
    assert config.num_train_epochs == 3
    assert config.per_device_train_batch_size == 4


def test_config_from_json(tmp_path):
    data = {
        "model_name": "mistralai/Mistral-7B-v0.3",
        "lora_r": 32,
        "lora_alpha": 64,
        "num_train_epochs": 5,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    config = TrainingConfig.from_json(str(path))
    assert config.lora_r == 32
    assert config.num_train_epochs == 5
    assert config.per_device_train_batch_size == 4


def test_config_effective_batch_size():
    config = TrainingConfig(per_device_train_batch_size=4, gradient_accumulation_steps=4)
    assert config.effective_batch_size == 16
