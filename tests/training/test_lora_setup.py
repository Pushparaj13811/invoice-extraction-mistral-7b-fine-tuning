import pytest
from src.training.config import TrainingConfig
from src.training.lora_setup import build_bnb_config, build_lora_config


def test_build_bnb_config():
    config = TrainingConfig()
    bnb = build_bnb_config(config)
    assert bnb.load_in_4bit is True
    assert bnb.bnb_4bit_quant_type == "nf4"
    assert bnb.bnb_4bit_use_double_quant is True


def test_build_lora_config():
    config = TrainingConfig(lora_r=32, lora_alpha=64)
    lora = build_lora_config(config)
    assert lora.r == 32
    assert lora.lora_alpha == 64
    assert lora.lora_dropout == 0.05
    assert "q_proj" in lora.target_modules
    assert lora.task_type == "CAUSAL_LM"
