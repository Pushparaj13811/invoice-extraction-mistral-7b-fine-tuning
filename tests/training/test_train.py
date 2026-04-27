import pytest
from src.training.config import TrainingConfig
from src.training.train import build_training_args, format_for_sft


def test_build_training_args():
    config = TrainingConfig(
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
    )
    args = build_training_args(config, output_dir="/tmp/test_output")
    assert args.num_train_epochs == 3
    assert args.per_device_train_batch_size == 4
    assert args.learning_rate == 2e-4
    assert args.output_dir == "/tmp/test_output"


def test_format_for_sft():
    example = {
        "instruction": "Extract fields.",
        "input": "Invoice text here",
        "output": '{"vendor_name": "Acme"}',
    }
    result = format_for_sft(example)
    assert "### Instruction:" in result
    assert "Invoice text here" in result
    assert '{"vendor_name": "Acme"}' in result
