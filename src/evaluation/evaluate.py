from __future__ import annotations

import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.schema import Invoice
from src.data.format import load_jsonl
from src.evaluation.metrics import compute_invoice_metrics
from src.training.train import format_for_sft


def load_finetuned_model(base_model: str, adapter_path: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def run_finetuned_inference(
    model,
    tokenizer,
    eval_data: list[dict],
    max_new_tokens: int = 1024,
) -> list[Invoice | None]:
    predictions = []

    for i, example in enumerate(eval_data):
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
            predictions.append(Invoice.model_validate(data))
        except Exception:
            predictions.append(None)

        if (i + 1) % 50 == 0:
            print(f"Inference progress: {i + 1}/{len(eval_data)}")

    return predictions


def aggregate_metrics(
    predictions: list[Invoice | None],
    golds: list[Invoice],
) -> dict:
    per_field_totals: dict[str, list[bool]] = {}
    line_item_scores = []
    overall_scores = []
    parse_successes = 0

    for pred, gold in zip(predictions, golds):
        if pred is None:
            overall_scores.append(0.0)
            continue

        parse_successes += 1
        result = compute_invoice_metrics(pred, gold)

        for field_name, matched in result["fields"].items():
            if field_name not in per_field_totals:
                per_field_totals[field_name] = []
            per_field_totals[field_name].append(matched)

        line_item_scores.append(result["line_item_score"])
        overall_scores.append(result["overall_accuracy"])

    per_field_accuracy = {
        name: sum(vals) / len(vals) for name, vals in per_field_totals.items()
    }

    return {
        "overall_accuracy": sum(overall_scores) / len(overall_scores) if overall_scores else 0.0,
        "json_parse_success_rate": parse_successes / len(predictions) if predictions else 0.0,
        "per_field": per_field_accuracy,
        "line_item_score": sum(line_item_scores) / len(line_item_scores) if line_item_scores else 0.0,
        "total_examples": len(predictions),
        "parse_failures": len(predictions) - parse_successes,
    }


def generate_report(ft_metrics: dict, baseline_metrics: dict) -> str:
    lines = [
        "# Invoice Extraction \u2014 Evaluation Report\n",
        "## Overall Results\n",
        "| Metric | Fine-Tuned Mistral 7B | GPT-4o-mini Baseline | Improvement |",
        "|--------|----------------------|---------------------|-------------|",
    ]

    ft_acc = ft_metrics["overall_accuracy"]
    bl_acc = baseline_metrics["overall_accuracy"]
    improvement = ((ft_acc - bl_acc) / bl_acc * 100) if bl_acc > 0 else 0
    lines.append(
        f"| Overall Accuracy | {ft_acc:.1%} | {bl_acc:.1%} | {improvement:+.1f}% |"
    )

    ft_parse = ft_metrics["json_parse_success_rate"]
    bl_parse = baseline_metrics["json_parse_success_rate"]
    lines.append(
        f"| JSON Parse Rate | {ft_parse:.1%} | {bl_parse:.1%} | \u2014 |"
    )

    ft_li = ft_metrics.get("line_item_score", 0)
    bl_li = baseline_metrics.get("line_item_score", 0)
    li_imp = ((ft_li - bl_li) / bl_li * 100) if bl_li > 0 else 0
    lines.append(
        f"| Line Item Score | {ft_li:.1%} | {bl_li:.1%} | {li_imp:+.1f}% |"
    )

    lines.append("\n## Per-Field Accuracy\n")
    lines.append("| Field | Fine-Tuned | GPT-4o-mini | Improvement |")
    lines.append("|-------|-----------|-------------|-------------|")

    all_fields = set(list(ft_metrics.get("per_field", {}).keys()) + list(baseline_metrics.get("per_field", {}).keys()))
    for field_name in sorted(all_fields):
        ft_val = ft_metrics.get("per_field", {}).get(field_name, 0)
        bl_val = baseline_metrics.get("per_field", {}).get(field_name, 0)
        imp = ((ft_val - bl_val) / bl_val * 100) if bl_val > 0 else 0
        lines.append(f"| {field_name} | {ft_val:.1%} | {bl_val:.1%} | {imp:+.1f}% |")

    return "\n".join(lines)
