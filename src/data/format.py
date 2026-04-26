from __future__ import annotations

import json

from src.data.schema import Invoice

INSTRUCTION = "Extract all invoice fields from the following invoice text as JSON."


def format_example(invoice_text: str, invoice: Invoice) -> dict:
    return {
        "instruction": INSTRUCTION,
        "input": invoice_text,
        "output": invoice.model_dump_json(indent=2),
    }


def format_dataset(data: list[tuple[str, Invoice]]) -> list[dict]:
    return [format_example(text, inv) for text, inv in data]


def save_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
