from __future__ import annotations

import json
import os

from openai import AzureOpenAI

from src.data.schema import Invoice


BASELINE_SYSTEM_PROMPT = """You are an invoice data extraction system. Given raw invoice text, extract all fields into a JSON object.

Required fields:
- vendor_name (string)
- invoice_number (string)
- invoice_date (string)
- due_date (string)
- total_amount (float)
- currency (string)

Optional fields (use null if not found):
- tax_amount (float or null)
- discount (float or null)
- billing_address (string or null)
- payment_terms (string or null)

line_items (list of objects, each with):
- description (string)
- quantity (number)
- unit_price (float)
- line_total (float)

Return ONLY valid JSON. No explanation, no markdown fences."""


def build_baseline_prompt(invoice_text: str) -> str:
    return f"""Extract all invoice fields from the following invoice text as JSON.

{BASELINE_SYSTEM_PROMPT}

Invoice text:
---
{invoice_text}
---

Return the extracted JSON:"""


def parse_baseline_response(response_text: str) -> Invoice | None:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
        return Invoice.model_validate(data)
    except Exception:
        return None


def run_baseline(
    eval_data: list[dict],
    client: AzureOpenAI | None = None,
) -> list[Invoice | None]:
    if client is None:
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    predictions = []

    for i, example in enumerate(eval_data):
        invoice_text = example["input"]
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract all invoice fields from this text:\n\n{invoice_text}"},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            result = parse_baseline_response(response.choices[0].message.content)
            predictions.append(result)
        except Exception as e:
            print(f"Baseline error on example {i}: {e}")
            predictions.append(None)

        if (i + 1) % 50 == 0:
            print(f"Baseline progress: {i + 1}/{len(eval_data)}")

    return predictions
