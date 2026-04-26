from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from openai import AzureOpenAI

from src.data.schema import Invoice


@dataclass
class SyntheticConfig:
    batch_size: int = 5
    currency: str = "USD"
    min_items: int = 1
    max_items: int = 5
    date_formats: list[str] = field(
        default_factory=lambda: ["YYYY-MM-DD", "MM/DD/YYYY", "DD-Mon-YYYY", "DD/MM/YYYY"]
    )


SCHEMA_DESCRIPTION = """
Each invoice must have these fields:
- vendor_name (string): company name
- invoice_number (string): unique ID
- invoice_date (string): when invoice was issued
- due_date (string): payment deadline
- total_amount (float): total including tax
- currency (string): e.g., USD, EUR, GBP
- tax_amount (float or null): tax portion
- discount (float or null): discount applied
- billing_address (string or null): customer address
- payment_terms (string or null): e.g., "Net 30"
- line_items (list): each with description, quantity, unit_price, line_total
"""


def build_generation_prompt(config: SyntheticConfig) -> str:
    return f"""Generate exactly {config.batch_size} realistic invoice examples.

For each invoice, provide:
1. "invoice_text" - the raw text of the invoice as it would appear in a document
2. "extracted" - the structured JSON extraction with all fields

{SCHEMA_DESCRIPTION}

Requirements:
- Use {config.currency} as the primary currency (mix in others for ~20% of invoices)
- Line items per invoice: between {config.min_items} and {config.max_items}
- Vary date formats across: {', '.join(config.date_formats)}
- Use diverse vendor names, industries, and invoice styles
- Include edge cases: partial fields, unusual formatting, multi-line addresses
- Make invoice_text look realistic (not just key: value pairs)

Return a JSON array of {config.batch_size} objects, each with "invoice_text" and "extracted" keys.
Return ONLY the JSON array, no other text."""


def parse_synthetic_response(response_text: str) -> list[tuple[str, Invoice]]:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try to parse; if truncated, attempt to salvage partial JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Response was likely truncated — try to find last complete object
        last_brace = text.rfind("}")
        if last_brace > 0:
            truncated = text[:last_brace + 1]
            # Close the array
            if not truncated.rstrip().endswith("]"):
                truncated = truncated.rstrip().rstrip(",") + "]"
            try:
                data = json.loads(truncated)
            except json.JSONDecodeError:
                return []
        else:
            return []
    results = []

    for entry in data:
        if not isinstance(entry, dict):
            continue
        invoice_text = entry.get("invoice_text", "")
        extracted = entry.get("extracted", {})
        if not invoice_text or not isinstance(extracted, dict):
            continue
        try:
            invoice = Invoice.model_validate(extracted)
            results.append((invoice_text, invoice))
        except Exception:
            continue

    return results


def generate_batch(
    config: SyntheticConfig,
    client: AzureOpenAI | None = None,
) -> list[tuple[str, Invoice]]:
    if client is None:
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    prompt = build_generation_prompt(config)
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=16384,
    )

    response_text = response.choices[0].message.content
    return parse_synthetic_response(response_text)


def generate_dataset(
    total: int = 1500,
    batch_size: int = 5,
    client: AzureOpenAI | None = None,
) -> list[tuple[str, Invoice]]:
    if client is None:
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )

    all_results = []
    currencies = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "INR", "NPR"]

    batch_num = 0
    while len(all_results) < total:
        currency = currencies[batch_num % len(currencies)]
        config = SyntheticConfig(
            batch_size=min(batch_size, total - len(all_results)),
            currency=currency,
            min_items=1 + (batch_num % 3),
            max_items=3 + (batch_num % 4),
        )
        try:
            batch = generate_batch(config, client=client)
            all_results.extend(batch)
            print(f"Batch {batch_num + 1}: generated {len(batch)} invoices (total: {len(all_results)}/{total})")
        except Exception as e:
            print(f"Batch {batch_num + 1}: failed ({e}), retrying...")
        batch_num += 1

    return all_results[:total]
