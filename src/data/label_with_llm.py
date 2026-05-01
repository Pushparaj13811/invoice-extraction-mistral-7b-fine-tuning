"""Label unlabeled OCR invoice texts using Azure GPT-4o-mini.

Reads the HF dataset, finds records with OCR text but empty structured labels,
and uses GPT-4o-mini to extract structured fields from the OCR text.
"""
from __future__ import annotations

import ast
import json
import os
import time

from openai import AzureOpenAI
from dotenv import load_dotenv

from src.data.schema import Invoice, LineItem

load_dotenv()

EXTRACTION_PROMPT = """Extract all invoice fields from the following OCR text into a JSON object.

Required fields:
- vendor_name (string): the seller/company name only, not their address
- invoice_number (string): the invoice ID/number
- invoice_date (string): format as YYYY-MM-DD
- due_date (string): format as YYYY-MM-DD, use "" if not found
- total_amount (float): the total/gross amount
- currency (string): USD, EUR, GBP, etc. Use "USD" if $ symbol or unclear

Optional fields (use null if not found):
- tax_amount (float or null)
- discount (float or null)
- billing_address (string or null): the client/buyer name and address
- payment_terms (string or null)

Line items (list of objects, each with):
- description (string)
- quantity (float)
- unit_price (float): the net/unit price
- line_total (float): the gross/total for this line

IMPORTANT:
- Numbers may use commas as decimal separators (European format): "7,50" means 7.50
- Remove currency symbols ($, €) from numbers
- Return ONLY valid JSON, no explanation"""


def get_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )


def extract_fields(ocr_text: str, client: AzureOpenAI) -> Invoice | None:
    """Use GPT-4o-mini to extract structured fields from OCR text."""
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": ocr_text},
            ],
            temperature=0.0,
            max_tokens=2048,
        )

        text = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        data = json.loads(text)
        return Invoice.model_validate(data)

    except Exception:
        return None


def load_unlabeled_ocr_texts(
    dataset_name: str = "mychen76/invoices-and-receipts_ocr_v1",
) -> list[str]:
    """Load OCR texts from records that have empty structured labels."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    if "image" in ds.column_names:
        ds = ds.remove_columns(["image"])

    unlabeled = []
    for record in ds:
        # Check if structured labels are empty
        parsed = record.get("parsed_data", "")
        try:
            pd = ast.literal_eval(parsed)
        except Exception:
            try:
                pd = json.loads(parsed)
            except Exception:
                continue

        json_str = pd.get("json", "")
        if not json_str:
            continue

        try:
            inv = ast.literal_eval(json_str)
        except Exception:
            try:
                inv = json.loads(json_str)
            except Exception:
                continue

        header = inv.get("header", {})

        # If header is empty, this record needs labeling
        if not header.get("seller") and not header.get("invoice_no"):
            # Get OCR text
            raw = record.get("raw_data", "")
            try:
                rd = ast.literal_eval(raw)
                words = ast.literal_eval(rd.get("ocr_words", ""))
                if words and len(words) > 10:
                    ocr_text = " ".join(str(w) for w in words if w)
                    unlabeled.append(ocr_text)
            except Exception:
                continue

    return unlabeled


def label_dataset(
    output_path: str = "data/llm_labeled.jsonl",
    batch_size: int = 50,
    max_samples: int | None = None,
) -> list[tuple[str, Invoice]]:
    """Label all unlabeled OCR texts and save to JSONL.

    Saves progress incrementally — safe to interrupt and resume.
    """
    client = get_client()
    ocr_texts = load_unlabeled_ocr_texts()

    if max_samples:
        ocr_texts = ocr_texts[:max_samples]

    print(f"Found {len(ocr_texts)} unlabeled OCR texts to label")

    # Check for existing progress
    existing = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                existing.add(rec["input"][:100])
        print(f"Resuming: {len(existing)} already labeled")

    results = []
    labeled = 0
    failed = 0

    with open(output_path, "a") as f:
        for i, ocr_text in enumerate(ocr_texts):
            # Skip already labeled
            if ocr_text[:100] in existing:
                continue

            invoice = extract_fields(ocr_text, client)

            if invoice:
                record = {
                    "instruction": "Extract all invoice fields from the following invoice text as JSON.",
                    "input": ocr_text,
                    "output": invoice.model_dump_json(indent=2),
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                results.append((ocr_text, invoice))
                labeled += 1
            else:
                failed += 1

            if (i + 1) % batch_size == 0:
                print(f"Progress: {i + 1}/{len(ocr_texts)} | Labeled: {labeled} | Failed: {failed}")

            # Rate limiting — avoid hitting API limits
            time.sleep(0.2)

    print(f"\nDone: {labeled} labeled, {failed} failed out of {len(ocr_texts)} total")
    print(f"Saved to {output_path}")

    return results


if __name__ == "__main__":
    label_dataset()
