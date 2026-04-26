from __future__ import annotations

import ast
import json
from typing import Any

from datasets import load_dataset

from src.data.schema import Invoice, LineItem


def _safe_parse(text: str) -> dict | None:
    """Parse a Python dict string (single quotes) or JSON string."""
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def _safe_float(value) -> float | None:
    """Safely convert a value to float."""
    if value is None:
        return None
    try:
        # Handle strings like "1,234.56"
        if isinstance(value, str):
            value = value.replace(",", "").replace(" ", "")
        return float(value)
    except (ValueError, TypeError):
        return None


def parse_ocr_text(raw_data_str: str) -> str | None:
    """Extract OCR text from the raw_data column.

    raw_data contains a JSON/Python dict with 'ocr_words' key
    which is a string representation of a list of words.
    """
    raw_data = _safe_parse(raw_data_str)
    if not raw_data or not isinstance(raw_data, dict):
        return None

    ocr_words_str = raw_data.get("ocr_words", "")
    if not ocr_words_str:
        return None

    words = _safe_parse(ocr_words_str)
    if not words or not isinstance(words, list):
        return None

    return " ".join(str(w) for w in words if w)


def parse_invoice_labels(parsed_data_str: str) -> Invoice | None:
    """Extract structured invoice labels from the parsed_data column.

    parsed_data contains a JSON/Python dict with a 'json' key
    whose value is a string containing the structured invoice data
    with header, items, and summary sections.
    """
    parsed_data = _safe_parse(parsed_data_str)
    if not parsed_data or not isinstance(parsed_data, dict):
        return None

    invoice_str = parsed_data.get("json", "")
    if not invoice_str:
        return None

    invoice_data = _safe_parse(invoice_str)
    if not invoice_data or not isinstance(invoice_data, dict):
        return None

    header = invoice_data.get("header", {})
    items_list = invoice_data.get("items", [])
    summary = invoice_data.get("summary", {})

    # Extract header fields
    vendor_name = header.get("seller", "")
    invoice_number = header.get("invoice_no", "")
    invoice_date = header.get("invoice_date", "")

    if not vendor_name or not invoice_number or not invoice_date:
        return None

    # Extract total from summary
    total_amount = _safe_float(summary.get("total_gross_worth"))
    if total_amount is None:
        return None

    tax_amount = _safe_float(summary.get("total_vat"))

    # Extract line items
    line_items = []
    for item in items_list:
        if not isinstance(item, dict):
            continue
        desc = item.get("item_desc", "")
        qty = _safe_float(item.get("item_qty"))
        price = _safe_float(item.get("item_net_price"))
        total = _safe_float(item.get("item_gross_worth"))

        if desc and qty is not None and price is not None and total is not None:
            line_items.append(LineItem(
                description=desc,
                quantity=qty,
                unit_price=price,
                line_total=total,
            ))

    try:
        return Invoice(
            vendor_name=vendor_name,
            invoice_number=invoice_number,
            invoice_date=invoice_date,
            due_date=header.get("due_date", ""),
            total_amount=total_amount,
            currency="PLN",  # This dataset is Polish invoices, currency is PLN
            tax_amount=tax_amount,
            billing_address=header.get("client", None),
            line_items=line_items,
        )
    except Exception:
        return None


def load_existing_dataset(
    dataset_name: str = "mychen76/invoices-and-receipts_ocr_v1",
    split: str = "train",
    max_samples: int = 500,
) -> list[tuple[str, Invoice]]:
    """Load dataset and return (ocr_text, invoice) pairs.

    Uses the OCR text from raw_data as the input text and
    parsed_data as the structured ground truth labels.
    """
    ds = load_dataset(dataset_name, split=split)
    # Remove image column to avoid Pillow dependency during iteration
    if "image" in ds.column_names:
        ds = ds.remove_columns(["image"])

    results = []
    for record in ds:
        ocr_text = parse_ocr_text(record.get("raw_data", ""))
        invoice = parse_invoice_labels(record.get("parsed_data", ""))

        if ocr_text and invoice:
            results.append((ocr_text, invoice))
            if len(results) >= max_samples:
                break

    return results
