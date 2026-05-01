from __future__ import annotations

import ast
import json
import re
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
    """Safely convert a value to float.

    Handles European number formats (comma as decimal separator),
    currency symbols ($, €), and whitespace.
    Examples: "$7,50" -> 7.50, "1.234,56" -> 1234.56, "8,25" -> 8.25
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    # Remove currency symbols, whitespace, percentage signs
    cleaned = re.sub(r'[$€£¥%\s]', '', value.strip())
    if not cleaned:
        return None

    # Detect European format: "1.234,56" or "7,50"
    # European: dots are thousands separators, comma is decimal
    if ',' in cleaned and '.' in cleaned:
        # "1.234,56" -> "1234.56"
        cleaned = cleaned.replace('.', '').replace(',', '.')
    elif ',' in cleaned:
        # Could be "7,50" (European decimal) or "1,234" (US thousands)
        # If digits after comma <= 2, treat as decimal
        parts = cleaned.split(',')
        if len(parts) == 2 and len(parts[1]) <= 2:
            cleaned = cleaned.replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')

    try:
        return float(cleaned)
    except ValueError:
        return None


def _split_name_address(text: str) -> tuple[str, str]:
    """Split a 'Name Address' string into (name, address).

    The dataset stores seller/client as one string like:
    'Patel, Thompson and Montgomery 356 Kyle Vista New James, MA 46228'

    We split at the first number that starts an address.
    """
    if not text:
        return ("", "")

    # Find where the address starts (first digit sequence that looks like a street number)
    match = re.search(r'\s(\d+\s+[A-Z])', text)
    if match:
        name = text[:match.start()].strip()
        address = text[match.start():].strip()
        return (name, address)

    # Try splitting at common address patterns
    match = re.search(r'\s(USS?N?V?\s|FPO\s|APO\s|\d+\s)', text)
    if match:
        name = text[:match.start()].strip()
        address = text[match.start():].strip()
        return (name, address)

    return (text.strip(), "")


def parse_ocr_text(raw_data_str: str) -> str | None:
    """Extract OCR text from the raw_data column."""
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
    """Extract structured invoice labels from the parsed_data column."""
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

    # Extract and split seller name from address
    seller_raw = header.get("seller", "")
    vendor_name, _ = _split_name_address(seller_raw)

    invoice_number = header.get("invoice_no", "")
    invoice_date = header.get("invoice_date", "")

    if not vendor_name or not invoice_number or not invoice_date:
        return None

    # Extract totals — handles "$7,50" and "$ 63,69" formats
    total_amount = _safe_float(summary.get("total_gross_worth"))
    if total_amount is None:
        return None

    tax_amount = _safe_float(summary.get("total_vat"))

    # Client info for billing address
    client_raw = header.get("client", "")

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
            currency="USD",
            tax_amount=tax_amount,
            billing_address=client_raw if client_raw else None,
            line_items=line_items,
        )
    except Exception:
        return None


def load_existing_dataset(
    dataset_name: str = "mychen76/invoices-and-receipts_ocr_v1",
    split: str = "train",
    max_samples: int = 2000,
) -> list[tuple[str, Invoice]]:
    """Load dataset and return (ocr_text, invoice) pairs."""
    ds = load_dataset(dataset_name, split=split)
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
