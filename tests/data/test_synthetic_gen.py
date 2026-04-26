import json
import pytest
from src.data.schema import Invoice
from src.data.synthetic_gen import (
    build_generation_prompt,
    parse_synthetic_response,
    SyntheticConfig,
)


def test_build_generation_prompt_contains_schema():
    config = SyntheticConfig(batch_size=5, currency="EUR", min_items=2, max_items=5)
    prompt = build_generation_prompt(config)
    assert "vendor_name" in prompt
    assert "invoice_number" in prompt
    assert "line_items" in prompt
    assert "EUR" in prompt
    assert "5" in prompt


def test_parse_synthetic_response_valid():
    sample = {
        "invoice_text": "Invoice #INV-001\nFrom: Acme Corp\nDate: 2024-01-15\nDue: 2024-02-15\nWidget x10 @ $5.00 = $50.00\nTotal: $55.00\nTax: $5.00\nCurrency: USD",
        "extracted": {
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-001",
            "invoice_date": "2024-01-15",
            "due_date": "2024-02-15",
            "total_amount": 55.0,
            "currency": "USD",
            "tax_amount": 5.0,
            "discount": 0.0,
            "billing_address": None,
            "payment_terms": None,
            "line_items": [
                {"description": "Widget", "quantity": 10, "unit_price": 5.0, "line_total": 50.0}
            ],
        },
    }
    response_text = json.dumps([sample])
    results = parse_synthetic_response(response_text)
    assert len(results) == 1
    text, invoice = results[0]
    assert isinstance(invoice, Invoice)
    assert "Acme Corp" in text
    assert invoice.vendor_name == "Acme Corp"


def test_parse_synthetic_response_skips_invalid():
    data = [
        {"invoice_text": "Valid", "extracted": {"vendor_name": "A", "invoice_number": "1", "invoice_date": "2024-01-01", "due_date": "2024-02-01", "total_amount": 100, "currency": "USD"}},
        {"invoice_text": "Bad", "extracted": {"garbage": True}},
    ]
    results = parse_synthetic_response(json.dumps(data))
    assert len(results) == 1
