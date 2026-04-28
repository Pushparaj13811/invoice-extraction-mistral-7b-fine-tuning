import json
import pytest
from src.evaluation.baseline import build_baseline_prompt, parse_baseline_response
from src.data.schema import Invoice


def test_build_baseline_prompt():
    prompt = build_baseline_prompt("Invoice #001 from Acme Corp...")
    assert "Invoice #001 from Acme Corp..." in prompt
    assert "vendor_name" in prompt
    assert "line_items" in prompt
    assert "JSON" in prompt


def test_parse_baseline_response_valid():
    response = json.dumps({
        "vendor_name": "Acme Corp",
        "invoice_number": "INV-001",
        "invoice_date": "2024-01-15",
        "due_date": "2024-02-15",
        "total_amount": 500.0,
        "currency": "USD",
        "line_items": [],
    })
    invoice = parse_baseline_response(response)
    assert isinstance(invoice, Invoice)
    assert invoice.vendor_name == "Acme Corp"


def test_parse_baseline_response_invalid():
    invoice = parse_baseline_response("not json at all")
    assert invoice is None


def test_parse_baseline_response_with_markdown_fences():
    response = '```json\n{"vendor_name": "Test", "invoice_number": "T-1", "invoice_date": "2024-01-01", "due_date": "2024-02-01", "total_amount": 100, "currency": "USD"}\n```'
    invoice = parse_baseline_response(response)
    assert isinstance(invoice, Invoice)
    assert invoice.vendor_name == "Test"
