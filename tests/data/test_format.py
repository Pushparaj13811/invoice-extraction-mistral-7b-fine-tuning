import json
import pytest
from src.data.schema import Invoice, LineItem
from src.data.format import format_example, format_dataset, save_jsonl, load_jsonl


def _make_pair() -> tuple[str, Invoice]:
    invoice = Invoice(
        vendor_name="Acme Corp",
        invoice_number="INV-001",
        invoice_date="2024-01-15",
        due_date="2024-02-15",
        total_amount=550.0,
        currency="USD",
        tax_amount=50.0,
        line_items=[
            LineItem(description="Widget", quantity=10, unit_price=50.0, line_total=500.0)
        ],
    )
    return ("Invoice #INV-001 from Acme Corp...", invoice)


def test_format_example_structure():
    text, invoice = _make_pair()
    result = format_example(text, invoice)
    assert "instruction" in result
    assert "input" in result
    assert "output" in result
    assert result["input"] == text
    parsed = json.loads(result["output"])
    assert parsed["vendor_name"] == "Acme Corp"


def test_format_dataset():
    pairs = [_make_pair(), _make_pair()]
    results = format_dataset(pairs)
    assert len(results) == 2
    assert all("instruction" in r for r in results)


def test_save_and_load_jsonl(tmp_path):
    pairs = [_make_pair()]
    formatted = format_dataset(pairs)
    path = tmp_path / "test.jsonl"
    save_jsonl(formatted, str(path))
    loaded = load_jsonl(str(path))
    assert len(loaded) == 1
    assert loaded[0]["input"] == "Invoice #INV-001 from Acme Corp..."
