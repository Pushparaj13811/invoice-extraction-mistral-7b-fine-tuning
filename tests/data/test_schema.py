import json
import pytest
from src.data.schema import LineItem, Invoice


def test_line_item_valid():
    item = LineItem(
        description="Widget A",
        quantity=10,
        unit_price=5.99,
        line_total=59.90,
    )
    assert item.description == "Widget A"
    assert item.quantity == 10
    assert item.unit_price == 5.99
    assert item.line_total == 59.90


def test_invoice_full():
    invoice = Invoice(
        vendor_name="Acme Corp",
        invoice_number="INV-2024-001",
        invoice_date="2024-01-15",
        due_date="2024-02-15",
        total_amount=1299.99,
        currency="USD",
        tax_amount=99.99,
        discount=0.0,
        billing_address="123 Main St, Springfield, IL 62701",
        payment_terms="Net 30",
        line_items=[
            LineItem(
                description="Widget A",
                quantity=10,
                unit_price=120.00,
                line_total=1200.00,
            )
        ],
    )
    assert invoice.vendor_name == "Acme Corp"
    assert invoice.total_amount == 1299.99
    assert len(invoice.line_items) == 1


def test_invoice_minimal():
    """Invoice with only required fields."""
    invoice = Invoice(
        vendor_name="Test Vendor",
        invoice_number="001",
        invoice_date="2024-01-01",
        due_date="2024-02-01",
        total_amount=100.0,
        currency="USD",
    )
    assert invoice.tax_amount is None
    assert invoice.line_items == []


def test_invoice_to_json_roundtrip():
    invoice = Invoice(
        vendor_name="Acme Corp",
        invoice_number="INV-001",
        invoice_date="2024-01-15",
        due_date="2024-02-15",
        total_amount=500.0,
        currency="EUR",
        tax_amount=50.0,
        discount=10.0,
        billing_address="456 Oak Ave",
        payment_terms="Net 60",
        line_items=[
            LineItem(
                description="Service Fee",
                quantity=1,
                unit_price=500.0,
                line_total=500.0,
            )
        ],
    )
    json_str = invoice.model_dump_json()
    parsed = Invoice.model_validate_json(json_str)
    assert parsed == invoice


def test_invoice_from_dict():
    data = {
        "vendor_name": "Test",
        "invoice_number": "T-001",
        "invoice_date": "2024-03-01",
        "due_date": "2024-04-01",
        "total_amount": 250.0,
        "currency": "GBP",
        "line_items": [
            {
                "description": "Item 1",
                "quantity": 5,
                "unit_price": 50.0,
                "line_total": 250.0,
            }
        ],
    }
    invoice = Invoice.model_validate(data)
    assert invoice.currency == "GBP"
    assert invoice.line_items[0].quantity == 5
