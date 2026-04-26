import json
import pytest
from src.data.schema import Invoice
from src.data.existing_loader import parse_ocr_text, parse_invoice_labels


def test_parse_ocr_text_valid():
    raw_data = json.dumps({
        "ocr_words": "['Invoice', 'No', '001', 'Date', '2024-01-15']",
        "ocr_boxes": "",
        "ocr_labels": "",
    })
    text = parse_ocr_text(raw_data)
    assert text is not None
    assert "Invoice" in text
    assert "001" in text


def test_parse_ocr_text_empty():
    raw_data = json.dumps({"ocr_words": "", "ocr_boxes": "", "ocr_labels": ""})
    assert parse_ocr_text(raw_data) is None


def test_parse_ocr_text_invalid():
    assert parse_ocr_text("not valid") is None
    assert parse_ocr_text("") is None


def test_parse_invoice_labels_valid():
    invoice_json = str({
        "header": {
            "invoice_no": "INV-001",
            "invoice_date": "2024-01-15",
            "seller": "Acme Corp",
            "client": "Test Client LLC",
            "seller_tax_id": "123456",
            "client_tax_id": "789012",
            "iban": "PL123",
        },
        "items": [
            {
                "item_desc": "Widget A",
                "item_qty": "10",
                "item_net_price": "50.00",
                "item_net_worth": "500.00",
                "item_vat": "115.00",
                "item_gross_worth": "615.00",
            }
        ],
        "summary": {
            "total_net_worth": "500.00",
            "total_vat": "115.00",
            "total_gross_worth": "615.00",
        },
    })
    parsed_data = json.dumps({"json": invoice_json, "xml": "", "kie": ""})

    invoice = parse_invoice_labels(parsed_data)
    assert isinstance(invoice, Invoice)
    assert invoice.vendor_name == "Acme Corp"
    assert invoice.invoice_number == "INV-001"
    assert invoice.total_amount == 615.0
    assert invoice.tax_amount == 115.0
    assert invoice.billing_address == "Test Client LLC"
    assert len(invoice.line_items) == 1
    assert invoice.line_items[0].description == "Widget A"
    assert invoice.line_items[0].quantity == 10.0
    assert invoice.line_items[0].line_total == 615.0


def test_parse_invoice_labels_missing_required():
    invoice_json = str({
        "header": {"seller": "", "invoice_no": "", "invoice_date": ""},
        "items": [],
        "summary": {"total_gross_worth": "100"},
    })
    parsed_data = json.dumps({"json": invoice_json, "xml": "", "kie": ""})
    assert parse_invoice_labels(parsed_data) is None


def test_parse_invoice_labels_invalid():
    assert parse_invoice_labels("garbage") is None
    assert parse_invoice_labels("") is None


def test_parse_invoice_labels_no_json_key():
    parsed_data = json.dumps({"xml": "", "kie": ""})
    assert parse_invoice_labels(parsed_data) is None
