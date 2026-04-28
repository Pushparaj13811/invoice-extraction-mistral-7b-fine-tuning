import pytest
from src.data.schema import Invoice, LineItem
from src.evaluation.metrics import (
    exact_match,
    fuzzy_match,
    compute_field_metrics,
    compute_line_item_metrics,
    compute_invoice_metrics,
)


def test_exact_match_true():
    assert exact_match("INV-001", "INV-001") is True


def test_exact_match_false():
    assert exact_match("INV-001", "INV-002") is False


def test_exact_match_numeric():
    assert exact_match(500.0, 500.0) is True
    assert exact_match(500.0, 500.01) is False


def test_fuzzy_match_exact():
    assert fuzzy_match("Acme Corp", "Acme Corp") is True


def test_fuzzy_match_close():
    assert fuzzy_match("Acme Corp", "Acme Corp.", threshold=0.85) is True


def test_fuzzy_match_different():
    assert fuzzy_match("Acme Corp", "Beta Industries", threshold=0.85) is False


def test_compute_field_metrics():
    pred = Invoice(
        vendor_name="Acme Corp.",
        invoice_number="INV-001",
        invoice_date="2024-01-15",
        due_date="2024-02-15",
        total_amount=500.0,
        currency="USD",
    )
    gold = Invoice(
        vendor_name="Acme Corp",
        invoice_number="INV-001",
        invoice_date="2024-01-15",
        due_date="2024-02-15",
        total_amount=500.0,
        currency="USD",
    )
    metrics = compute_field_metrics(pred, gold)
    assert metrics["invoice_number"] is True
    assert metrics["vendor_name"] is True  # fuzzy match
    assert metrics["total_amount"] is True


def test_compute_line_item_metrics():
    pred_items = [
        LineItem(description="Widget A", quantity=10, unit_price=5.0, line_total=50.0),
        LineItem(description="Widget B", quantity=5, unit_price=10.0, line_total=50.0),
    ]
    gold_items = [
        LineItem(description="Widget A", quantity=10, unit_price=5.0, line_total=50.0),
        LineItem(description="Widget B", quantity=5, unit_price=10.0, line_total=50.0),
    ]
    score = compute_line_item_metrics(pred_items, gold_items)
    assert score == 1.0


def test_compute_line_item_metrics_partial():
    pred_items = [
        LineItem(description="Widget A", quantity=10, unit_price=5.0, line_total=50.0),
    ]
    gold_items = [
        LineItem(description="Widget A", quantity=10, unit_price=5.0, line_total=50.0),
        LineItem(description="Widget B", quantity=5, unit_price=10.0, line_total=50.0),
    ]
    score = compute_line_item_metrics(pred_items, gold_items)
    assert score < 1.0
