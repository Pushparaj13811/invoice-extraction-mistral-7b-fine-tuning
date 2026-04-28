import json
import pytest
from src.data.schema import Invoice, LineItem
from src.evaluation.evaluate import (
    aggregate_metrics,
    generate_report,
)


def _make_invoice(vendor="Acme", number="INV-001", total=500.0):
    return Invoice(
        vendor_name=vendor,
        invoice_number=number,
        invoice_date="2024-01-15",
        due_date="2024-02-15",
        total_amount=total,
        currency="USD",
    )


def test_aggregate_metrics():
    golds = [_make_invoice(), _make_invoice("Beta", "INV-002", 300.0)]
    preds = [_make_invoice(), _make_invoice("Beta", "INV-002", 300.0)]
    result = aggregate_metrics(preds, golds)
    assert result["overall_accuracy"] == 1.0
    assert result["json_parse_success_rate"] == 1.0
    assert result["per_field"]["invoice_number"] == 1.0


def test_aggregate_metrics_with_none_preds():
    golds = [_make_invoice(), _make_invoice("Beta", "INV-002")]
    preds = [_make_invoice(), None]
    result = aggregate_metrics(preds, golds)
    assert result["json_parse_success_rate"] == 0.5
    assert result["overall_accuracy"] < 1.0


def test_generate_report():
    ft_metrics = {
        "overall_accuracy": 0.85,
        "json_parse_success_rate": 0.95,
        "per_field": {"vendor_name": 0.90, "invoice_number": 0.88},
        "line_item_score": 0.80,
    }
    baseline_metrics = {
        "overall_accuracy": 0.60,
        "json_parse_success_rate": 0.85,
        "per_field": {"vendor_name": 0.65, "invoice_number": 0.70},
        "line_item_score": 0.55,
    }
    report = generate_report(ft_metrics, baseline_metrics)
    assert "Fine-Tuned" in report
    assert "GPT-4o-mini" in report
    assert "Improvement" in report
