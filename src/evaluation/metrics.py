from __future__ import annotations

from rapidfuzz import fuzz

from src.data.schema import Invoice, LineItem

EXACT_FIELDS = [
    "invoice_number", "invoice_date", "due_date",
    "total_amount", "currency", "tax_amount", "discount", "payment_terms",
]

FUZZY_FIELDS = ["vendor_name", "billing_address"]


def exact_match(pred, gold) -> bool:
    if pred is None and gold is None:
        return True
    if pred is None or gold is None:
        return False
    return str(pred).strip() == str(gold).strip()


def fuzzy_match(pred: str, gold: str, threshold: float = 0.85) -> bool:
    if pred is None and gold is None:
        return True
    if pred is None or gold is None:
        return False
    ratio = fuzz.ratio(str(pred).strip(), str(gold).strip()) / 100.0
    return ratio >= threshold


def compute_field_metrics(pred: Invoice, gold: Invoice) -> dict[str, bool]:
    results = {}
    for field_name in EXACT_FIELDS:
        pred_val = getattr(pred, field_name, None)
        gold_val = getattr(gold, field_name, None)
        results[field_name] = exact_match(pred_val, gold_val)
    for field_name in FUZZY_FIELDS:
        pred_val = getattr(pred, field_name, None)
        gold_val = getattr(gold, field_name, None)
        results[field_name] = fuzzy_match(pred_val, gold_val)
    return results


def compute_line_item_metrics(
    pred_items: list[LineItem],
    gold_items: list[LineItem],
) -> float:
    if not gold_items and not pred_items:
        return 1.0
    if not gold_items or not pred_items:
        return 0.0

    total_fields = len(gold_items) * 4
    matched_fields = 0

    used_pred = set()
    for gold_item in gold_items:
        best_match_idx = -1
        best_score = 0.0
        for i, pred_item in enumerate(pred_items):
            if i in used_pred:
                continue
            score = fuzz.ratio(pred_item.description, gold_item.description) / 100.0
            if score > best_score:
                best_score = score
                best_match_idx = i

        if best_match_idx >= 0 and best_score >= 0.5:
            used_pred.add(best_match_idx)
            pred_item = pred_items[best_match_idx]

            if best_score >= 0.85:
                matched_fields += 1
            if exact_match(pred_item.quantity, gold_item.quantity):
                matched_fields += 1
            if exact_match(pred_item.unit_price, gold_item.unit_price):
                matched_fields += 1
            if exact_match(pred_item.line_total, gold_item.line_total):
                matched_fields += 1

    return matched_fields / total_fields


def compute_invoice_metrics(pred: Invoice, gold: Invoice) -> dict:
    field_results = compute_field_metrics(pred, gold)
    line_item_score = compute_line_item_metrics(pred.line_items, gold.line_items)

    field_correct = sum(1 for v in field_results.values() if v)
    field_total = len(field_results)

    overall_accuracy = (field_correct + line_item_score) / (field_total + 1)

    return {
        "fields": field_results,
        "line_item_score": line_item_score,
        "field_accuracy": field_correct / field_total if field_total > 0 else 0.0,
        "overall_accuracy": overall_accuracy,
    }
