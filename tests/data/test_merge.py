import pytest
from src.data.schema import Invoice, LineItem
from src.data.merge import deduplicate, merge_and_split


def _make_invoice(vendor: str, number: str, total: float = 100.0) -> Invoice:
    return Invoice(
        vendor_name=vendor,
        invoice_number=number,
        invoice_date="2024-01-01",
        due_date="2024-02-01",
        total_amount=total,
        currency="USD",
    )


def test_deduplicate_removes_exact_dups():
    inv1 = ("text A", _make_invoice("Acme", "INV-001"))
    inv2 = ("text A", _make_invoice("Acme", "INV-001"))
    inv3 = ("text B", _make_invoice("Beta", "INV-002"))
    result = deduplicate([inv1, inv2, inv3])
    assert len(result) == 2


def test_deduplicate_removes_fuzzy_dups():
    inv1 = ("Invoice from Acme Corp for widgets", _make_invoice("Acme", "INV-001"))
    inv2 = ("Invoice from Acme Corp for widgetts", _make_invoice("Acme", "INV-001"))
    inv3 = ("Totally different invoice text", _make_invoice("Beta", "INV-002"))
    result = deduplicate([inv1, inv2, inv3], threshold=0.85)
    assert len(result) == 2


def test_merge_and_split_sizes():
    existing = [("text_e" + str(i), _make_invoice("E", f"E-{i}")) for i in range(100)]
    synthetic = [("text_s" + str(i), _make_invoice("S", f"S-{i}")) for i in range(300)]
    train, eval_set = merge_and_split(existing, synthetic, eval_size=50, seed=42)
    assert len(train) + len(eval_set) == 400
    assert len(eval_set) == 50


def test_merge_and_split_deterministic():
    existing = [("text_e" + str(i), _make_invoice("E", f"E-{i}")) for i in range(50)]
    synthetic = [("text_s" + str(i), _make_invoice("S", f"S-{i}")) for i in range(150)]
    train1, eval1 = merge_and_split(existing, synthetic, eval_size=20, seed=42)
    train2, eval2 = merge_and_split(existing, synthetic, eval_size=20, seed=42)
    assert [t[0] for t in train1] == [t[0] for t in train2]
    assert [e[0] for e in eval1] == [e[0] for e in eval2]
