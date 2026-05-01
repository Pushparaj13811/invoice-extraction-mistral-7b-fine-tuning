# Invoice Extraction Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Mistral 7B on invoice data extraction using QLoRA with benchmarked evaluation against GPT-4o-mini.

**Architecture:** Modular Python package (`src/`) with three subpackages (data, training, evaluation) consumed by three Kaggle notebooks. Data flows as JSONL through the pipeline: raw invoices → normalized schema → instruction-tuning format → training → eval comparison.

**Tech Stack:** Python 3.10+, Hugging Face (transformers, peft, trl, datasets), bitsandbytes, Pydantic, rapidfuzz, wandb, openai (Azure), anthropic

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `configs/default.json`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/training/__init__.py`
- Create: `src/evaluation/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/data/__init__.py`
- Create: `tests/training/__init__.py`
- Create: `tests/evaluation/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```txt
transformers>=4.40.0
peft>=0.10.0
bitsandbytes>=0.43.0
trl>=0.8.0
datasets>=2.19.0
wandb>=0.17.0
openai>=1.30.0
anthropic>=0.25.0
pydantic>=2.7.0
rapidfuzz>=3.9.0
accelerate>=0.30.0
torch>=2.2.0
scipy>=1.13.0
sentencepiece>=0.2.0
protobuf>=5.26.0
python-dotenv>=1.0.0
pytest>=8.2.0
```

- [ ] **Step 2: Create .gitignore**

```
# Data
data/
*.jsonl
*.csv

# Python
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/

# Environment
.env
.venv/
venv/

# IDE
.vscode/
.idea/

# Jupyter
.ipynb_checkpoints/

# Training outputs
outputs/
checkpoints/
wandb/

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 3: Create .env.example**

```
AZURE_OPENAI_API_KEY=your-azure-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-15-preview
ANTHROPIC_API_KEY=your-anthropic-key-here
WANDB_API_KEY=your-wandb-key-here
HF_TOKEN=your-huggingface-token-here
```

- [ ] **Step 4: Create configs/default.json**

```json
{
  "model_name": "mistralai/Mistral-7B-v0.3",
  "lora": {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "task_type": "CAUSAL_LM"
  },
  "quantization": {
    "load_in_4bit": true,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": true,
    "bnb_4bit_compute_dtype": "float16"
  },
  "training": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 50,
    "max_seq_length": 2048,
    "optim": "paged_adamw_8bit",
    "save_steps": 200,
    "logging_steps": 10,
    "eval_steps": 100
  },
  "wandb": {
    "project": "invoice-extraction-finetune"
  },
  "data": {
    "train_size": 2000,
    "eval_size": 500,
    "seed": 42,
    "synthetic_count": 1500,
    "existing_count": 500
  }
}
```

- [ ] **Step 5: Create all `__init__.py` files**

Create empty `__init__.py` files at:
- `src/__init__.py`
- `src/data/__init__.py`
- `src/training/__init__.py`
- `src/evaluation/__init__.py`
- `tests/__init__.py`
- `tests/data/__init__.py`
- `tests/training/__init__.py`
- `tests/evaluation/__init__.py`

Each file is empty (just a blank file).

- [ ] **Step 6: Create data/ directory**

```bash
mkdir -p data
```

This directory is gitignored — it holds downloaded/generated data locally.

---

### Task 2: Invoice Schema (Pydantic Models)

**Files:**
- Create: `src/data/schema.py`
- Create: `tests/data/test_schema.py`

- [ ] **Step 1: Write tests for schema models**

Create `tests/data/test_schema.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/hompushparajmehta/Desktop/job/llm-fine-tuning
python -m pytest tests/data/test_schema.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.schema'`

- [ ] **Step 3: Implement schema.py**

Create `src/data/schema.py`:

```python
from pydantic import BaseModel


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    line_total: float


class Invoice(BaseModel):
    vendor_name: str
    invoice_number: str
    invoice_date: str
    due_date: str
    total_amount: float
    currency: str
    tax_amount: float | None = None
    discount: float | None = None
    billing_address: str | None = None
    payment_terms: str | None = None
    line_items: list[LineItem] = []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/data/test_schema.py -v
```

Expected: All 5 tests PASS.

---

### Task 3: Existing Dataset Loader

**Files:**
- Create: `src/data/existing_loader.py`
- Create: `tests/data/test_existing_loader.py`

- [ ] **Step 1: Write tests for the loader**

Create `tests/data/test_existing_loader.py`:

```python
import pytest
from src.data.schema import Invoice
from src.data.existing_loader import normalize_record, load_existing_dataset


def test_normalize_record_maps_fields():
    raw = {
        "company": "Acme Corp",
        "invoice_no": "INV-001",
        "date": "2024-01-15",
        "due": "2024-02-15",
        "total": 500.0,
        "currency": "USD",
        "tax": 50.0,
        "items": [
            {
                "desc": "Widget",
                "qty": 2,
                "price": 225.0,
                "total": 450.0,
            }
        ],
    }
    invoice = normalize_record(raw)
    assert isinstance(invoice, Invoice)
    assert invoice.vendor_name == "Acme Corp"
    assert invoice.invoice_number == "INV-001"
    assert invoice.total_amount == 500.0
    assert len(invoice.line_items) == 1
    assert invoice.line_items[0].description == "Widget"


def test_normalize_record_missing_optional_fields():
    raw = {
        "company": "Test Co",
        "invoice_no": "T-001",
        "date": "2024-03-01",
        "due": "2024-04-01",
        "total": 100.0,
        "currency": "EUR",
    }
    invoice = normalize_record(raw)
    assert invoice.tax_amount is None
    assert invoice.line_items == []


def test_normalize_record_returns_none_for_junk():
    raw = {"garbage": "data"}
    result = normalize_record(raw)
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/data/test_existing_loader.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement existing_loader.py**

Create `src/data/existing_loader.py`:

```python
from __future__ import annotations
from typing import Any

from datasets import load_dataset

from src.data.schema import Invoice, LineItem


# Mapping from common HF dataset field names to our schema
FIELD_ALIASES = {
    "vendor_name": ["company", "vendor", "supplier", "seller", "vendor_name"],
    "invoice_number": ["invoice_no", "invoice_num", "inv_number", "invoice_number", "invoice_id"],
    "invoice_date": ["date", "invoice_date", "inv_date", "issue_date"],
    "due_date": ["due", "due_date", "payment_due", "due_by"],
    "total_amount": ["total", "total_amount", "amount", "grand_total", "invoice_total"],
    "currency": ["currency", "cur", "currency_code"],
    "tax_amount": ["tax", "tax_amount", "vat", "tax_total"],
    "discount": ["discount", "discount_amount"],
    "billing_address": ["billing_address", "address", "bill_to", "customer_address"],
    "payment_terms": ["payment_terms", "terms", "pay_terms"],
}

ITEM_ALIASES = {
    "description": ["desc", "description", "item", "item_name", "name", "product"],
    "quantity": ["qty", "quantity", "count", "units"],
    "unit_price": ["price", "unit_price", "rate", "unit_cost"],
    "line_total": ["total", "line_total", "amount", "ext_price", "extended"],
}


def _resolve(data: dict, aliases: dict[str, list[str]]) -> dict[str, Any]:
    """Resolve field aliases to canonical names."""
    resolved = {}
    for canonical, names in aliases.items():
        for name in names:
            if name in data and data[name] is not None:
                resolved[canonical] = data[name]
                break
    return resolved


def normalize_record(raw: dict) -> Invoice | None:
    """Normalize a raw dataset record to our Invoice schema.

    Returns None if required fields are missing.
    """
    fields = _resolve(raw, FIELD_ALIASES)

    required = ["vendor_name", "invoice_number", "invoice_date", "due_date", "total_amount", "currency"]
    if not all(k in fields for k in required):
        return None

    # Coerce total_amount to float
    try:
        fields["total_amount"] = float(fields["total_amount"])
    except (ValueError, TypeError):
        return None

    # Optional float fields
    for key in ("tax_amount", "discount"):
        if key in fields:
            try:
                fields[key] = float(fields[key])
            except (ValueError, TypeError):
                fields[key] = None

    # Normalize line items
    raw_items = raw.get("items", raw.get("line_items", []))
    line_items = []
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            item_fields = _resolve(item, ITEM_ALIASES)
            try:
                line_items.append(
                    LineItem(
                        description=str(item_fields.get("description", "")),
                        quantity=float(item_fields.get("quantity", 0)),
                        unit_price=float(item_fields.get("unit_price", 0)),
                        line_total=float(item_fields.get("line_total", 0)),
                    )
                )
            except (ValueError, TypeError):
                continue

    fields["line_items"] = line_items

    try:
        return Invoice.model_validate(fields)
    except Exception:
        return None


def load_existing_dataset(
    dataset_name: str = "mychen76/invoices-and-receipts_ocr_v1",
    split: str = "train",
    max_samples: int = 500,
) -> list[Invoice]:
    """Load and normalize an existing HF dataset.

    Returns a list of Invoice objects, filtering out records
    that cannot be normalized.
    """
    ds = load_dataset(dataset_name, split=split)
    invoices = []
    for record in ds:
        invoice = normalize_record(record)
        if invoice is not None:
            invoices.append(invoice)
            if len(invoices) >= max_samples:
                break
    return invoices
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/data/test_existing_loader.py -v
```

Expected: All 3 tests PASS.

---

### Task 4: Synthetic Data Generator

**Files:**
- Create: `src/data/synthetic_gen.py`
- Create: `tests/data/test_synthetic_gen.py`

- [ ] **Step 1: Write tests for synthetic generation**

Create `tests/data/test_synthetic_gen.py`:

```python
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
    assert "5" in prompt  # batch_size


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
                {
                    "description": "Widget",
                    "quantity": 10,
                    "unit_price": 5.0,
                    "line_total": 50.0,
                }
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/data/test_synthetic_gen.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement synthetic_gen.py**

Create `src/data/synthetic_gen.py`:

```python
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from anthropic import Anthropic

from src.data.schema import Invoice


@dataclass
class SyntheticConfig:
    batch_size: int = 50
    currency: str = "USD"
    min_items: int = 1
    max_items: int = 10
    date_formats: list[str] = field(
        default_factory=lambda: ["YYYY-MM-DD", "MM/DD/YYYY", "DD-Mon-YYYY", "DD/MM/YYYY"]
    )


SCHEMA_DESCRIPTION = """
Each invoice must have these fields:
- vendor_name (string): company name
- invoice_number (string): unique ID
- invoice_date (string): when invoice was issued
- due_date (string): payment deadline
- total_amount (float): total including tax
- currency (string): e.g., USD, EUR, GBP
- tax_amount (float or null): tax portion
- discount (float or null): discount applied
- billing_address (string or null): customer address
- payment_terms (string or null): e.g., "Net 30"
- line_items (list): each with description, quantity, unit_price, line_total
"""


def build_generation_prompt(config: SyntheticConfig) -> str:
    return f"""Generate exactly {config.batch_size} realistic invoice examples.

For each invoice, provide:
1. "invoice_text" - the raw text of the invoice as it would appear in a document (include headers, addresses, line items, totals, etc.)
2. "extracted" - the structured JSON extraction with all fields

{SCHEMA_DESCRIPTION}

Requirements:
- Use {config.currency} as the primary currency (mix in others for ~20% of invoices)
- Line items per invoice: between {config.min_items} and {config.max_items}
- Vary date formats across: {', '.join(config.date_formats)}
- Use diverse vendor names, industries, and invoice styles
- Include edge cases: partial fields, unusual formatting, multi-line addresses
- Make invoice_text look realistic (not just key: value pairs)

Return a JSON array of {config.batch_size} objects, each with "invoice_text" and "extracted" keys.
Return ONLY the JSON array, no other text."""


def parse_synthetic_response(response_text: str) -> list[tuple[str, Invoice]]:
    """Parse the LLM response into (invoice_text, Invoice) pairs.

    Skips entries that fail validation.
    """
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    data = json.loads(text)
    results = []

    for entry in data:
        if not isinstance(entry, dict):
            continue
        invoice_text = entry.get("invoice_text", "")
        extracted = entry.get("extracted", {})
        if not invoice_text or not isinstance(extracted, dict):
            continue
        try:
            invoice = Invoice.model_validate(extracted)
            results.append((invoice_text, invoice))
        except Exception:
            continue

    return results


def generate_batch(
    config: SyntheticConfig,
    client: Anthropic | None = None,
) -> list[tuple[str, Invoice]]:
    """Generate a batch of synthetic invoices using Claude API."""
    if client is None:
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = build_generation_prompt(config)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = response.content[0].text
    return parse_synthetic_response(response_text)


def generate_dataset(
    total: int = 1500,
    batch_size: int = 50,
    client: Anthropic | None = None,
) -> list[tuple[str, Invoice]]:
    """Generate the full synthetic dataset in batches."""
    if client is None:
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    all_results = []
    currencies = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY"]

    batch_num = 0
    while len(all_results) < total:
        currency = currencies[batch_num % len(currencies)]
        config = SyntheticConfig(
            batch_size=min(batch_size, total - len(all_results)),
            currency=currency,
            min_items=1 + (batch_num % 3),
            max_items=5 + (batch_num % 6),
        )
        batch = generate_batch(config, client=client)
        all_results.extend(batch)
        batch_num += 1
        print(f"Batch {batch_num}: generated {len(batch)} invoices (total: {len(all_results)}/{total})")

    return all_results[:total]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/data/test_synthetic_gen.py -v
```

Expected: All 3 tests PASS.

---

### Task 5: Dataset Merge & Deduplication

**Files:**
- Create: `src/data/merge.py`
- Create: `tests/data/test_merge.py`

- [ ] **Step 1: Write tests for merge and dedup**

Create `tests/data/test_merge.py`:

```python
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
    inv2 = ("Invoice from Acme Corp for widgetts", _make_invoice("Acme", "INV-001"))  # typo
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/data/test_merge.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement merge.py**

Create `src/data/merge.py`:

```python
from __future__ import annotations

import random

from rapidfuzz import fuzz

from src.data.schema import Invoice


def deduplicate(
    data: list[tuple[str, Invoice]],
    threshold: float = 0.90,
) -> list[tuple[str, Invoice]]:
    """Remove duplicate invoices by fuzzy-matching on text.

    Keeps the first occurrence when two texts have similarity >= threshold.
    """
    kept: list[tuple[str, Invoice]] = []
    kept_texts: list[str] = []

    for text, invoice in data:
        is_dup = False
        for existing_text in kept_texts:
            if fuzz.ratio(text, existing_text) >= threshold * 100:
                is_dup = True
                break
        if not is_dup:
            kept.append((text, invoice))
            kept_texts.append(text)

    return kept


def merge_and_split(
    existing: list[tuple[str, Invoice]],
    synthetic: list[tuple[str, Invoice]],
    eval_size: int = 500,
    seed: int = 42,
) -> tuple[list[tuple[str, Invoice]], list[tuple[str, Invoice]]]:
    """Merge existing and synthetic data, deduplicate, and split into train/eval.

    The eval set is balanced: proportional samples from existing and synthetic.
    """
    combined = deduplicate(existing + synthetic)

    rng = random.Random(seed)
    rng.shuffle(combined)

    eval_size = min(eval_size, len(combined) // 4)
    eval_set = combined[:eval_size]
    train_set = combined[eval_size:]

    return train_set, eval_set
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/data/test_merge.py -v
```

Expected: All 4 tests PASS.

---

### Task 6: Instruction-Tuning Formatter

**Files:**
- Create: `src/data/format.py`
- Create: `tests/data/test_format.py`

- [ ] **Step 1: Write tests for the formatter**

Create `tests/data/test_format.py`:

```python
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
    # output should be valid JSON
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/data/test_format.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement format.py**

Create `src/data/format.py`:

```python
from __future__ import annotations

import json

from src.data.schema import Invoice

INSTRUCTION = "Extract all invoice fields from the following invoice text as JSON."


def format_example(invoice_text: str, invoice: Invoice) -> dict:
    """Convert an (invoice_text, Invoice) pair to instruction-tuning format."""
    return {
        "instruction": INSTRUCTION,
        "input": invoice_text,
        "output": invoice.model_dump_json(indent=2),
    }


def format_dataset(
    data: list[tuple[str, Invoice]],
) -> list[dict]:
    """Format a full dataset of (text, Invoice) pairs."""
    return [format_example(text, inv) for text, inv in data]


def save_jsonl(records: list[dict], path: str) -> None:
    """Save formatted records to a JSONL file."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def load_jsonl(path: str) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/data/test_format.py -v
```

Expected: All 3 tests PASS.

---

### Task 7: Training Config

**Files:**
- Create: `src/training/config.py`
- Create: `tests/training/test_config.py`

- [ ] **Step 1: Write tests for training config**

Create `tests/training/test_config.py`:

```python
import json
import pytest
from src.training.config import TrainingConfig


def test_default_config():
    config = TrainingConfig()
    assert config.model_name == "mistralai/Mistral-7B-v0.3"
    assert config.lora_r == 64
    assert config.lora_alpha == 128
    assert config.num_train_epochs == 3
    assert config.per_device_train_batch_size == 4


def test_config_from_json(tmp_path):
    data = {
        "model_name": "mistralai/Mistral-7B-v0.3",
        "lora_r": 32,
        "lora_alpha": 64,
        "num_train_epochs": 5,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    config = TrainingConfig.from_json(str(path))
    assert config.lora_r == 32
    assert config.num_train_epochs == 5
    # defaults for unset fields
    assert config.per_device_train_batch_size == 4


def test_config_effective_batch_size():
    config = TrainingConfig(per_device_train_batch_size=4, gradient_accumulation_steps=4)
    assert config.effective_batch_size == 16
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/training/test_config.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement config.py**

Create `src/training/config.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    # Model
    model_name: str = "mistralai/Mistral-7B-v0.3"

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 50
    max_seq_length: int = 2048
    optim: str = "paged_adamw_8bit"
    save_steps: int = 200
    logging_steps: int = 10
    eval_steps: int = 100

    # W&B
    wandb_project: str = "invoice-extraction-finetune"

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_json(cls, path: str) -> TrainingConfig:
        with open(path) as f:
            data = json.load(f)
        # Flatten nested config structure if needed
        flat = {}
        for key, value in data.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value
        # Only pass keys that match our fields
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in flat.items() if k in valid_keys}
        return cls(**filtered)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/training/test_config.py -v
```

Expected: All 3 tests PASS.

---

### Task 8: QLoRA Model Setup

**Files:**
- Create: `src/training/lora_setup.py`
- Create: `tests/training/test_lora_setup.py`

- [ ] **Step 1: Write tests for LoRA setup**

Create `tests/training/test_lora_setup.py`:

```python
import pytest
from src.training.config import TrainingConfig
from src.training.lora_setup import build_bnb_config, build_lora_config


def test_build_bnb_config():
    config = TrainingConfig()
    bnb = build_bnb_config(config)
    assert bnb.load_in_4bit is True
    assert bnb.bnb_4bit_quant_type == "nf4"
    assert bnb.bnb_4bit_use_double_quant is True


def test_build_lora_config():
    config = TrainingConfig(lora_r=32, lora_alpha=64)
    lora = build_lora_config(config)
    assert lora.r == 32
    assert lora.lora_alpha == 64
    assert lora.lora_dropout == 0.05
    assert "q_proj" in lora.target_modules
    assert lora.task_type == "CAUSAL_LM"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/training/test_lora_setup.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement lora_setup.py**

Create `src/training/lora_setup.py`:

```python
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.training.config import TrainingConfig


def build_bnb_config(config: TrainingConfig) -> BitsAndBytesConfig:
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_lora_config(config: TrainingConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        task_type=config.task_type if hasattr(config, "task_type") else "CAUSAL_LM",
        bias="none",
    )


def load_model_and_tokenizer(config: TrainingConfig):
    """Load quantized model and tokenizer, apply LoRA adapters.

    Returns (model, tokenizer) ready for training.
    """
    bnb_config = build_bnb_config(config)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model = prepare_model_for_kbit_training(model)

    lora_config = build_lora_config(config)
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, tokenizer
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/training/test_lora_setup.py -v
```

Expected: All 2 tests PASS. (These tests only check config building, not model loading — that requires GPU.)

---

### Task 9: Training Loop

**Files:**
- Create: `src/training/train.py`
- Create: `tests/training/test_train.py`

- [ ] **Step 1: Write tests for training utilities**

Create `tests/training/test_train.py`:

```python
import pytest
from src.training.config import TrainingConfig
from src.training.train import build_training_args, format_for_sft


def test_build_training_args():
    config = TrainingConfig(
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
    )
    args = build_training_args(config, output_dir="/tmp/test_output")
    assert args.num_train_epochs == 3
    assert args.per_device_train_batch_size == 4
    assert args.learning_rate == 2e-4
    assert args.output_dir == "/tmp/test_output"


def test_format_for_sft():
    example = {
        "instruction": "Extract fields.",
        "input": "Invoice text here",
        "output": '{"vendor_name": "Acme"}',
    }
    result = format_for_sft(example)
    assert "### Instruction:" in result
    assert "Invoice text here" in result
    assert '{"vendor_name": "Acme"}' in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/training/test_train.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement train.py**

Create `src/training/train.py`:

```python
from __future__ import annotations

import os

import wandb
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from src.training.config import TrainingConfig
from src.training.lora_setup import load_model_and_tokenizer
from src.data.format import load_jsonl


def format_for_sft(example: dict) -> str:
    """Convert an instruction-tuning example to a prompt string for SFTTrainer."""
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


def build_training_args(config: TrainingConfig, output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        optim=config.optim,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        fp16=True,
        gradient_checkpointing=True,
        group_by_length=True,
    )


def train(
    config: TrainingConfig,
    train_path: str,
    eval_path: str,
    output_dir: str = "outputs/",
):
    """Full training pipeline: load data, model, train, save."""
    # Init W&B
    wandb.init(
        project=config.wandb_project,
        config={
            "model": config.model_name,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "epochs": config.num_train_epochs,
            "batch_size": config.effective_batch_size,
            "learning_rate": config.learning_rate,
        },
    )

    # Load data
    train_data = load_jsonl(train_path)
    eval_data = load_jsonl(eval_path)

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    # Load model
    model, tokenizer = load_model_and_tokenizer(config)

    # Training args
    training_args = build_training_args(config, output_dir)

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=lambda x: [format_for_sft(x)],
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(os.path.join(output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_adapter"))

    wandb.finish()

    return trainer
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/training/test_train.py -v
```

Expected: All 2 tests PASS.

---

### Task 10: Evaluation Metrics

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `tests/evaluation/test_metrics.py`

- [ ] **Step 1: Write tests for metrics**

Create `tests/evaluation/test_metrics.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/evaluation/test_metrics.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement metrics.py**

Create `src/evaluation/metrics.py`:

```python
from __future__ import annotations

from rapidfuzz import fuzz

from src.data.schema import Invoice, LineItem

# Fields that use exact match
EXACT_FIELDS = [
    "invoice_number", "invoice_date", "due_date",
    "total_amount", "currency", "tax_amount", "discount", "payment_terms",
]

# Fields that use fuzzy match
FUZZY_FIELDS = ["vendor_name", "billing_address"]


def exact_match(pred, gold) -> bool:
    """Check if two values are exactly equal."""
    if pred is None and gold is None:
        return True
    if pred is None or gold is None:
        return False
    return str(pred).strip() == str(gold).strip()


def fuzzy_match(pred: str, gold: str, threshold: float = 0.85) -> bool:
    """Check if two strings are similar enough (Levenshtein ratio)."""
    if pred is None and gold is None:
        return True
    if pred is None or gold is None:
        return False
    ratio = fuzz.ratio(str(pred).strip(), str(gold).strip()) / 100.0
    return ratio >= threshold


def compute_field_metrics(pred: Invoice, gold: Invoice) -> dict[str, bool]:
    """Compute per-field match for a single prediction vs gold invoice."""
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
    """Compute line item matching score.

    Matches predicted items to gold items by description similarity,
    then checks quantity, unit_price, line_total for each match.
    Returns a score between 0.0 and 1.0.
    """
    if not gold_items and not pred_items:
        return 1.0
    if not gold_items or not pred_items:
        return 0.0

    total_fields = len(gold_items) * 4  # description, qty, price, total per item
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

            # Description (fuzzy)
            if best_score >= 0.85:
                matched_fields += 1

            # Quantity (exact)
            if exact_match(pred_item.quantity, gold_item.quantity):
                matched_fields += 1

            # Unit price (exact)
            if exact_match(pred_item.unit_price, gold_item.unit_price):
                matched_fields += 1

            # Line total (exact)
            if exact_match(pred_item.line_total, gold_item.line_total):
                matched_fields += 1

    return matched_fields / total_fields


def compute_invoice_metrics(pred: Invoice, gold: Invoice) -> dict:
    """Full evaluation of a single prediction.

    Returns field-level results + line item score + overall accuracy.
    """
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/evaluation/test_metrics.py -v
```

Expected: All 9 tests PASS.

---

### Task 11: Baseline Runner (GPT-4o-mini via Azure)

**Files:**
- Create: `src/evaluation/baseline.py`
- Create: `tests/evaluation/test_baseline.py`

- [ ] **Step 1: Write tests for baseline**

Create `tests/evaluation/test_baseline.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/evaluation/test_baseline.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement baseline.py**

Create `src/evaluation/baseline.py`:

```python
from __future__ import annotations

import json
import os

from openai import AzureOpenAI

from src.data.schema import Invoice


BASELINE_SYSTEM_PROMPT = """You are an invoice data extraction system. Given raw invoice text, extract all fields into a JSON object.

Required fields:
- vendor_name (string)
- invoice_number (string)
- invoice_date (string)
- due_date (string)
- total_amount (float)
- currency (string)

Optional fields (use null if not found):
- tax_amount (float or null)
- discount (float or null)
- billing_address (string or null)
- payment_terms (string or null)

Line items (list of objects, each with):
- description (string)
- quantity (number)
- unit_price (float)
- line_total (float)

Return ONLY valid JSON. No explanation, no markdown fences."""


def build_baseline_prompt(invoice_text: str) -> str:
    return f"""Extract all invoice fields from the following invoice text as JSON.

{BASELINE_SYSTEM_PROMPT}

Invoice text:
---
{invoice_text}
---

Return the extracted JSON:"""


def parse_baseline_response(response_text: str) -> Invoice | None:
    """Parse a baseline model response into an Invoice.

    Returns None if parsing fails.
    """
    text = response_text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
        return Invoice.model_validate(data)
    except Exception:
        return None


def run_baseline(
    eval_data: list[dict],
    client: AzureOpenAI | None = None,
) -> list[Invoice | None]:
    """Run GPT-4o-mini baseline on eval examples.

    Args:
        eval_data: list of dicts with "input" key (invoice text)
        client: Azure OpenAI client (created from env vars if None)

    Returns:
        List of Invoice predictions (None for failures)
    """
    if client is None:
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    predictions = []

    for i, example in enumerate(eval_data):
        invoice_text = example["input"]
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract all invoice fields from this text:\n\n{invoice_text}"},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            result = parse_baseline_response(response.choices[0].message.content)
            predictions.append(result)
        except Exception as e:
            print(f"Baseline error on example {i}: {e}")
            predictions.append(None)

        if (i + 1) % 50 == 0:
            print(f"Baseline progress: {i + 1}/{len(eval_data)}")

    return predictions
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/evaluation/test_baseline.py -v
```

Expected: All 4 tests PASS.

---

### Task 12: Evaluation Runner & Report Generator

**Files:**
- Create: `src/evaluation/evaluate.py`
- Create: `tests/evaluation/test_evaluate.py`

- [ ] **Step 1: Write tests for evaluation**

Create `tests/evaluation/test_evaluate.py`:

```python
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
    preds = [_make_invoice(), None]  # second prediction failed
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/evaluation/test_evaluate.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement evaluate.py**

Create `src/evaluation/evaluate.py`:

```python
from __future__ import annotations

import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.schema import Invoice
from src.data.format import load_jsonl
from src.evaluation.metrics import compute_invoice_metrics
from src.training.train import format_for_sft


def load_finetuned_model(base_model: str, adapter_path: str):
    """Load the base model with merged LoRA adapter for inference."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def run_finetuned_inference(
    model,
    tokenizer,
    eval_data: list[dict],
    max_new_tokens: int = 1024,
) -> list[Invoice | None]:
    """Run fine-tuned model on eval examples."""
    predictions = []

    for i, example in enumerate(eval_data):
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Strip markdown fences if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
            predictions.append(Invoice.model_validate(data))
        except Exception:
            predictions.append(None)

        if (i + 1) % 50 == 0:
            print(f"Inference progress: {i + 1}/{len(eval_data)}")

    return predictions


def aggregate_metrics(
    predictions: list[Invoice | None],
    golds: list[Invoice],
) -> dict:
    """Aggregate metrics across all eval examples."""
    per_field_totals: dict[str, list[bool]] = {}
    line_item_scores = []
    overall_scores = []
    parse_successes = 0

    for pred, gold in zip(predictions, golds):
        if pred is None:
            # Count as all-wrong for this example
            overall_scores.append(0.0)
            continue

        parse_successes += 1
        result = compute_invoice_metrics(pred, gold)

        for field_name, matched in result["fields"].items():
            if field_name not in per_field_totals:
                per_field_totals[field_name] = []
            per_field_totals[field_name].append(matched)

        line_item_scores.append(result["line_item_score"])
        overall_scores.append(result["overall_accuracy"])

    per_field_accuracy = {
        name: sum(vals) / len(vals) for name, vals in per_field_totals.items()
    }

    return {
        "overall_accuracy": sum(overall_scores) / len(overall_scores) if overall_scores else 0.0,
        "json_parse_success_rate": parse_successes / len(predictions) if predictions else 0.0,
        "per_field": per_field_accuracy,
        "line_item_score": sum(line_item_scores) / len(line_item_scores) if line_item_scores else 0.0,
        "total_examples": len(predictions),
        "parse_failures": len(predictions) - parse_successes,
    }


def generate_report(ft_metrics: dict, baseline_metrics: dict) -> str:
    """Generate a markdown comparison report."""
    lines = [
        "# Invoice Extraction — Evaluation Report\n",
        "## Overall Results\n",
        "| Metric | Fine-Tuned Mistral 7B | GPT-4o-mini Baseline | Improvement |",
        "|--------|----------------------|---------------------|-------------|",
    ]

    ft_acc = ft_metrics["overall_accuracy"]
    bl_acc = baseline_metrics["overall_accuracy"]
    improvement = ((ft_acc - bl_acc) / bl_acc * 100) if bl_acc > 0 else 0
    lines.append(
        f"| Overall Accuracy | {ft_acc:.1%} | {bl_acc:.1%} | {improvement:+.1f}% |"
    )

    ft_parse = ft_metrics["json_parse_success_rate"]
    bl_parse = baseline_metrics["json_parse_success_rate"]
    lines.append(
        f"| JSON Parse Rate | {ft_parse:.1%} | {bl_parse:.1%} | — |"
    )

    ft_li = ft_metrics.get("line_item_score", 0)
    bl_li = baseline_metrics.get("line_item_score", 0)
    li_imp = ((ft_li - bl_li) / bl_li * 100) if bl_li > 0 else 0
    lines.append(
        f"| Line Item Score | {ft_li:.1%} | {bl_li:.1%} | {li_imp:+.1f}% |"
    )

    lines.append("\n## Per-Field Accuracy\n")
    lines.append("| Field | Fine-Tuned | GPT-4o-mini | Improvement |")
    lines.append("|-------|-----------|-------------|-------------|")

    all_fields = set(list(ft_metrics.get("per_field", {}).keys()) + list(baseline_metrics.get("per_field", {}).keys()))
    for field_name in sorted(all_fields):
        ft_val = ft_metrics.get("per_field", {}).get(field_name, 0)
        bl_val = baseline_metrics.get("per_field", {}).get(field_name, 0)
        imp = ((ft_val - bl_val) / bl_val * 100) if bl_val > 0 else 0
        lines.append(f"| {field_name} | {ft_val:.1%} | {bl_val:.1%} | {imp:+.1f}% |")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/evaluation/test_evaluate.py -v
```

Expected: All 3 tests PASS.

---

### Task 13: Notebook 01 — Data Preparation

**Files:**
- Create: `notebooks/01_data_preparation.ipynb`

- [ ] **Step 1: Create the data preparation notebook**

Create `notebooks/01_data_preparation.ipynb` as a Jupyter notebook with the following cells:

**Cell 1 (markdown):**
```markdown
# 01 — Data Preparation
Prepare the invoice extraction dataset: load existing data, generate synthetic examples, merge, and format for training.
```

**Cell 2 (code) — Install deps:**
```python
!pip install -q transformers datasets pydantic anthropic rapidfuzz python-dotenv
```

**Cell 3 (code) — Setup:**
```python
import os
import sys
from dotenv import load_dotenv

# If running on Kaggle, clone the repo or upload src/
# sys.path.insert(0, "/kaggle/input/your-dataset/llm-fine-tuning")
sys.path.insert(0, "..")

load_dotenv()

from src.data.schema import Invoice, LineItem
from src.data.existing_loader import load_existing_dataset, normalize_record
from src.data.synthetic_gen import generate_dataset
from src.data.merge import merge_and_split, deduplicate
from src.data.format import format_dataset, save_jsonl
```

**Cell 4 (markdown):**
```markdown
## Step 1: Load Existing Dataset
```

**Cell 5 (code):**
```python
# Load and normalize existing HF dataset
existing_invoices = load_existing_dataset(
    dataset_name="mychen76/invoices-and-receipts_ocr_v1",
    max_samples=500,
)
print(f"Loaded {len(existing_invoices)} existing invoices")

# Preview
if existing_invoices:
    print(existing_invoices[0].model_dump_json(indent=2))
```

**Cell 6 (markdown):**
```markdown
## Step 2: Generate Synthetic Dataset
```

**Cell 7 (code):**
```python
# Generate synthetic invoices using Claude API
# This will take several minutes and make ~30 API calls
synthetic_data = generate_dataset(total=1500, batch_size=50)
print(f"Generated {len(synthetic_data)} synthetic invoices")

# Preview
if synthetic_data:
    text, invoice = synthetic_data[0]
    print("=== Invoice Text ===")
    print(text[:500])
    print("\n=== Extracted Fields ===")
    print(invoice.model_dump_json(indent=2))
```

**Cell 8 (markdown):**
```markdown
## Step 3: Merge and Split
```

**Cell 9 (code):**
```python
# Convert existing invoices to (text, Invoice) pairs
# For existing dataset, we'll use the raw text field as invoice_text
existing_pairs = []
for inv in existing_invoices:
    # Create a text representation from the invoice fields
    text = f"Invoice #{inv.invoice_number}\nFrom: {inv.vendor_name}\nDate: {inv.invoice_date}\nDue: {inv.due_date}\nTotal: {inv.currency} {inv.total_amount}"
    if inv.line_items:
        text += "\n\nItems:"
        for item in inv.line_items:
            text += f"\n  {item.description} x{item.quantity} @ {item.unit_price} = {item.line_total}"
    existing_pairs.append((text, inv))

# Merge and split
train_data, eval_data = merge_and_split(
    existing_pairs,
    synthetic_data,
    eval_size=500,
    seed=42,
)

print(f"Train set: {len(train_data)} examples")
print(f"Eval set: {len(eval_data)} examples")
```

**Cell 10 (markdown):**
```markdown
## Step 4: Format and Save
```

**Cell 11 (code):**
```python
# Format for instruction tuning
train_formatted = format_dataset(train_data)
eval_formatted = format_dataset(eval_data)

# Save
os.makedirs("../data", exist_ok=True)
save_jsonl(train_formatted, "../data/train.jsonl")
save_jsonl(eval_formatted, "../data/eval.jsonl")

print(f"Saved {len(train_formatted)} training examples to data/train.jsonl")
print(f"Saved {len(eval_formatted)} eval examples to data/eval.jsonl")

# Preview formatted example
import json
print("\n=== Formatted Example ===")
print(json.dumps(train_formatted[0], indent=2)[:1000])
```

- [ ] **Step 2: Verify notebook file is valid JSON**

```bash
python -c "import json; json.load(open('notebooks/01_data_preparation.ipynb'))"
```

Expected: No error.

---

### Task 14: Notebook 02 — Training

**Files:**
- Create: `notebooks/02_training.ipynb`

- [ ] **Step 1: Create the training notebook**

Create `notebooks/02_training.ipynb` as a Jupyter notebook with these cells:

**Cell 1 (markdown):**
```markdown
# 02 — Fine-Tuning Mistral 7B with QLoRA
Train the invoice extraction model using QLoRA on Kaggle's T4 GPU.
```

**Cell 2 (code) — Install deps:**
```python
!pip install -q transformers peft bitsandbytes trl datasets accelerate wandb sentencepiece protobuf scipy
```

**Cell 3 (code) — Setup:**
```python
import os
import sys
sys.path.insert(0, "..")

import wandb
from src.training.config import TrainingConfig
from src.training.train import train

# Login to W&B
wandb.login()
```

**Cell 4 (markdown):**
```markdown
## Configure Training
```

**Cell 5 (code):**
```python
config = TrainingConfig(
    model_name="mistralai/Mistral-7B-v0.3",
    lora_r=64,
    lora_alpha=128,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_seq_length=2048,
)

print(f"Model: {config.model_name}")
print(f"LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}")
print(f"Effective batch size: {config.effective_batch_size}")
print(f"Epochs: {config.num_train_epochs}")
```

**Cell 6 (markdown):**
```markdown
## Train
```

**Cell 7 (code):**
```python
trainer = train(
    config=config,
    train_path="../data/train.jsonl",
    eval_path="../data/eval.jsonl",
    output_dir="../outputs/",
)

print("Training complete!")
print(f"Best checkpoint saved to ../outputs/final_adapter")
```

**Cell 8 (markdown):**
```markdown
## Verify Adapter
```

**Cell 9 (code):**
```python
import os
adapter_path = "../outputs/final_adapter"
files = os.listdir(adapter_path)
print(f"Adapter files: {files}")

# Quick sanity check — load adapter and run one example
from src.evaluation.evaluate import load_finetuned_model

model, tokenizer = load_finetuned_model(config.model_name, adapter_path)

test_input = "Invoice #TEST-001\nFrom: Test Corp\nDate: 2024-06-15\nDue: 2024-07-15\nItem: Widget x1 @ $100 = $100\nTotal: $110 (tax $10)\nCurrency: USD"
prompt = f"### Instruction:\nExtract all invoice fields from the following invoice text as JSON.\n\n### Input:\n{test_input}\n\n### Response:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
import torch
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("Model output:")
print(response)
```

- [ ] **Step 2: Verify notebook file is valid JSON**

```bash
python -c "import json; json.load(open('notebooks/02_training.ipynb'))"
```

Expected: No error.

---

### Task 15: Notebook 03 — Evaluation

**Files:**
- Create: `notebooks/03_evaluation.ipynb`

- [ ] **Step 1: Create the evaluation notebook**

Create `notebooks/03_evaluation.ipynb` as a Jupyter notebook with these cells:

**Cell 1 (markdown):**
```markdown
# 03 — Evaluation: Fine-Tuned Mistral 7B vs GPT-4o-mini
Compare fine-tuned model against GPT-4o-mini (Azure) baseline on 500 eval examples.
```

**Cell 2 (code) — Install deps:**
```python
!pip install -q transformers peft bitsandbytes datasets openai rapidfuzz wandb python-dotenv
```

**Cell 3 (code) — Setup:**
```python
import os
import sys
import json
sys.path.insert(0, "..")

from dotenv import load_dotenv
load_dotenv()

from src.data.format import load_jsonl
from src.data.schema import Invoice
from src.evaluation.evaluate import (
    load_finetuned_model,
    run_finetuned_inference,
    aggregate_metrics,
    generate_report,
)
from src.evaluation.baseline import run_baseline

import wandb
wandb.init(project="invoice-extraction-finetune", job_type="evaluation")
```

**Cell 4 (markdown):**
```markdown
## Load Eval Data
```

**Cell 5 (code):**
```python
eval_data = load_jsonl("../data/eval.jsonl")
print(f"Loaded {len(eval_data)} eval examples")

# Parse gold labels
gold_invoices = []
for ex in eval_data:
    gold_invoices.append(Invoice.model_validate_json(ex["output"]))
```

**Cell 6 (markdown):**
```markdown
## Run Fine-Tuned Model
```

**Cell 7 (code):**
```python
model, tokenizer = load_finetuned_model(
    base_model="mistralai/Mistral-7B-v0.3",
    adapter_path="../outputs/final_adapter",
)

ft_predictions = run_finetuned_inference(model, tokenizer, eval_data)
ft_metrics = aggregate_metrics(ft_predictions, gold_invoices)

print(f"Fine-tuned overall accuracy: {ft_metrics['overall_accuracy']:.1%}")
print(f"JSON parse rate: {ft_metrics['json_parse_success_rate']:.1%}")
```

**Cell 8 (markdown):**
```markdown
## Run GPT-4o-mini Baseline
```

**Cell 9 (code):**
```python
baseline_predictions = run_baseline(eval_data)
baseline_metrics = aggregate_metrics(baseline_predictions, gold_invoices)

print(f"Baseline overall accuracy: {baseline_metrics['overall_accuracy']:.1%}")
print(f"JSON parse rate: {baseline_metrics['json_parse_success_rate']:.1%}")
```

**Cell 10 (markdown):**
```markdown
## Comparison Report
```

**Cell 11 (code):**
```python
report = generate_report(ft_metrics, baseline_metrics)
print(report)

# Save report
os.makedirs("../reports", exist_ok=True)
with open("../reports/evaluation_report.md", "w") as f:
    f.write(report)

# Log to W&B
wandb.log({
    "ft_overall_accuracy": ft_metrics["overall_accuracy"],
    "baseline_overall_accuracy": baseline_metrics["overall_accuracy"],
    "ft_parse_rate": ft_metrics["json_parse_success_rate"],
    "baseline_parse_rate": baseline_metrics["json_parse_success_rate"],
    "improvement_pct": (
        (ft_metrics["overall_accuracy"] - baseline_metrics["overall_accuracy"])
        / baseline_metrics["overall_accuracy"] * 100
    ),
})

# Log per-field metrics
for field_name in ft_metrics["per_field"]:
    wandb.log({
        f"ft_{field_name}": ft_metrics["per_field"].get(field_name, 0),
        f"baseline_{field_name}": baseline_metrics["per_field"].get(field_name, 0),
    })

wandb.finish()
print("\nReport saved to reports/evaluation_report.md")
print("Metrics logged to W&B")
```

- [ ] **Step 2: Verify notebook file is valid JSON**

```bash
python -c "import json; json.load(open('notebooks/03_evaluation.ipynb'))"
```

Expected: No error.

---

### Task 16: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README.md**

```markdown
# LLM Fine-Tuning: Invoice Data Extraction

Fine-tuned Mistral 7B on invoice data extraction using QLoRA, benchmarked against GPT-4o-mini baseline.

## Results

| Metric | Fine-Tuned Mistral 7B | GPT-4o-mini Baseline | Improvement |
|--------|----------------------|---------------------|-------------|
| Overall Accuracy | TBD | TBD | TBD |
| JSON Parse Rate | TBD | TBD | — |

> Results will be updated after training completes.

## Project Structure

```
src/
├── data/           # Data loading, generation, formatting
├── training/       # QLoRA setup, training loop
└── evaluation/     # Metrics, baseline, comparison
notebooks/
├── 01_data_preparation.ipynb
├── 02_training.ipynb
└── 03_evaluation.ipynb
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # Fill in your API keys
```

## Usage

### 1. Prepare Data
Run `notebooks/01_data_preparation.ipynb` or:
```bash
python -c "from src.data.synthetic_gen import generate_dataset; generate_dataset()"
```

### 2. Train
Run `notebooks/02_training.ipynb` on Kaggle (T4 GPU required).

### 3. Evaluate
Run `notebooks/03_evaluation.ipynb` to compare against GPT-4o-mini baseline.

## Tech Stack

- **Model:** Mistral 7B v0.3
- **Fine-tuning:** QLoRA (4-bit NF4, LoRA rank 64)
- **Training:** Hugging Face TRL SFTTrainer
- **Tracking:** Weights & Biases
- **Baseline:** GPT-4o-mini via Azure OpenAI
- **Evaluation:** Field-level exact + fuzzy matching (rapidfuzz)

## Reproducibility

All notebooks are designed to run on Kaggle free tier (T4 GPU, 16GB VRAM).
Training takes approximately 2-3 hours for 3 epochs on 2,000 examples.
```

- [ ] **Step 2: Verify README renders properly**

Visual check — open the file and confirm markdown formatting looks correct.

---

### Task 17: Run All Tests

- [ ] **Step 1: Run the full test suite**

```bash
cd /Users/hompushparajmehta/Desktop/job/llm-fine-tuning
python -m pytest tests/ -v
```

Expected: All tests PASS (approximately 26 tests across 8 test files).

- [ ] **Step 2: Fix any failures**

If any tests fail, fix the issues and re-run until all pass.
