# Invoice Extraction Fine-Tuning — Design Spec

## Overview

Fine-tune Mistral 7B on invoice data extraction using QLoRA, evaluated against a GPT-4o-mini (Azure) baseline. The goal is a portfolio project demonstrating practical LLM fine-tuning skills with benchmarked results.

**Target resume bullet:** "Fine-tuned Mistral 7B on invoice extraction dataset using QLoRA; achieved X% improvement in field extraction accuracy over GPT-4o-mini baseline, measured across 500-sample eval set."

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base model | Mistral 7B (v0.3) | Ungated, strong LoRA ecosystem, fits T4 with QLoRA |
| Fine-tuning method | QLoRA (4-bit NF4) | Fits T4 16GB VRAM on Kaggle free tier |
| Domain task | Invoice data extraction | Concrete, measurable, enterprise-relevant |
| Dataset | Existing HF datasets + synthetic generation | Best of both worlds: real-world variety + controlled quality |
| Baseline | GPT-4o-mini via Azure | Stronger baseline than GPT-3.5, more impressive to beat |
| Eval metrics | Field-level exact + fuzzy matching | Granular per-field metrics, realistic text comparison |
| Experiment tracking | Weights & Biases | Free, industry standard, shareable dashboards |
| Compute | Kaggle free tier (T4 GPU) | 30 hrs/week, stable sessions |
| Project structure | Modular Python package + Kaggle notebooks | Professional GitHub repo + reproducible notebooks |

## Project Structure

```
llm-fine-tuning/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py          # Invoice field schema (Pydantic models)
│   │   ├── existing_loader.py # Load & normalize existing HF datasets
│   │   ├── synthetic_gen.py   # Generate synthetic invoices via Claude/GPT
│   │   ├── merge.py           # Combine existing + synthetic, deduplicate
│   │   └── format.py          # Convert to instruction-tuning format
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.py          # Training hyperparameters dataclass
│   │   ├── lora_setup.py      # QLoRA config, model loading, PEFT wrapping
│   │   └── train.py           # Training loop with W&B logging
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py         # Field-level exact + fuzzy matching
│       ├── baseline.py        # Run GPT-4o-mini (Azure) on eval set
│       └── evaluate.py        # Run fine-tuned model, compare, generate report
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── data/                      # Generated/downloaded data (gitignored)
├── configs/
│   └── default.json           # Default hyperparameters
├── requirements.txt
├── .env.example               # Azure API key placeholder
├── .gitignore
└── README.md
```

## Invoice Schema

### Core Fields
- `vendor_name` (string)
- `invoice_number` (string)
- `invoice_date` (string)
- `due_date` (string)
- `total_amount` (float)
- `currency` (string)

### Additional Fields
- `tax_amount` (float)
- `discount` (float)
- `billing_address` (string)
- `payment_terms` (string)

### Line Items (list)
Each item:
- `description` (string)
- `quantity` (int/float)
- `unit_price` (float)
- `line_total` (float)

## Data Pipeline

### Existing Datasets
- Source from Hugging Face (e.g., `mychen76/invoices-and-receipts_ocr_v1` or similar)
- Normalize field names to match our schema
- Filter out low-quality examples (missing key fields, garbled OCR)

### Synthetic Generation
- Use Claude API to generate diverse, realistic invoice texts with matching JSON extractions
- Cover edge cases: multi-page invoices, partial fields, different date formats (MM/DD/YYYY, DD-Mon-YYYY, etc.), multiple currencies, varying line item counts (1-20)
- Generate in batches of ~50, with diversity prompts to avoid repetitive patterns
- Target ~1,500 synthetic examples

### Combined Dataset
- ~500 from existing datasets + ~1,500 synthetic = ~2,000 training examples
- 500 held-out eval set (balanced mix of existing + synthetic)
- Deduplicate by fuzzy matching on invoice text
- Shuffle and split with fixed random seed for reproducibility

### Instruction-Tuning Format
```json
{
  "instruction": "Extract all invoice fields from the following invoice text as JSON.",
  "input": "<invoice text here>",
  "output": "<structured JSON with all fields>"
}
```

## Training Pipeline

### Model Loading (QLoRA)
- Load Mistral 7B in 4-bit quantization (NF4) via `bitsandbytes`
- `double_quant=True` for additional memory savings on T4
- Compute dtype: `float16`

### LoRA Configuration
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Rank: 64
- Alpha: 128 (alpha/rank = 2)
- Dropout: 0.05
- Task type: `CAUSAL_LM`

### Training Hyperparameters
- Epochs: 3
- Batch size: 4 (gradient accumulation steps = 4, effective batch = 16)
- Learning rate: 2e-4 with cosine scheduler
- Warmup steps: 50
- Max sequence length: 2048
- Optimizer: paged AdamW 8-bit

### W&B Integration
- Log: training loss, learning rate, GPU memory usage
- Log eval metrics every 100 steps
- Project name: `invoice-extraction-finetune`
- Tag runs with hyperparameter configs

### Checkpointing
- Save adapter weights every 200 steps
- Keep best checkpoint based on eval loss

## Evaluation Pipeline

### Baseline (GPT-4o-mini via Azure)
- Run 500 eval examples through GPT-4o-mini with zero-shot prompt
- Prompt includes the schema definition and instruction to output JSON
- Parse JSON response, handle malformed outputs (count as failures)
- Store all predictions for comparison

### Fine-Tuned Model Inference
- Load base Mistral 7B + merged LoRA adapter
- Run same 500 eval examples with same instruction format
- Parse JSON output, handle malformed outputs

### Metrics (Per-Field)
- **Exact match** for: invoice_number, invoice_date, due_date, total_amount, currency, tax_amount, discount, payment_terms
- **Fuzzy match** (Levenshtein ratio > 0.85) for: vendor_name, billing_address, line item descriptions
- **Line items**: match by description similarity, compare quantity/unit_price/line_total per matched item
- **Aggregates**: overall field accuracy, per-field accuracy breakdown, JSON parse success rate

### Report
- Summary table: fine-tuned vs GPT-4o-mini per-field accuracy
- Overall improvement percentage
- Failure analysis: JSON parse failures, wrong field values, missing fields
- Save as markdown report + W&B summary

## End-to-End Workflow

### Step 1 — Data Prep (notebook 01)
1. Download existing HF datasets, normalize to schema
2. Generate synthetic examples via Claude API
3. Merge, deduplicate, split into train (2,000) / eval (500)
4. Format as instruction-tuning JSONL
5. Optionally upload dataset to HF Hub for Kaggle access

### Step 2 — Training (notebook 02)
1. Install dependencies, login to W&B and HF
2. Load Mistral 7B with QLoRA
3. Load formatted dataset
4. Train with SFTTrainer from `trl` library
5. Save adapter weights, optionally push to HF Hub

### Step 3 — Evaluation (notebook 03)
1. Load fine-tuned model (base + adapter)
2. Run inference on 500 eval examples
3. Run GPT-4o-mini baseline on same examples (Azure API)
4. Compute per-field metrics, generate comparison report
5. Log final metrics to W&B

## Dependencies
- `transformers`, `peft`, `bitsandbytes`, `trl` — training stack
- `datasets` — data loading
- `wandb` — experiment tracking
- `openai` — Azure GPT-4o-mini baseline calls
- `anthropic` — synthetic data generation
- `pydantic` — schema validation
- `rapidfuzz` — fuzzy string matching for eval

## Deliverables
- GitHub repo with clean, modular Python code
- 3 Kaggle notebooks (reproducible, one-click run)
- W&B dashboard with training curves and final metrics
- Markdown evaluation report with per-field comparison table
- Updated resume bullet with real benchmark numbers
