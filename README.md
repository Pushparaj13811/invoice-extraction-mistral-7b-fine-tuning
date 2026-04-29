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
