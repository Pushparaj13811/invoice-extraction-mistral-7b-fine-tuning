from __future__ import annotations

import random

from rapidfuzz import fuzz

from src.data.schema import Invoice


def deduplicate(
    data: list[tuple[str, Invoice]],
    threshold: float = 0.90,
) -> list[tuple[str, Invoice]]:
    kept: list[tuple[str, Invoice]] = []
    kept_texts: list[str] = []

    for text, invoice in data:
        is_dup = False
        for existing_text in kept_texts:
            # Skip fuzzy comparison for very short strings to avoid false positives;
            # require exact match instead.
            min_len = min(len(text), len(existing_text))
            if min_len < 20:
                if text == existing_text:
                    is_dup = True
                    break
            elif fuzz.ratio(text, existing_text) >= threshold * 100:
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
    combined = deduplicate(existing + synthetic)

    rng = random.Random(seed)
    rng.shuffle(combined)

    eval_size = min(eval_size, len(combined) // 4)
    eval_set = combined[:eval_size]
    train_set = combined[eval_size:]

    return train_set, eval_set
