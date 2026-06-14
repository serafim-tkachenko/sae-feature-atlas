from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from sae_feature_atlas.config import ExperimentConfig
from sae_feature_atlas.io_utils import write_jsonl


# Manual texts to extend the first corpus - will be changed in the next updates
MANUAL_TEXTS: list[tuple[str, str]] = [
    (
        "code",
        "Python functions, classes, loops, and dictionaries are used to organize "
        "computation and data transformations.",
    ),
    (
        "science",
        "In quantum mechanics, measurement changes the state of a physical system "
        "and produces probabilistic outcomes.",
    ),
    (
        "finance",
        "The company reported higher revenue but lower operating margins due to "
        "rising operational costs.",
    ),
    (
        "health",
        "A healthy training plan balances strength, endurance, mobility, nutrition, "
        "and recovery.",
    ),
    (
        "assistant",
        "The user asked the assistant to explain the concept step by step using "
        "simple examples.",
    ),
    (
        "math",
        "Linear algebra studies vectors, matrices, eigenvalues, projections, and "
        "transformations between vector spaces.",
    ),
    (
        "security",
        "The phishing email asked the recipient to reset their password using a "
        "suspicious external link.",
    ),
    (
        "literature",
        "The old house stood at the edge of the forest, its windows reflecting the "
        "evening light.",
    ),
]


def clean_text(text: str) -> str:
    # Conservative cleanup for common artifacts observed in TinyStories
    return (
        text.replace("â€™", "'")
        .replace("â€œ", '"')
        .replace("â€\x9d", '"')
        .replace("â€“", "-")
        .replace("â€”", "-")
        .strip()
    )


def build_text_dataset(cfg: ExperimentConfig) -> list[dict]:
    texts: list[dict] = []

    # More diverse than TinyStories
    pile = load_dataset("NeelNanda/pile-10k", split="train")

    target_from_pile = max(cfg.collection.max_texts - len(MANUAL_TEXTS), 0)

    for row in pile:
        text = clean_text(row["text"])
        if len(text) > 300:
            texts.append({"source": "pile-10k", "text": text})
        if len(texts) >= target_from_pile:
            break

    for source, text in MANUAL_TEXTS:
        texts.append({"source": source, "text": clean_text(text)})

    texts = texts[: cfg.collection.max_texts]

    for i, item in enumerate(texts):
        item["text_id"] = i

    return texts


def save_text_dataset(cfg: ExperimentConfig, texts: list[dict]) -> None:
    Path(cfg.paths.raw_texts_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(cfg.paths.raw_texts_path, texts)


def build_token_metadata(model, cfg: ExperimentConfig, texts: list[dict], device: str) -> pd.DataFrame:
    token_rows: list[dict] = []

    for item in tqdm(texts, desc="Tokenizing texts"):
        text_id = item["text_id"]
        source = item["source"]
        text = item["text"]

        tokens = model.to_tokens(
            text,
            truncate=True,
            prepend_bos=True,
        )[:, : cfg.collection.max_seq_len].to(device)

        token_ids = tokens[0].detach().cpu().tolist()
        token_strs = model.to_str_tokens(tokens[0])

        for pos, (token_id, token_str) in enumerate(zip(token_ids, token_strs)):
            token_rows.append(
                {
                    "text_id": int(text_id),
                    "source": source,
                    "token_pos": int(pos),
                    "token_id": int(token_id),
                    "token_str": token_str,
                }
            )

    token_meta = pd.DataFrame(token_rows)
    cfg.token_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    token_meta.to_parquet(cfg.token_metadata_path, index=False)
    return token_meta
