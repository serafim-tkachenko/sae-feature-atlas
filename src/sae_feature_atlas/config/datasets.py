from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from sae_feature_atlas.config.schema import ExperimentConfig
from sae_feature_atlas.util.io import read_jsonl, write_jsonl

MANUAL_TEXTS: list[tuple[str, str]] = [
    ("manual_code", "Python functions, classes, loops, and dictionaries organize computation."),
    ("manual_science", "In quantum mechanics, measurement changes the state of a physical system."),
    ("manual_finance", "The company reported higher revenue but lower operating margins."),
    ("manual_health", "A healthy training plan balances strength, endurance, mobility, and recovery."),
    ("manual_assistant", "The user asked the assistant to explain the concept step by step."),
    ("manual_math", "Linear algebra studies vectors, matrices, eigenvalues, and projections."),
    ("manual_security", "The phishing email asked the recipient to reset their password."),
    ("manual_literature", "The old house stood at the edge of the forest."),
]


def clean_text(text: str) -> str:
    return (
        str(text)
        .replace("â€™", "'")
        .replace("â€œ", '"')
        .replace("â€\x9d", '"')
        .replace("â€“", "-")
        .replace("â€”", "-")
        .strip()
    )


def _load_pile_10k(max_items: int) -> list[dict]:
    rows = []
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    for row in ds:
        text = clean_text(row["text"])
        if len(text) > 300:
            rows.append({"source": "pile-10k", "text": text})
        if len(rows) >= max_items:
            break
    return rows


def _load_tinystories(max_items: int) -> list[dict]:
    rows = []
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for row in ds:
        text = clean_text(row["text"])
        if len(text) > 200:
            rows.append({"source": "tinystories", "text": text})
        if len(rows) >= max_items:
            break
    return rows


def _load_fineweb_edu_sample(max_items: int) -> list[dict]:
    rows = []
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for row in ds:
        text = clean_text(row.get("text", ""))
        if len(text) > 300:
            rows.append({"source": "fineweb-edu-sample", "text": text})
        if len(rows) >= max_items:
            break
    return rows


def _load_jsonl(corpus: str, max_items: int) -> list[dict]:
    path = corpus.removeprefix("jsonl:")
    raw = read_jsonl(path)[:max_items]
    rows = []
    for item in raw:
        text = clean_text(item.get("text", ""))
        if text:
            rows.append({"source": item.get("source", "custom-jsonl"), "text": text})
    return rows


def build_text_dataset(cfg: ExperimentConfig) -> list[dict]:
    corpus = cfg.collection.corpus
    max_items = max(cfg.collection.max_texts - len(MANUAL_TEXTS), 0)

    if corpus == "pile-10k":
        texts = _load_pile_10k(max_items)
    elif corpus == "tinystories":
        texts = _load_tinystories(max_items)
    elif corpus == "fineweb-edu-sample":
        texts = _load_fineweb_edu_sample(max_items)
    elif corpus == "mixed-research":
        pile_n = max_items // 2
        fineweb_n = max_items - pile_n
        texts = _load_pile_10k(pile_n)
        try:
            texts.extend(_load_fineweb_edu_sample(fineweb_n))
        except Exception as exc:
            print(f"FineWeb-Edu streaming failed ({exc!r}); falling back to more pile-10k rows.")
            texts.extend(_load_pile_10k(fineweb_n))
    elif corpus.startswith("jsonl:"):
        texts = _load_jsonl(corpus, max_items)
    else:
        raise ValueError(
            f"Unsupported corpus {corpus!r}. Supported: pile-10k, tinystories, "
            "fineweb-edu-sample, mixed-research, jsonl:/path"
        )

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
        tokens = model.to_tokens(item["text"], truncate=True, prepend_bos=True)[:, : cfg.collection.max_seq_len].to(device)
        token_ids = tokens[0].detach().cpu().tolist()
        token_strs = model.to_str_tokens(tokens[0])
        for pos, (token_id, token_str) in enumerate(zip(token_ids, token_strs)):
            token_rows.append(
                {
                    "text_id": int(item["text_id"]),
                    "source": item["source"],
                    "token_pos": int(pos),
                    "token_id": int(token_id),
                    "token_str": token_str,
                }
            )
    token_meta = pd.DataFrame(token_rows)
    cfg.token_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    token_meta.to_parquet(cfg.token_metadata_path, index=False)
    return token_meta
