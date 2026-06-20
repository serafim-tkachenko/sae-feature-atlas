
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from sae_feature_atlas.config.schema import ExperimentConfig
from sae_feature_atlas.util.io import read_jsonl, write_jsonl


@dataclass(frozen=True)
class CorpusDescriptor:
    """Human-readable corpus preset metadata.

    The loader name is intentionally separate from the Hugging Face dataset name:
    users should be able to discover corpus presets without reading source code.
    """

    name: str
    description: str
    domains: tuple[str, ...]
    size: str
    streaming: bool
    hf_dataset: str | None = None
    hf_config: str | None = None
    notes: str = ""


MANUAL_TEXTS: list[tuple[str, str]] = [
    ("manual_code", "Python functions, classes, loops, and dictionaries organize computation."),
    ("manual_science", "In quantum mechanics, measurement changes the state of a physical system."),
    ("manual_finance", "The company reported higher revenue but lower operating margins."),
    ("manual_health", "A healthy training plan balances strength, endurance, mobility, and recovery."),
    ("manual_assistant", "The user asked the assistant to explain the concept step by step."),
    ("manual_math", "Linear algebra studies vectors, matrices, eigenvalues, and projections."),
    ("manual_security", "The phishing email asked the recipient to reset their password."),
    ("manual_literature", "The old house stood at the edge of the forest.")]


CORPUS_REGISTRY: dict[str, CorpusDescriptor] = {
    "pile-10k": CorpusDescriptor(
        name="pile-10k",
        description="Small general-purpose sample from The Pile.",
        domains=("web", "books", "code", "academic", "misc"),
        size="small",
        streaming=False,
        hf_dataset="NeelNanda/pile-10k",
        notes="Good smoke-test corpus; not broad enough for final claims by itself.",
    ),
    "tinystories": CorpusDescriptor(
        name="tinystories",
        description="Simple narrative text; useful for story/grammar features.",
        domains=("fiction", "stories", "simple language"),
        size="large-streaming",
        streaming=True,
        hf_dataset="roneneldan/TinyStories",
    ),
    "fineweb-edu-sample": CorpusDescriptor(
        name="fineweb-edu-sample",
        description="Educational web text sample from FineWeb-Edu.",
        domains=("web", "education", "explanatory text"),
        size="large-streaming",
        streaming=True,
        hf_dataset="HuggingFaceFW/fineweb-edu",
        hf_config="sample-10BT",
    ),
    "wikimedia-en": CorpusDescriptor(
        name="wikimedia-en",
        description="Wikipedia-style encyclopedic text.",
        domains=("encyclopedic", "factual", "long-form"),
        size="large-streaming",
        streaming=True,
        hf_dataset="wikimedia/wikipedia",
        hf_config="20231101.en",
    ),
    "openwebtext": CorpusDescriptor(
        name="openwebtext",
        description="OpenWebText-style internet text; useful as a broad web baseline.",
        domains=("web", "internet", "general"),
        size="large-streaming",
        streaming=True,
        hf_dataset="Skylion007/openwebtext",
        notes="Dataset availability may vary; loader falls back to other corpora in mixed presets.",
    ),
    "math-small": CorpusDescriptor(
        name="math-small",
        description="Small synthetic/manual math-heavy corpus for targeted probing.",
        domains=("math", "linear algebra", "probability", "optimization"),
        size="tiny-manual",
        streaming=False,
        notes="Good for qualitative probing only, not objective corpus-level claims.",
    ),
    "code-small": CorpusDescriptor(
        name="code-small",
        description="Small synthetic/manual code-heavy corpus for targeted probing.",
        domains=("python", "programming", "software"),
        size="tiny-manual",
        streaming=False,
        notes="Good for qualitative probing only, not objective corpus-level claims.",
    ),
    "mixed-research": CorpusDescriptor(
        name="mixed-research",
        description="Document-balanced small/medium research mix across Pile and FineWeb-Edu.",
        domains=("general", "web", "education", "manual probes"),
        size="medium",
        streaming=True,
        notes="Recommended default for pilot runs. Document shares are fixed before tokenization; token shares may differ.",
    ),
    "mixed-broad": CorpusDescriptor(
        name="mixed-broad",
        description="Document-balanced broad mix across Pile, FineWeb-Edu, Wikipedia, TinyStories, and small manual probes.",
        domains=("general", "web", "encyclopedic", "stories", "code", "math"),
        size="medium-large",
        streaming=True,
        notes="Manual math/code rows are qualitative probes, not representative 5% slices. Inspect saved source summaries before making corpus-level claims.",
    ),
    "mixed-large": CorpusDescriptor(
        name="mixed-large",
        description="Larger document-balanced broad corpus preset for heavier research runs.",
        domains=("general", "web", "encyclopedic", "stories", "education"),
        size="large",
        streaming=True,
        notes="Use with higher max_texts/max_seq_len and expect longer collection time. Document shares are fixed before tokenization; token shares may differ.",
    ),
}


MATH_TEXTS = [
    "Eigenvectors define directions that are only scaled by a linear transformation.",
    "Gradient descent updates parameters in the opposite direction of the loss gradient.",
    "A covariance matrix describes how pairs of variables vary together.",
    "Principal component analysis rotates data into directions of maximum variance.",
    "Conditional probability measures the probability of an event after observing evidence."]

CODE_TEXTS = [
    "A Python decorator wraps a function and can modify its behavior before returning it.",
    "Hash maps provide average constant-time lookup for keys when collisions are controlled.",
    "A unit test should isolate one behavior and assert the expected output clearly.",
    "Vectorized NumPy operations avoid slow Python loops over individual elements.",
    "A command-line interface should validate arguments before launching expensive jobs."]


def list_supported_corpora() -> list[str]:
    return sorted(CORPUS_REGISTRY)


def describe_corpus(name: str) -> CorpusDescriptor:
    if name.startswith("jsonl:"):
        return CorpusDescriptor(
            name=name,
            description="Custom JSONL corpus with at least a text field.",
            domains=("custom",),
            size="custom",
            streaming=False,
            notes="Rows may optionally contain a source field.",
        )
    try:
        return CORPUS_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported corpus {name!r}. Supported: {', '.join(list_supported_corpora())}, jsonl:/path"
        ) from exc


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


def _keep_text(text: str, min_chars: int = 200) -> bool:
    return bool(text and len(text) >= min_chars)


def _load_hf_text_dataset(
    *,
    source: str,
    dataset_name: str,
    max_items: int,
    config_name: str | None = None,
    split: str = "train",
    text_field: str = "text",
    streaming: bool = True,
    min_chars: int = 200,
) -> list[dict]:
    if max_items <= 0:
        return []
    rows: list[dict] = []
    kwargs = {"split": split, "streaming": streaming}
    if config_name is not None:
        ds = load_dataset(dataset_name, name=config_name, **kwargs)
    else:
        ds = load_dataset(dataset_name, **kwargs)
    for row in ds:
        text = clean_text(row.get(text_field, "")) if isinstance(row, dict) else ""
        if _keep_text(text, min_chars=min_chars):
            rows.append({"source": source, "text": text})
        if len(rows) >= max_items:
            break
    return rows


def _load_pile_10k(max_items: int) -> list[dict]:
    return _load_hf_text_dataset(
        source="pile-10k",
        dataset_name="NeelNanda/pile-10k",
        max_items=max_items,
        streaming=False,
        min_chars=300,
    )


def _load_tinystories(max_items: int) -> list[dict]:
    return _load_hf_text_dataset(
        source="tinystories",
        dataset_name="roneneldan/TinyStories",
        max_items=max_items,
        streaming=True,
        min_chars=200,
    )


def _load_fineweb_edu_sample(max_items: int) -> list[dict]:
    return _load_hf_text_dataset(
        source="fineweb-edu-sample",
        dataset_name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        max_items=max_items,
        streaming=True,
        min_chars=300,
    )


def _load_wikimedia_en(max_items: int) -> list[dict]:
    return _load_hf_text_dataset(
        source="wikimedia-en",
        dataset_name="wikimedia/wikipedia",
        config_name="20231101.en",
        max_items=max_items,
        streaming=True,
        min_chars=300,
    )


def _load_openwebtext(max_items: int) -> list[dict]:
    return _load_hf_text_dataset(
        source="openwebtext",
        dataset_name="Skylion007/openwebtext",
        max_items=max_items,
        streaming=True,
        min_chars=300,
    )


def _load_jsonl(corpus: str, max_items: int) -> list[dict]:
    path = corpus.removeprefix("jsonl:")
    raw = read_jsonl(path)[:max_items]
    rows = []
    for item in raw:
        text = clean_text(item.get("text", ""))
        if text:
            rows.append({"source": item.get("source", "custom-jsonl"), "text": text})
    return rows


def _manual_rows(name: str, texts: list[str]) -> list[dict]:
    return [{"source": name, "text": clean_text(text)} for text in texts]


def _append_manual_probe_texts(texts: list[dict]) -> list[dict]:
    for source, text in MANUAL_TEXTS:
        texts.append({"source": source, "text": clean_text(text)})
    return texts


def _summary_sidecar_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}.csv")


def summarize_text_sources(texts: list[dict]) -> pd.DataFrame:
    """Return document counts and shares by source for a loaded text corpus."""

    total = len(texts)
    counts = Counter(str(item.get("source", "unknown")) for item in texts)
    rows = [
        {"source": source, "texts": int(count), "text_share": count / total if total else 0.0}
        for source, count in counts.most_common()
    ]
    return pd.DataFrame(rows, columns=["source", "texts", "text_share"])


def summarize_token_sources(token_meta: pd.DataFrame) -> pd.DataFrame:
    """Return post-tokenization token counts and shares by source."""

    columns = ["source", "texts", "tokens", "token_share"]
    if token_meta.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        token_meta.groupby("source", dropna=False)
        .agg(texts=("text_id", "nunique"), tokens=("token_id", "size"))
        .reset_index()
    )
    total_tokens = int(summary["tokens"].sum())
    summary["token_share"] = summary["tokens"] / total_tokens if total_tokens else 0.0
    return summary.sort_values(["tokens", "source"], ascending=[False, True]).reset_index(drop=True)[columns]


def _safe_extend(texts: list[dict], loader: Callable[[int], list[dict]], n: int, fallback: Callable[[int], list[dict]] | None = None) -> None:
    if n <= 0:
        return
    try:
        texts.extend(loader(n))
    except Exception as exc:
        if fallback is None:
            print(f"Corpus loader failed ({loader.__name__}, {exc!r}); skipping this slice.")
            return
        print(f"Corpus loader failed ({loader.__name__}, {exc!r}); falling back to {fallback.__name__}.")
        texts.extend(fallback(n))


def _load_mixed_research(max_items: int) -> list[dict]:
    pile_n = max_items // 2
    fineweb_n = max_items - pile_n
    texts = _load_pile_10k(pile_n)
    _safe_extend(texts, _load_fineweb_edu_sample, fineweb_n, fallback=_load_pile_10k)
    return texts


def _load_mixed_broad(max_items: int) -> list[dict]:
    texts: list[dict] = []
    slices = [
        (_load_pile_10k, max_items * 25 // 100),
        (_load_fineweb_edu_sample, max_items * 25 // 100),
        (_load_wikimedia_en, max_items * 20 // 100),
        (_load_tinystories, max_items * 15 // 100),
        (lambda n: _manual_rows("manual_math", MATH_TEXTS)[:n], max(1, max_items * 5 // 100)),
        (lambda n: _manual_rows("manual_code", CODE_TEXTS)[:n], max(1, max_items * 5 // 100))]
    used = 0
    for loader, n in slices:
        n = min(n, max_items - used)
        _safe_extend(texts, loader, n, fallback=_load_pile_10k)
        used = len(texts)
        if used >= max_items:
            break
    if len(texts) < max_items:
        _safe_extend(texts, _load_pile_10k, max_items - len(texts))
    return texts[:max_items]


def _load_mixed_large(max_items: int) -> list[dict]:
    texts: list[dict] = []
    slices = [
        (_load_fineweb_edu_sample, max_items * 35 // 100),
        (_load_wikimedia_en, max_items * 25 // 100),
        (_load_pile_10k, max_items * 20 // 100),
        (_load_tinystories, max_items * 10 // 100),
        (_load_openwebtext, max_items * 10 // 100)]
    for loader, n in slices:
        _safe_extend(texts, loader, n, fallback=_load_pile_10k)
    if len(texts) < max_items:
        _safe_extend(texts, _load_pile_10k, max_items - len(texts))
    return texts[:max_items]


LOADERS: dict[str, Callable[[int], list[dict]]] = {
    "pile-10k": _load_pile_10k,
    "tinystories": _load_tinystories,
    "fineweb-edu-sample": _load_fineweb_edu_sample,
    "wikimedia-en": _load_wikimedia_en,
    "openwebtext": _load_openwebtext,
    "math-small": lambda n: _manual_rows("manual_math", MATH_TEXTS)[:n],
    "code-small": lambda n: _manual_rows("manual_code", CODE_TEXTS)[:n],
    "mixed-research": _load_mixed_research,
    "mixed-broad": _load_mixed_broad,
    "mixed-large": _load_mixed_large,
}


def build_text_dataset(cfg: ExperimentConfig) -> list[dict]:
    corpus = cfg.collection.corpus
    max_items = max(cfg.collection.max_texts - len(MANUAL_TEXTS), 0)

    if corpus.startswith("jsonl:"):
        texts = _load_jsonl(corpus, max_items)
    else:
        loader = LOADERS.get(corpus)
        if loader is None:
            raise ValueError(
                f"Unsupported corpus {corpus!r}. Supported: {', '.join(list_supported_corpora())}, jsonl:/path"
            )
        texts = loader(max_items)

    _append_manual_probe_texts(texts)
    texts = texts[: cfg.collection.max_texts]
    for i, item in enumerate(texts):
        item["text_id"] = i
    return texts


def save_text_dataset(cfg: ExperimentConfig, texts: list[dict]) -> None:
    raw_texts_path = Path(cfg.paths.raw_texts_path)
    raw_texts_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(raw_texts_path, texts)
    summarize_text_sources(texts).to_csv(_summary_sidecar_path(raw_texts_path, "source_summary"), index=False)


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
    summarize_token_sources(token_meta).to_csv(_summary_sidecar_path(cfg.token_metadata_path, "source_summary"), index=False)
    return token_meta
