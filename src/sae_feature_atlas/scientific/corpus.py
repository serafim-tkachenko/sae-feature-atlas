"""Fail-closed, revision-pinned sampling from configured research corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi

from sae_feature_atlas.config.datasets import CORPUS_REGISTRY
from sae_feature_atlas.util.io import write_json, write_jsonl


def text_digest(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def select_documents(rows, count, source, seen):
    """Preserve raw text; normalize only for global duplicate detection."""
    selected = []
    rejected = {"short_or_invalid": 0, "duplicate": 0}
    for ordinal, row in enumerate(rows):
        raw = row.get("text")
        if not isinstance(raw, str) or len(raw.strip()) < 300:
            rejected["short_or_invalid"] += 1
            continue
        digest = text_digest(raw)
        if digest in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(digest)
        selected.append(
            {
                "source": source,
                "text": raw,
                "text_sha256": digest,
                "stream_ordinal": ordinal,
                "source_id": str(row.get("id", row.get("url", ordinal))),
            }
        )
        if len(selected) == count:
            return selected, rejected
    raise ValueError(f"{source}: only {len(selected)} eligible documents; required {count}.")


def prepare(
    output,
    sources=("fineweb-edu-sample", "wikimedia-en"),
    per_source=6000,
    seed=20260908,
    shuffle_buffer=10000,
    exclude_paths=(),
):
    if per_source < 2 or shuffle_buffer < 1:
        raise ValueError("Need >=2 documents per source and a positive shuffle buffer.")
    if len(set(sources)) != len(sources):
        raise ValueError("Duplicate corpus sources.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "corpus_manifest.json").exists() or (output / "texts.jsonl").exists():
        raise ValueError("Corpus output already exists; use a new directory.")
    api = HfApi()
    manifest = {
        "seed": seed,
        "per_source": per_source,
        "shuffle_buffer": shuffle_buffer,
        "sampling": "seeded finite-buffer streaming shuffle; not full-corpus uniform",
        "deduplication": "global exact NFKC/whitespace-normalized SHA256",
        "near_duplicate_audit": "pending; not cleared for confirmatory evaluation",
        "sources": [],
    }
    all_rows, seen = [], set()
    manifest["excluded_corpora"] = []
    for excluded_path in exclude_paths:
        excluded_path = Path(excluded_path)
        for line in excluded_path.read_text().splitlines():
            if line.strip():
                seen.add(text_digest(json.loads(line)["text"]))
        manifest["excluded_corpora"].append(
            {
                "sha256": hashlib.sha256(excluded_path.read_bytes()).hexdigest(),
                "path": str(excluded_path),
            }
        )
    for index, source in enumerate(sources):
        descriptor = CORPUS_REGISTRY[source]
        if descriptor.hf_dataset is None:
            raise ValueError(f"{source} is not a concrete Hugging Face dataset.")
        revision = api.dataset_info(descriptor.hf_dataset).sha
        ds = load_dataset(
            descriptor.hf_dataset,
            name=descriptor.hf_config,
            revision=revision,
            split="train",
            streaming=True,
        )
        rows, rejected = select_documents(
            ds.shuffle(seed=seed + index, buffer_size=shuffle_buffer), per_source, source, seen
        )
        order = np.random.default_rng(seed + index).permutation(len(rows))
        discovery = set(order[: len(rows) // 2].tolist())
        for i, row in enumerate(rows):
            row.update(text_id=len(all_rows), split="discovery" if i in discovery else "evaluation")
            all_rows.append(row)
        manifest["sources"].append(
            {
                "name": source,
                "dataset": descriptor.hf_dataset,
                "config": descriptor.hf_config,
                "revision": revision,
                "count": len(rows),
                "rejected": rejected,
            }
        )
        print(f"CORPUS {source}: {len(rows)} documents; exclusions {rejected}", flush=True)
    path = output / "texts.jsonl"
    write_jsonl(path, all_rows)
    manifest["texts_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / "corpus_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sources", nargs="+", default=["fineweb-edu-sample", "wikimedia-en"])
    parser.add_argument("--per-source", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--exclude", nargs="*", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.output,
                args.sources,
                args.per_source,
                args.seed,
                args.shuffle_buffer,
                args.exclude,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
