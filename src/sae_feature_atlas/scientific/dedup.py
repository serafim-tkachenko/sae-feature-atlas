"""MinHash candidate audit of the actual token windows, with exact Jaccard checks."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from sae_feature_atlas.scientific.collect import sha256
from sae_feature_atlas.util.io import write_json, write_jsonl


def audit(root, model="google/gemma-3-4b-pt", max_length=512, seed=20260908):
    root = Path(root)
    manifest = json.loads((root / "corpus_manifest.json").read_text())
    path = root / "texts.jsonl"
    if manifest["texts_sha256"] != sha256(path):
        raise ValueError("Corpus hash mismatch before duplicate audit.")
    if manifest.get("near_duplicate_audit") != "pending; not cleared for confirmatory evaluation":
        raise ValueError("Corpus already audited; use its frozen manifest.")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    revision = HfApi().model_info(model).sha
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**31 - 1, 64, dtype=np.uint64)
    b = rng.integers(0, 2**31 - 1, 64, dtype=np.uint64)
    buckets, shingles, edges = defaultdict(list), [], []
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    exact_windows = {}
    for i, row in enumerate(rows):
        ids = tokenizer(row["text"], truncation=True, max_length=max_length)["input_ids"]
        window = tuple(ids)
        if window in exact_windows:
            j = exact_windows[window]
            parent[find(i)] = find(j)
            edges.append({"left": j, "right": i, "jaccard": 1.0, "kind": "exact_window"})
        else:
            exact_windows[window] = i
        grams = {tuple(ids[k : k + 5]) for k in range(max(0, len(ids) - 4))}
        shingles.append(grams)
        if not grams:
            continue
        values = np.array(
            [
                int.from_bytes(
                    hashlib.blake2b(np.asarray(g, dtype="<i4").tobytes(), digest_size=4).digest(),
                    "little",
                )
                for g in grams
            ],
            dtype=np.uint64,
        )
        sig = ((values[:, None] * a + b) % np.uint64(2**61 - 1)).min(axis=0)
        candidates = set()
        keys = [(band, tuple(sig[band * 4 : band * 4 + 4].tolist())) for band in range(16)]
        for key in keys:
            candidates.update(buckets[key])
            buckets[key].append(i)
        for j in sorted(candidates):
            similarity = len(grams & shingles[j]) / len(grams | shingles[j])
            if similarity >= 0.8:
                parent[find(i)] = find(j)
                edges.append({"left": j, "right": i, "jaccard": similarity, "kind": "minhash"})
        if (i + 1) % 1000 == 0:
            print(f"DUPLICATE AUDIT {i + 1}/{len(rows)}", flush=True)
    # Hash each connected group into a split. Approximate balance is intentional;
    # connected documents can never cross the discovery/evaluation boundary.
    for i, row in enumerate(rows):
        group = find(i)
        digest = hashlib.sha256(f"{seed}:{rows[group]['text_sha256']}".encode()).digest()
        row["duplicate_group"] = group
        row["split"] = "discovery" if digest[0] < 128 else "evaluation"
    write_jsonl(path, rows)
    write_json(
        root / "duplicate_audit.json",
        {
            "method": "token 5-shingles, 64 MinHash permutations, 16 bands of 4; exact Jaccard >=0.8",
            "limitation": "Approximate candidate retrieval can miss near duplicates; no exhaustive guarantee.",
            "model": model,
            "model_revision": revision,
            "max_length": max_length,
            "seed": seed,
            "edges": edges,
            "groups": len({find(i) for i in range(len(rows))}),
        },
    )
    manifest.update(
        texts_sha256=sha256(path),
        near_duplicate_audit="completed; approximate candidate retrieval",
        duplicate_audit_sha256=sha256(root / "duplicate_audit.json"),
        split_method="seeded group hash; approximately half per source",
        tokenizer_model=model,
        tokenizer_revision=revision,
        max_length=max_length,
    )
    write_json(root / "corpus_manifest.json", manifest)
    print(f"AUDIT COMPLETE: {len(edges)} links; {manifest['texts_sha256']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    audit(args.root)
