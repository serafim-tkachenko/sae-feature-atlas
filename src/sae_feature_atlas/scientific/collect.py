"""Checkpointed, native-Hugging-Face residual collection for Gemma Scope 2."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModel, AutoTokenizer

from sae_feature_atlas.config.registry import make_config
from sae_feature_atlas.pipeline.lineage import write_lineage, git_provenance
from sae_feature_atlas.runtime.loaders import load_sae
from sae_feature_atlas.util.io import write_json, write_jsonl


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def configuration(
    run_name="gemma1b_regimes_positive",
    max_texts=1000,
    max_seq_len=256,
    *,
    model="gemma-3-1b-pt",
    layer=13,
    width="16k",
    l0="medium",
    corpus="pile-10k",
):
    return make_config(
        run_name=run_name,
        max_texts=max_texts,
        max_seq_len=max_seq_len,
        activation_mode="positive",
        model=model,
        layer=layer,
        width=width,
        l0=l0,
        corpus=corpus,
    )


def collect(cfg, chunk_size=25, *, corpus_path=None, local_files_only=False):
    """Save each completed chunk; reruns require identical collection provenance.

    Native HF layer output avoids coordinate-changing TransformerLens weight
    preprocessing. Every nonfinite output is fatal, never silently skipped.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the real experiment.")
    if cfg.collection.activation_mode != "positive":
        raise ValueError("Fresh scientific collection requires all-positive storage.")
    if cfg.model.site != "resid_post":
        raise ValueError("Scientific collection currently supports resid_post only.")
    if corpus_path is None and cfg.collection.corpus != "pile-10k":
        raise ValueError("A non-Pile corpus requires an explicit prepared corpus_path.")
    corpus_hash = sha256(corpus_path) if corpus_path is not None else None
    torch.manual_seed(cfg.collection.random_seed)
    np.random.seed(cfg.collection.random_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = cfg.run_data_dir
    root.mkdir(parents=True, exist_ok=True)
    chunks = root / "collection_chunks"
    chunks.mkdir(exist_ok=True)
    api = HfApi()
    metadata_path = root / "collection_provenance.json"
    if metadata_path.exists():
        provenance = json.loads(metadata_path.read_text())
        if provenance["collection"] != asdict(cfg.collection):
            raise ValueError("Collection settings changed: use a new run name.")
        if provenance["model"] != asdict(cfg.model):
            raise ValueError("Model/SAE settings changed: use a new run name.")
        if provenance.get("prepared_corpus_sha256") != corpus_hash:
            raise ValueError("Prepared corpus changed: use a new run name.")
        if provenance["collector_sha256"] != sha256(__file__) and list(chunks.glob("*.done.json")):
            raise ValueError("Collector code changed: use a new run name.")
        provenance["collector_sha256"] = sha256(__file__)
    else:
        provenance = {
            "collection_started_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_revision": api.model_info(cfg.model.model_name).sha,
            "sae_repo": cfg.model.model_name.replace("gemma-3-", "gemma-scope-2-"),
            "sae_revision": api.model_info(
                cfg.model.model_name.replace("gemma-3-", "gemma-scope-2-")
            ).sha,
            "dataset": "prepared-jsonl" if corpus_path else "NeelNanda/pile-10k",
            "dataset_revision": None if corpus_path else api.dataset_info("NeelNanda/pile-10k").sha,
            "prepared_corpus_sha256": corpus_hash,
            "collection": asdict(cfg.collection),
            "model": asdict(cfg.model),
            "collector_sha256": sha256(__file__),
            "git": git_provenance(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "packages": {
                x: importlib.metadata.version(x)
                for x in [
                    "transformers",
                    "sae-lens",
                    "datasets",
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "pandas",
                    "pyarrow",
                    "huggingface-hub",
                ]
            },
            "collection_backend": f"native HF text backbone; layers[{cfg.model.layer}] output",
            "model_dtype": "bfloat16",
            "sae_dtype": "float32",
            "corpus_selection": "seeded permutation, unique raw texts >=300 chars; no cleanup",
        }
        write_json(metadata_path, provenance)
        (root / "collection_source.py").write_text(Path(__file__).read_text(), encoding="utf-8")
    (root / "environment_freeze.txt").write_text(
        "\n".join(
            sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name,
        revision=provenance["model_revision"],
        local_files_only=local_files_only,
    )
    if corpus_path:
        ds = [
            json.loads(line) for line in Path(corpus_path).read_text().splitlines() if line.strip()
        ]
        if len(ds) != cfg.collection.max_texts:
            raise ValueError("Prepared corpus length must exactly match max_texts.")
        manifest_path = Path(corpus_path).parent / "corpus_manifest.json"
        if not manifest_path.exists():
            raise ValueError("Prepared corpus requires adjacent corpus_manifest.json.")
        corpus_manifest = json.loads(manifest_path.read_text())
        if corpus_manifest["texts_sha256"] != corpus_hash:
            raise ValueError("Prepared corpus does not match its manifest.")
        if corpus_manifest.get("tokenizer_model", cfg.model.model_name) != cfg.model.model_name:
            raise ValueError("Corpus audit tokenizer differs from collection model.")
        if (
            corpus_manifest.get("tokenizer_revision", provenance["model_revision"])
            != provenance["model_revision"]
        ):
            raise ValueError("Model revision differs from the frozen corpus tokenizer revision.")
        if (
            corpus_manifest.get("max_length", cfg.collection.max_seq_len)
            != cfg.collection.max_seq_len
        ):
            raise ValueError("Corpus audit window differs from collection length.")
        provenance["corpus_selection"] = "prepared manifest; fixed document ordering and splits"
        write_json(root / "corpus_manifest.json", corpus_manifest)
    else:
        ds = load_dataset(
            provenance["dataset"], revision=provenance["dataset_revision"], split="train"
        )
    rng = np.random.default_rng(cfg.collection.random_seed)
    texts, seen = [], set()
    for idx in range(len(ds)) if corpus_path else rng.permutation(len(ds)):
        raw = ds[int(idx)]["text"]
        if len(raw) < 300 or raw in seen:
            continue
        seen.add(raw)
        item = (
            dict(ds[int(idx)])
            if corpus_path
            else {"source": "pile-10k", "dataset_row": int(idx), "text": raw}
        )
        item["text_id"] = len(texts)
        texts.append(item)
        if len(texts) == cfg.collection.max_texts:
            break
    if len(texts) != cfg.collection.max_texts:
        raise ValueError("Requested more eligible unique documents than available.")
    write_jsonl(cfg.source_texts_path, texts)
    # Pin and verify native SAE files against the registry-loaded tensors below.
    subfolder = "resid_post/" + cfg.model.sae_id
    native_cfg_path = hf_hub_download(
        provenance["sae_repo"], subfolder + "/config.json", revision=provenance["sae_revision"]
    )
    native = json.loads(Path(native_cfg_path).read_text())
    expected_hook = f"model.layers.{cfg.model.layer}.output"
    if native["hf_hook_point_in"] != expected_hook or native["hf_hook_point_out"] != expected_hook:
        raise ValueError("Unexpected SAE hook; revise the collector explicitly.")
    if native["model_name"] != cfg.model.model_name:
        raise ValueError("Native SAE model does not match the configured model.")
    # SAE Lens resolves this release to the same pinned snapshot. Verify bytes.
    sae = load_sae(cfg, "cuda")
    from safetensors.torch import load_file

    weights_path = hf_hub_download(
        provenance["sae_repo"],
        subfolder + "/params.safetensors",
        revision=provenance["sae_revision"],
    )
    native_weights = load_file(weights_path)
    for key, value in native_weights.items():
        attr = {"w_enc": "W_enc", "w_dec": "W_dec"}.get(key, key)
        if not torch.equal(getattr(sae, attr).detach().cpu(), value):
            raise ValueError(f"Registry SAE differs from pinned native checkpoint: {key}")
    del native_weights
    provenance.update(
        sae_native_config=native,
        sae_weights_sha256=sha256(weights_path),
        sae_config=str(sae.cfg),
        text_count=len(texts),
        source_texts_sha256=sha256(cfg.source_texts_path),
    )
    write_json(metadata_path, provenance)
    model = AutoModel.from_pretrained(
        cfg.model.model_name,
        revision=provenance["model_revision"],
        dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    # Multimodal Gemma 3 checkpoints wrap the same text backbone. Drop the unused
    # vision tower before transfer; preserve native text weights and coordinates.
    if hasattr(model, "language_model"):
        model = model.language_model
    if not hasattr(model, "layers"):
        raise ValueError("Unrecognized native text backbone; refusing a guessed hook.")
    model = model.to("cuda").eval()
    capture = {}

    def hook(_module, _args, output):
        capture["resid"] = output[0] if isinstance(output, tuple) else output

    handle = model.layers[cfg.model.layer].register_forward_hook(hook)
    smoke = []
    for start in range(0, len(texts), chunk_size):
        done = chunks / f"{start:06d}.done.json"
        if done.exists():
            saved = json.loads(done.read_text())
            if any(sha256(chunks / name) != digest for name, digest in saved.items()):
                raise ValueError("Corrupt collection chunk.")
            continue
        acts, tokens_out, summaries = [], [], []
        for item in texts[start : start + chunk_size]:
            inputs = tokenizer(
                item["text"],
                return_tensors="pt",
                truncation=True,
                max_length=cfg.collection.max_seq_len,
            ).to("cuda")
            with torch.inference_mode():
                model(**inputs, use_cache=False)
                resid = capture.pop("resid").float()
                a = sae.encode(resid)[0]
                if not torch.isfinite(resid).all() or not torch.isfinite(a).all():
                    raise ValueError(f"Nonfinite activations in text {item['text_id']}")
                if a.shape[-1] != native["width"] or resid.shape[-1] != sae.cfg.d_in:
                    raise ValueError("SAE dimension mismatch.")
                if not smoke:
                    reconstruction = sae.decode(a.unsqueeze(0))
                    smoke.append(
                        {
                            "text_id": item["text_id"],
                            "mean_l0": float((a > 0).sum(-1).float().mean()),
                            "relative_reconstruction_mse": float(
                                (resid - reconstruction).square().mean() / resid.square().mean()
                            ),
                            "residual_shape": list(resid.shape),
                            "sae_shape": list(a.shape),
                        }
                    )
                    write_json(root / "sanity_check.json", smoke[0])
                    if smoke[0]["mean_l0"] <= 0 or smoke[0]["relative_reconstruction_mse"] > 1:
                        raise ValueError(
                            "Smoke validation failed: dead SAE or reconstruction error >1."
                        )
                    print("SANITY", smoke[0], flush=True)
                pos, fid = torch.nonzero(a > 0, as_tuple=True)
                av = a[pos, fid].cpu().numpy()
                pos, fid = pos.cpu().numpy(), fid.cpu().numpy()
                ids = inputs["input_ids"][0].cpu().tolist()
                strs = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in ids]
                acts.append(
                    pd.DataFrame(
                        {
                            "text_id": np.int32(item["text_id"]),
                            "token_pos": pos.astype("int16"),
                            "feature_id": fid.astype("int32"),
                            "activation": av,
                        }
                    )
                )
                tokens_out.append(
                    pd.DataFrame(
                        {
                            "text_id": item["text_id"],
                            "token_pos": np.arange(len(ids)),
                            "token_id": ids,
                            "token_str": strs,
                            "source": item["source"],
                            **({"split": item["split"]} if "split" in item else {}),
                            **(
                                {"duplicate_group": item["duplicate_group"]}
                                if "duplicate_group" in item
                                else {}
                            ),
                        }
                    )
                )
                summaries.append(
                    pd.DataFrame(
                        {
                            "text_id": item["text_id"],
                            "token_pos": np.arange(len(ids)),
                            "n_positive_features": (a > 0).sum(-1).cpu().numpy(),
                        }
                    )
                )
        hashes = {}
        for kind, frames in [("acts", acts), ("tokens", tokens_out), ("support", summaries)]:
            path = chunks / f"{start:06d}.{kind}.parquet"
            pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
            hashes[path.name] = sha256(path)
        write_json(done, hashes)
        print(f"COLLECTED {min(start + chunk_size, len(texts))}/{len(texts)} documents", flush=True)
    handle.remove()
    del model, sae
    gc.collect()
    torch.cuda.empty_cache()
    for kind, path in [
        ("acts", cfg.sae_activations_path),
        ("tokens", cfg.token_metadata_path),
        ("support", cfg.token_activation_summary_path),
    ]:
        # Arrow streaming keeps finalization independent of the whole collection size.
        import pyarrow.parquet as pq

        writer = None
        try:
            for part in sorted(chunks.glob(f"*.{kind}.parquet")):
                table = pq.read_table(part)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
    write_lineage(cfg, "collect")
    return provenance
