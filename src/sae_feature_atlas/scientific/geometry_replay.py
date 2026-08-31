"""Lossless BF16 residual replay at frozen token locations, with chunk checksums."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from sae_feature_atlas.scientific.collect import sha256
from sae_feature_atlas.util.io import write_json


def replay(layers):
    torch.backends.cuda.matmul.allow_tf32 = False
    roots = {layer: Path(f"data/processed/gemma4b_foundation_v1_l{layer}") for layer in layers}
    targets, indices = {}, {}
    for layer, root in roots.items():
        plan = json.loads((root / "geometry/plan.json").read_text())
        if sha256(root / "geometry/targets.parquet") != plan["targets_sha256"]:
            raise ValueError("Replay targets changed")
        keys = pd.read_parquet(root / "geometry/targets.parquet")
        keys = (
            keys[["text_id", "token_pos", "token_id"]]
            .drop_duplicates()
            .sort_values(["text_id", "token_pos"])
        )
        if keys.duplicated(["text_id", "token_pos"]).any():
            raise ValueError("Conflicting token identity")
        targets[layer] = {int(k): v for k, v in keys.groupby("text_id")}
        folder = root / "geometry/residual_chunks"
        folder.mkdir(exist_ok=True)
        provenance = {
            "plan_sha256": sha256(root / "geometry/plan.json"),
            "replay_sha256": sha256(__file__),
            "storage": "native bfloat16 bit pattern in uint16; no quantization",
        }
        p = root / "geometry/replay.json"
        if p.exists() and json.loads(p.read_text()) != provenance:
            raise ValueError("Replay provenance changed")
        write_json(p, provenance)
        indices[layer] = []
    p = json.loads((roots[layers[0]] / "collection_provenance.json").read_text())
    texts = [
        json.loads(s) for s in (roots[layers[0]] / "source_texts.jsonl").read_text().splitlines()
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        p["model"]["model_name"], revision=p["model_revision"], local_files_only=True
    )
    model = AutoModel.from_pretrained(
        p["model"]["model_name"],
        revision=p["model_revision"],
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    if hasattr(model, "language_model"):
        model = model.language_model
    model = model.to("cuda").eval()
    captured = {}
    handles = []
    for layer in layers:

        def hook(module, args, output, layer=layer):
            captured[layer] = output[0] if isinstance(output, tuple) else output

        handles.append(model.layers[layer].register_forward_hook(hook))
    for start in range(0, len(texts), 25):
        pending = []
        for layer, root in roots.items():
            done = root / f"geometry/residual_chunks/{start:06d}.done.json"
            if done.exists():
                for name, digest in json.loads(done.read_text()).items():
                    if sha256(done.parent / name) != digest:
                        raise ValueError("Corrupt replay chunk")
            else:
                pending.append(layer)
        if not pending:
            continue
        arrays = {layer: [] for layer in pending}
        records = {layer: [] for layer in pending}
        for item in texts[start : start + 25]:
            tid = int(item["text_id"])
            needed = [layer for layer in pending if tid in targets[layer]]
            if not needed:
                continue
            inputs = tokenizer(
                item["text"], return_tensors="pt", truncation=True, max_length=512
            ).to("cuda")
            with torch.inference_mode():
                model(**inputs, use_cache=False)
                for layer in needed:
                    frame = targets[layer][tid]
                    positions = frame.token_pos.to_numpy().astype(int)
                    ids = inputs.input_ids[0, positions].cpu().numpy()
                    if not np.array_equal(ids, frame.token_id.to_numpy()):
                        raise ValueError("Replay tokenizer mismatch")
                    values = captured[layer][0, positions].contiguous()
                    if values.dtype != torch.bfloat16 or not torch.isfinite(values).all():
                        raise ValueError("Invalid native residual")
                    arrays[layer].append(values.view(torch.uint16).cpu().numpy())
                    records[layer].append(frame)
            captured.clear()
        for layer in pending:
            folder = roots[layer] / "geometry/residual_chunks"
            stem = f"{start:06d}"
            matrix = (
                np.concatenate(arrays[layer])
                if arrays[layer]
                else np.empty((0, model.config.hidden_size), dtype=np.uint16)
            )
            frame = (
                pd.concat(records[layer], ignore_index=True)
                if records[layer]
                else pd.DataFrame(columns=["text_id", "token_pos", "token_id"])
            )
            np.save(folder / (stem + ".npy"), matrix)
            frame.to_parquet(folder / (stem + ".parquet"), index=False)
            write_json(
                folder / (stem + ".done.json"),
                {name: sha256(folder / name) for name in [stem + ".npy", stem + ".parquet"]},
            )
        if start % 250 == 0:
            print("REPLAY", min(start + 25, len(texts)), "/", len(texts), flush=True)
    for h in handles:
        h.remove()
    for layer, root in roots.items():
        write_json(
            root / "geometry/replay_complete.json",
            {
                "documents": len(texts),
                "chunks": len(list((root / "geometry/residual_chunks").glob("*.done.json"))),
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, nargs="+", default=[17, 22])
    replay(parser.parse_args().layers)
