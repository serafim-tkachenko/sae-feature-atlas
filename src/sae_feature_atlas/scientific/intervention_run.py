"""Float32 native-model factorial pilot with paired edits and resumable artifacts."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sae_feature_atlas.scientific.collect import sha256
from sae_feature_atlas.scientific.intervention_math import (
    context_delta,
    interaction,
    null_projector,
)
from sae_feature_atlas.util.io import write_json


class NativeProbe:
    """Run the whole prefix in each arm; never resume from an isolated token state."""

    def __init__(self, model, layer, readouts):
        self.model, self.layer, self.readouts = model, layer, readouts
        self.replacement = None
        self.target_id = None
        self.handles = []
        for index, block in enumerate(model.layers):
            if index < layer:
                continue

            def hook(module, args, output, index=index):
                hidden = output[0] if isinstance(output, tuple) else output
                if index == self.layer and self.replacement is not None:
                    hidden = hidden.clone()
                    hidden[0, -1] = self.replacement.to(hidden.device, hidden.dtype)
                    output = (hidden, *output[1:]) if isinstance(output, tuple) else hidden
                self.states[index] = hidden[0, -1].detach().double().cpu().numpy()
                return output

            self.handles.append(block.register_forward_hook(hook))

        def pre_norm(module, args):
            self.pre_norm = args[0][0, -1].detach()

        self.handles.append(model.norm.register_forward_pre_hook(pre_norm))

    def close(self):
        for handle in self.handles:
            handle.remove()

    def evaluate(self, ids, replacement=None):
        self.replacement = None if replacement is None else torch.as_tensor(replacement)
        self.states = {}
        with torch.inference_mode():
            result = self.model(input_ids=ids, use_cache=False)
            after = result.last_hidden_state[0, -1]
            if after.dtype != torch.float32:
                raise ValueError("Constraint-preserving pilot requires float32 inference")
            y = (self.readouts @ after).double().cpu().numpy()
            pre = (self.readouts @ self.pre_norm).double().cpu().numpy()
            nll = None
            if self.target_id is not None:
                logits = self.model.get_input_embeddings().weight @ after
                cap = getattr(self.model.config, "final_logit_softcapping", None)
                if cap is not None:
                    logits = cap * torch.tanh(logits / cap)
                nll = float(torch.logsumexp(logits, 0) - logits[self.target_id])
        if not np.isfinite(y).all() or any(not np.isfinite(x).all() for x in self.states.values()):
            raise ValueError("Nonfinite model state")
        return dict(y=y, pre=pre, nll=nll, states=self.states.copy())


def audit_cells(cells, w, u, b, threshold, tolerance):
    """Audit representable FP32 states, not merely ideal FP64 construction."""
    h, decoder, context, combined = np.asarray(cells, dtype=np.float64)
    scores = np.asarray(cells) @ w + b
    activations = np.where(scores > threshold, scores, 0)
    errors = dict(
        encoder_relative=max(
            abs(scores[2] - scores[0]) / max(1, abs(scores[0])),
            abs(scores[3] - scores[1]) / max(1, abs(scores[1])),
        ),
        decoder_coordinate_relative=abs(u @ (context - h)) / max(1, np.linalg.norm(h)),
        norm_relative=max(
            abs(np.linalg.norm(context) / np.linalg.norm(h) - 1),
            abs(np.linalg.norm(combined) / np.linalg.norm(decoder) - 1),
        ),
        patch_additivity_relative=float(
            np.linalg.norm(combined - context - decoder + h) / max(1, np.linalg.norm(h))
        ),
        focal_activation_relative=max(
            abs(activations[2] - activations[0]) / max(1, abs(activations[0])),
            abs(activations[3] - activations[1]) / max(1, abs(activations[1])),
        ),
    )
    if not np.isfinite(list(errors.values())).all() or max(errors.values()) > tolerance:
        raise ValueError(f"Representable-state constraint failure: {errors}")
    return {k: float(v) for k, v in errors.items()}


def run(bundle, out, limit=None, local_files_only=False, device="cuda"):
    from transformers import AutoModel

    plan = json.loads((bundle / "plan.json").read_text())
    cfg = json.loads((bundle / "config.json").read_text())
    for name, digest in plan["files"].items():
        if sha256(bundle / name) != digest:
            raise ValueError("Prepared pilot inputs changed")
    if "development" not in plan["status"]:
        raise ValueError("This runner is not a confirmation protocol")
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(
        plan_sha256=sha256(bundle / "plan.json"),
        runner_sha256=sha256(__file__),
        math_sha256=sha256(Path(__file__).with_name("intervention_math.py")),
        torch=torch.__version__,
        dtype="float32",
        tf32=False,
        limit=limit,
        device=device,
    )
    if (out / "provenance.json").exists():
        if json.loads((out / "provenance.json").read_text()) != provenance:
            raise ValueError("Changed inputs/code/runtime: use a new output directory")
    write_json(out / "provenance.json", provenance)
    prompts = [json.loads(s) for s in (bundle / "prompts.jsonl").read_text().splitlines()]
    # A smoke limit samples across feature/source/split cells, not just the first feature.
    prompts.sort(key=lambda p: (p["text_id"], p["feature_id"], p["token_pos"]))
    if limit is not None:
        if limit < 1:
            raise ValueError("Smoke limit must be positive")
        groups = {}
        for prompt in prompts:
            groups.setdefault(prompt["feature_id"], []).append(prompt)
        selected = []
        for index in range(max(map(len, groups.values()))):
            for group in groups.values():
                if index < len(group):
                    selected.append(group[index])
        prompts = selected[:limit]
    model = AutoModel.from_pretrained(
        plan["model"]["model_name"],
        revision=plan["model_revision"],
        dtype=torch.float32,
        local_files_only=local_files_only,
        attn_implementation="eager",
    )
    if hasattr(model, "language_model"):
        model = model.language_model
    if not model.config.tie_word_embeddings or not hasattr(model, "layers"):
        raise ValueError("Unsupported backbone or untied output head")
    model = model.to(device).eval()
    embedding = model.get_input_embeddings().weight
    readouts = torch.stack(
        [
            embedding[r["positive_ids"]].mean(0) - embedding[r["negative_ids"]].mean(0)
            for r in plan["readouts"]
        ]
    ).detach()
    probe = NativeProbe(model, cfg["layer"], readouts)
    vectors = np.load(bundle / "vectors.npz")
    features = {r["feature_id"]: r for r in plan["features"] if r["status"] == "ok"}
    try:
        for number, prompt in enumerate(prompts):
            started = time.monotonic()
            target = out / f"prompt_{number:05d}.json"
            if target.exists():
                checksum = target.with_suffix(".sha256")
                if not checksum.exists() or checksum.read_text().strip() != sha256(target):
                    raise ValueError("Corrupt or incomplete prompt checkpoint")
                stored = json.loads(target.read_text())
                if stored["prompt"] != prompt:
                    raise ValueError("Checkpoint prompt mismatch")
                continue
            fid = prompt["feature_id"]
            w, u = vectors[f"{fid}_w"], vectors[f"{fid}_u"]
            b, threshold = float(vectors[f"{fid}_b"]), float(vectors[f"{fid}_threshold"])
            q = null_projector(w, u)
            ids = torch.tensor([prompt["input_ids"]], device=device)
            probe.target_id = prompt.get("next_token_id")
            base = probe.evaluate(ids)
            h = base["states"][cfg["layer"]]
            scores = dict(
                encoder=float(w @ h + b),
                norm=float(np.linalg.norm(h)),
                context=float(vectors[f"{fid}_directions"][0] @ h),
                pc8=(vectors["reference_pc8"].T @ h).tolist(),
                next_token_nll=base["nll"],
            )
            decoder_results = {}
            for dose in cfg["decoder_doses"]:
                alpha = dose * features[fid]["decoder_scale"]
                hd = (h + alpha * u).astype(np.float32)
                decoder_results[dose] = (hd, probe.evaluate(ids, hd))
            records = []
            for name, direction in zip(features[fid]["directions"], vectors[f"{fid}_directions"]):
                for fraction in cfg["context_lengths"]:
                    for sign in (-1, 1):
                        try:
                            delta = context_delta(
                                h, direction, q, fraction * np.linalg.norm(h), sign
                            )
                        except ValueError as exc:
                            records.append(
                                dict(
                                    direction=name,
                                    fraction=fraction,
                                    sign=sign,
                                    status="degenerate",
                                    reason=str(exc),
                                )
                            )
                            continue
                        hc = (h + delta).astype(np.float32)
                        context = probe.evaluate(ids, hc)
                        for dose in cfg["decoder_doses"]:
                            alpha = dose * features[fid]["decoder_scale"]
                            hd, dec = decoder_results[dose]
                            combined_h = (h + delta + alpha * u).astype(np.float32)
                            cells = np.stack([h, hd, hc, combined_h])
                            audit = audit_cells(
                                cells, w, u, b, threshold, cfg["relative_constraint_tolerance"]
                            )
                            combined = probe.evaluate(ids, combined_h)
                            results = [base, dec, context, combined]
                            values = np.stack([r["y"] for r in results])
                            pre_values = np.stack([r["pre"] for r in results])
                            with torch.inference_mode():
                                surrogate = model.norm(
                                    torch.tensor(cells, device=device, dtype=torch.float32)
                                )
                                surrogate = (surrogate @ readouts.T).double().cpu().numpy()
                            trajectory = {
                                str(i): float(
                                    np.linalg.norm(
                                        interaction(np.stack([r["states"][i] for r in results]))
                                    )
                                )
                                for i in base["states"]
                            }
                            records.append(
                                dict(
                                    status="ok",
                                    direction=name,
                                    fraction=fraction,
                                    sign=sign,
                                    dose=dose,
                                    alpha=alpha,
                                    length=float(np.linalg.norm(delta)),
                                    audit=audit,
                                    cells=values.tolist(),
                                    interaction=interaction(values).tolist(),
                                    pre_norm_cells=pre_values.tolist(),
                                    pre_norm_interaction=interaction(pre_values).tolist(),
                                    normalization_only_interaction=interaction(surrogate).tolist(),
                                    next_token_nll=[r["nll"] for r in results],
                                    layer_interaction_norm=trajectory,
                                )
                            )
            payload = dict(
                prompt=prompt,
                baseline=scores,
                records=records,
                elapsed_seconds=time.monotonic() - started,
            )
            temp = target.with_suffix(".tmp")
            write_json(temp, payload)
            temp.replace(target)
            target.with_suffix(".sha256").write_text(sha256(target) + "\n")
            print(
                f"PILOT {number + 1}/{len(prompts)} feature={fid} cells={len(records)}", flush=True
            )
        write_json(
            out / "complete.json",
            dict(
                status="development diagnostics only",
                prompts=len(prompts),
                files={p.name: sha256(p) for p in sorted(out.glob("prompt_*.json"))},
            ),
        )
    finally:
        probe.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.bundle, args.out, args.limit, args.local_files_only, args.device)
