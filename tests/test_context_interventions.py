"""Scientific invariants and native-HF plumbing; synthetic fixtures are not evidence."""

import numpy as np
import pytest
import torch

from sae_feature_atlas.scientific.intervention_math import (
    context_delta,
    interaction,
    matched_contrast,
    null_projector,
    paired_bootstrap,
)
from sae_feature_atlas.scientific.intervention_prepare import bucket
from sae_feature_atlas.scientific.intervention_run import NativeProbe, audit_cells


def test_rotation_and_normalization_counterexample():
    rng = np.random.default_rng(1847)
    h, w, u, c, a = rng.normal(size=(5, 32))
    u /= np.linalg.norm(u)
    q = null_projector(w, u)
    delta = context_delta(h, c, q, 0.2)
    assert np.linalg.norm(delta) == pytest.approx(0.2)
    assert w @ delta == pytest.approx(0, abs=1e-12)
    assert u @ delta == pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(h + delta) == pytest.approx(np.linalg.norm(h))
    alpha = 0.7
    cells = np.stack([h, h + alpha * u, h + delta, h + delta + alpha * u])
    audit_cells(cells.astype(np.float32), w, u, 3, 0, 1e-5)

    def radius(x):
        return np.sqrt(x @ x / len(x) + 1e-6)

    observed = interaction(np.asarray([a @ x / radius(x) for x in cells]))
    expected = (a @ delta) * (1 / radius(cells[1]) - 1 / radius(h))
    assert observed == pytest.approx(expected, abs=1e-12)
    assert abs(observed) > 1e-6


def test_collinear_constraints_and_invalid_edits():
    w = np.array([1.0, 0, 0])
    q = null_projector(w, -w)
    np.testing.assert_allclose(q, np.diag([0, 1, 1]), atol=1e-14)
    with pytest.raises(ValueError, match="tangent"):
        context_delta(np.array([0.0, 1, 0]), np.array([0.0, 1, 0]), q, 0.1)
    with pytest.raises(ValueError, match="excessive"):
        context_delta(np.array([0.0, 1, 0]), np.array([0.0, 0, 1]), q, 2)
    with pytest.raises(ValueError, match="constraint failure"):
        audit_cells(
            np.array([[1.0, 1, 1], [2.0, 1, 1], [3.0, 1, 1], [4.0, 1, 1]]), w, w, 0, 0, 1e-5
        )


def test_factorial_identifies_mixed_term_not_main_effect():
    # f(x,y)=3x+5y+7xy, edits alpha=2 and beta=3.
    def f(x, y):
        return 3 * x + 5 * y + 7 * x * y

    values = np.array([f(1, 2), f(3, 2), f(1, 5), f(3, 5)])
    assert interaction(values) == pytest.approx(7 * 2 * 3)


def test_stratum_adjustment_removes_between_group_confound():
    x = np.array([[0.0], [2.0], [100.0], [102.0]])
    contrast, support = matched_contrast(x, [0, 1, 0, 1], ["a", "a", "b", "b"])
    np.testing.assert_allclose(contrast, [2])
    assert support == 4
    with pytest.raises(ValueError, match="overlap"):
        matched_contrast(x, [0, 0, 1, 1], ["a", "a", "b", "b"])


def test_cluster_bootstrap_weights_documents_and_split_is_reproducible():
    result = paired_bootstrap([0.0, 0, 0, 10], ["a", "a", "a", "b"], draws=100)
    assert result["mean"] == 5
    assert result["documents"] == 2
    assert bucket("duplicate", 42) == bucket("duplicate", 42)
    assert bucket("duplicate", 42) != bucket("different", 42)


def tiny_backbone():
    from transformers import Gemma3TextConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel

    torch.manual_seed(9)
    cfg = Gemma3TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=64,
        sliding_window=16,
        layer_types=["full_attention"] * 3,
        query_pre_attn_scalar=8,
    )
    cfg._attn_implementation = "eager"
    return Gemma3TextModel(cfg).float().eval()


def test_native_probe_noop_patch_and_prefix_rerun():
    model = tiny_backbone()
    ids = torch.tensor([[2, 7, 9, 11]])
    readouts = torch.randn(2, 32)
    probe = NativeProbe(model, 1, readouts)
    try:
        baseline = probe.evaluate(ids)
        probe.target_id = 3
        quality = probe.evaluate(ids)
        with torch.inference_mode():
            hidden = model(input_ids=ids, use_cache=False).last_hidden_state[0, -1]
            logits = model.get_input_embeddings().weight @ hidden
            expected_nll = float(torch.logsumexp(logits, 0) - logits[3])
        assert quality["nll"] == pytest.approx(expected_nll, abs=1e-6)
        h = baseline["states"][1]
        replay = probe.evaluate(ids, h)
        np.testing.assert_allclose(baseline["y"], replay["y"], atol=1e-6)
        replacement = h + 0.02 * np.arange(32)
        edited = probe.evaluate(ids, replacement)
        np.testing.assert_allclose(edited["states"][1], replacement.astype(np.float32))
        assert not np.allclose(edited["y"], baseline["y"])
        again = probe.evaluate(ids)
        np.testing.assert_array_equal(again["y"], baseline["y"])
    finally:
        probe.close()
    assert not model.layers[1]._forward_hooks


def test_runner_checkpoint_on_tiny_native_model(tmp_path, monkeypatch):
    import json
    from transformers import AutoModel
    from sae_feature_atlas.scientific.collect import sha256
    from sae_feature_atlas.scientific.intervention_run import run

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    config = dict(
        seed=42,
        layer=1,
        decoder_doses=[-0.1, 0.1],
        context_lengths=[0.005],
        relative_constraint_tolerance=1e-5,
    )
    (bundle / "config.json").write_text(json.dumps(config))
    prompt = dict(feature_id=1, text_id=1, token_pos=3, input_ids=[2, 7, 9, 11])
    (bundle / "prompts.jsonl").write_text(json.dumps(prompt) + "\n")
    rng = np.random.default_rng(6)
    w, u, c = rng.normal(size=(3, 32))
    u /= np.linalg.norm(u)
    np.savez(
        bundle / "vectors.npz",
        **{
            "1_w": w,
            "1_u": u,
            "1_b": 1.0,
            "1_threshold": 0.0,
            "1_directions": c[None],
            "reference_pc8": np.eye(32)[:, :8],
        },
    )
    plan = dict(
        status="development",
        model={"model_name": "synthetic-test"},
        model_revision="test",
        features=[dict(feature_id=1, status="ok", directions=["learned"], decoder_scale=1)],
        readouts=[dict(positive_ids=[1], negative_ids=[2])],
        files={
            name: sha256(bundle / name) for name in ["config.json", "prompts.jsonl", "vectors.npz"]
        },
    )
    (bundle / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *a, **kw: tiny_backbone())
    run(bundle, tmp_path / "results", device="cpu")
    result = json.loads((tmp_path / "results/prompt_00000.json").read_text())
    assert len(result["records"]) == 4
    digest = sha256(tmp_path / "results/prompt_00000.json")
    run(bundle, tmp_path / "results", device="cpu")
    assert sha256(tmp_path / "results/prompt_00000.json") == digest
    (bundle / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="inputs changed"):
        run(bundle, tmp_path / "results", device="cpu")


def test_predictor_detects_context_but_gain_explains_collinear_effects():
    import pandas as pd
    from sae_feature_atlas.scientific.intervention_analysis import prediction_checks

    rng = np.random.default_rng(17)
    rows = []
    for i in range(48):
        context = float(rng.normal())
        rows.append(
            dict(
                feature_id=1,
                dose=0.1,
                source="a" if i % 2 else "b",
                split="fit" if i < 32 else "check",
                text_id=i,
                duplicate_group=str(i),
                encoder=1.0,
                norm=1.0,
                token_pos=1,
                support=1.0,
                trailing_digit_count=0,
                trailing_letter_count=0,
                token_id="same",
                context=context,
                effect=np.array([3 * context, 2 * context]),
                **{f"pc{k}": 0.0 for k in range(8)},
            )
        )
    errors = prediction_checks(pd.DataFrame(rows))
    pooled = errors[errors.train_source == "pooled"].groupby("model").mse.mean()
    assert pooled["context"] < pooled["nuisance"] / 100
    assert pooled["scalar_gain"] == pytest.approx(pooled["context"], rel=1e-8)
