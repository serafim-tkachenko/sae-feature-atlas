"""Independently verify the completed pilot's identities, keys and completeness."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from sae_feature_atlas.scientific.collect import sha256
from sae_feature_atlas.util.io import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
plan = json.loads((args.bundle / "plan.json").read_text())
cfg = json.loads((args.bundle / "config.json").read_text())
complete = json.loads((args.results / "complete.json").read_text())
provenance = json.loads((args.results / "provenance.json").read_text())
assert provenance["plan_sha256"] == sha256(args.bundle / "plan.json")
for name, digest in plan["files"].items():
    assert sha256(args.bundle / name) == digest
prompts = [json.loads(s) for s in (args.bundle / "prompts.jsonl").read_text().splitlines()]
fit = {p["duplicate_group"] for p in prompts if p["split"] == "fit"}
check = {p["duplicate_group"] for p in prompts if p["split"] == "check"}
assert not fit & check
expected = {(p["feature_id"], p["text_id"], p["token_pos"]): p for p in prompts}
seen, count, missing_loss, max_error = set(), 0, 0, 0.0
features = {f["feature_id"]: f for f in plan["features"] if f["status"] == "ok"}
for name, digest in complete["files"].items():
    assert sha256(args.results / name) == digest
    result = json.loads((args.results / name).read_text())
    p = result["prompt"]
    key = p["feature_id"], p["text_id"], p["token_pos"]
    assert key not in seen and expected[key] == p
    seen.add(key)
    expected_cells = set(
        itertools.product(
            features[p["feature_id"]]["directions"],
            cfg["context_lengths"],
            [-1, 1],
            cfg["decoder_doses"],
        )
    )
    observed = set()
    for r in result["records"]:
        assert r["status"] == "ok", "Pilot exclusion requires explicit review"
        cell_key = r["direction"], r["fraction"], r["sign"], r["dose"]
        assert cell_key not in observed
        observed.add(cell_key)
        for field, effect in [("cells", "interaction"), ("pre_norm_cells", "pre_norm_interaction")]:
            values = np.asarray(r[field], dtype=float)
            assert np.isfinite(values).all()
            np.testing.assert_allclose(
                values[3] - values[2] - values[1] + values[0], r[effect], atol=1e-12, rtol=0
            )
        error = max(r["audit"].values())
        assert np.isfinite(error) and error < cfg["relative_constraint_tolerance"]
        max_error = max(max_error, error)
        if p.get("next_token_id") is None:
            assert r["next_token_nll"] == [None] * 4
        else:
            assert np.isfinite(np.asarray(r["next_token_nll"], dtype=float)).all()
        count += 1
    missing_loss += p.get("next_token_id") is None
    assert observed == expected_cells
assert seen == set(expected)
assert len(seen) == complete["prompts"] == plan["prompts"]
write_json(
    args.out,
    dict(
        status="passed",
        prompts=len(seen),
        factorial_comparisons=count,
        fit_check_groups_disjoint=True,
        max_constraint_error=max_error,
        cases_without_next_token_target=missing_loss,
        reason_missing_target="Position is last token of the frozen 512-token collection window",
        complete_sha256=sha256(args.results / "complete.json"),
    ),
)
print(args.out)
