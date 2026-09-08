"""Independent local checks of frozen samples, corrected endpoints and replay hashes."""

import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def adjusted(p):
    p = np.asarray(p)
    order = np.argsort(p)
    ranks = np.arange(1, len(p) + 1)
    ordered = p[order] * len(p) * np.sum(1 / ranks) / ranks
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.minimum.accumulate(ordered[::-1])[::-1])
    return result


def main():
    records = []
    for layer in [17, 22]:
        root = Path(f"data/processed/gemma4b_foundation_v1_l{layer}")
        g = root / "geometry"
        plan = json.loads((g / "plan.json").read_text())
        assert digest(g / "targets.parquet") == plan["targets_sha256"]
        assert digest(root / "regime_feature_summary.parquet") == plan["discovery_fits_sha256"]
        target = pd.read_parquet(g / "targets.parquet")
        primary = target[target.purpose == "primary"]
        source = pd.read_parquet(root / "source_confirmation/source_results.parquet")
        assert len(source) == 2 * len(plan["primary_ids"])
        for fid in plan["primary_ids"]:
            for domain in ["fineweb-edu-sample", "wikimedia-en"]:
                expected = primary[
                    (primary.feature_id == fid) & (primary.source == domain)
                ].sort_values(["text_id", "token_pos"])
                actual = pd.read_parquet(
                    root / f"source_confirmation/{fid}_{domain}.sample.parquet"
                ).sort_values(["text_id", "token_pos"])
                assert not actual.duplicate_group.duplicated().any()
                np.testing.assert_array_equal(
                    expected[["text_id", "token_pos"]].to_numpy(),
                    actual[["text_id", "token_pos"]].to_numpy(),
                )
                np.testing.assert_array_equal(expected.level, actual.regime)
        for directory, filename, pname in [
            (root / "source_confirmation", "source_results.parquet", "token_identity_p"),
            (g, "orthogonal_source_results.parquet", "p"),
            (g, "encoder_control_source.parquet", "p"),
        ]:
            frame = pd.read_parquet(directory / filename)
            ps = frame[pname].where(frame.status == "ok", 1.0)
            p = frame.assign(value=ps).groupby("feature_id").value.max()
            saved = pd.read_parquet(
                directory
                / (
                    "conjunction.parquet"
                    if pname == "token_identity_p"
                    else "encoder_control_conjunction.parquet"
                    if filename == "encoder_control_source.parquet"
                    else "orthogonal_conjunction.parquet"
                )
            ).sort_values("feature_id")
            np.testing.assert_allclose(
                saved.conjunction_p, p.reindex(saved.feature_id), rtol=0, atol=1e-14
            )
            np.testing.assert_allclose(
                saved.q_by, adjusted(saved.conjunction_p), rtol=0, atol=1e-14
            )
        for filename, suffix in [
            ("orthogonal_source_results", "orthogonal"),
            ("encoder_control_source", "encoder_control"),
        ]:
            rows = pd.read_parquet(g / f"{filename}.parquet")
            for row in rows[rows.status == "ok"].itertuples():
                null = np.load(g / f"{row.feature_id}_{row.source}.{suffix}_null.npy")
                p = (
                    1
                    + np.sum(
                        null >= row.orthogonal_squared - 1e-10 * max(1, abs(row.orthogonal_squared))
                    )
                ) / (len(null) + 1)
                assert p == row.p
        replay = json.loads((g / "replay_activation_audit.json").read_text())
        assert replay["passed"] and replay["positive_membership_disagreements"] == 0
        complete = json.loads((root / "secondary_complete.json").read_text())
        pooled = pd.read_parquet(root / "regime_neighborhood_comparison.parquet")
        pooled = pooled[pooled.role == "candidate"].set_index("feature_id")
        assert set(pooled.index) == set(plan["primary_ids"])
        matched = pd.read_parquet(root / "regime_weak_matched_controls.parquet")
        weak = pd.read_parquet(root / "regime_weak_control_comparison.parquet")
        expected = {
            int(r.feature_id)
            for r in matched[matched.control_id.notna()].itertuples()
            if pooled.loc[r.feature_id, "status"] == "ok"
        }
        assert set(weak.candidate_id) == expected
        assert int((weak.status == "ok").sum()) == complete["weak_supported"]
        for row in weak[weak.status == "ok"].itertuples():
            candidate = pooled.loc[row.candidate_id]
            assert candidate.n_low == row.n_low and candidate.n_high == row.n_high
        chunks = list((g / "residual_chunks").glob("*.done.json"))
        assert len(chunks) == 480
        n = 0
        for done in chunks:
            for name, sha in json.loads(done.read_text()).items():
                assert digest(done.parent / name) == sha
            x = np.load(done.parent / (done.name.split(".")[0] + ".npy"), mmap_mode="r")
            assert x.dtype == np.uint16 and x.shape[1] == 2560
            n += len(x)
        assert n == plan["native_token_count"]
        records.append(
            {
                "layer": layer,
                "sample_keys_and_labels_match": True,
                "conjunctions_and_BY_recomputed": True,
                "numerical_ties_checked": True,
                "encoder_diagnostic_recomputed": True,
                "replay_membership_audit_passed": True,
                "secondary_controls_complete": True,
                "weak_control_pairs": complete["weak_supported"],
                "residual_chunks_verified": len(chunks),
                "native_locations": n,
                "frozen_targets_sha256": plan["targets_sha256"],
            }
        )
        print("VERIFIED", layer, flush=True)
    out = Path("reports/gemma4b_foundation_v1/verification.json")
    out.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
