"""One confident observation per document sensitivity to within-text dependence."""

import numpy as np
import pandas as pd
from scipy import sparse

from sae_feature_atlas.scientific.regimes import analyze_feature, bh_adjust


def one_per_document(cfg, rcfg):
    root = cfg.run_data_dir
    from sae_feature_atlas.scientific.regimes import require_regime_config

    require_regime_config(root, rcfg)
    population = pd.read_parquet(root / "regime_token_population.parquet")
    acts = pd.read_parquet(cfg.sae_activations_path).merge(
        population[["text_id", "token_pos", "token_index"]], on=["text_id", "token_pos"]
    )
    matrix = sparse.csr_matrix(
        (
            np.ones(len(acts), dtype=np.float32),
            (acts.token_index.to_numpy(), acts.feature_id.to_numpy()),
        ),
        shape=(len(population), int(acts.feature_id.max()) + 1),
    )
    assignments = pd.read_parquet(root / "regime_assignments.parquet")
    if assignments.empty and "role" not in assignments:
        assignments["role"] = pd.Series(dtype=str)
    assignments = assignments[assignments.role == "candidate"]
    rows, null_rows, samples = [], [], []
    for fid, group in assignments.groupby("feature_id"):
        # Uniform selection within the frozen confident set, without consulting
        # regime labels or partner structure. Each source document contributes once.
        selected = (
            group[group.regime >= 0]
            .groupby("text_id", group_keys=False)
            .sample(n=1, random_state=rcfg.seed + int(fid))
            .copy()
        )
        result, _, nulls, _ = analyze_feature(selected, matrix, int(fid), rcfg)
        rows.append({"feature_id": int(fid), **result})
        null_rows.extend(
            {"feature_id": int(fid), **r} for r in nulls if r["null_kind"] == "token_identity"
        )
        samples.append(selected)
        print("ONE PER DOCUMENT", fid, result["status"], result.get("token_identity_p"), flush=True)
    result = pd.DataFrame(
        rows, columns=None if rows else ["feature_id", "status", "token_identity_p", "q_bh", "q_by"]
    )
    if not result.empty:
        if "token_identity_p" not in result:
            result["token_identity_p"] = np.nan
        q = bh_adjust(result.token_identity_p.fillna(1))
        result["q_bh"] = q
        result["q_by"] = np.minimum(1, q * sum(1 / np.arange(1, len(q) + 1)))
    result.to_parquet(root / "regime_one_per_document.parquet", index=False)
    pd.DataFrame(null_rows).to_parquet(root / "regime_one_per_document_nulls.parquet", index=False)
    (pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()).to_parquet(
        root / "regime_one_per_document_sample.parquet", index=False
    )
    return result
