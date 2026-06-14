from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def compute_bimodality_candidates(
    acts: pd.DataFrame,
    min_points: int = 100,
) -> pd.DataFrame:
    rows = []

    for feature_id, group in tqdm(acts.groupby("feature_id"), desc="Bimodality"):
        values = group["activation"].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        values = values[values > 0]

        if len(values) < min_points:
            continue

        x = np.log1p(values).reshape(-1, 1)

        try:
            gmm1 = GaussianMixture(n_components=1, random_state=0).fit(x)
            gmm2 = GaussianMixture(n_components=2, random_state=0).fit(x)

            bic1 = gmm1.bic(x)
            bic2 = gmm2.bic(x)
            score = bic1 - bic2
            means = sorted(gmm2.means_.reshape(-1).tolist())

            rows.append(
                {
                    "feature_id": int(feature_id),
                    "n_points": int(len(values)),
                    "bic_1": float(bic1),
                    "bic_2": float(bic2),
                    "bimodality_score": float(score),
                    "log_mean_low": float(means[0]),
                    "log_mean_high": float(means[1]),
                    "activation_min": float(values.min()),
                    "activation_p50": float(np.quantile(values, 0.50)),
                    "activation_p95": float(np.quantile(values, 0.95)),
                    "activation_max": float(values.max()),
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values("bimodality_score", ascending=False).reset_index(drop=True)
