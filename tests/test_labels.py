import numpy as np
import pandas as pd
import pytest

from sae_feature_atlas.analysis.labels import assign_feature_labels


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0.])
def test_missing_or_nonfinite_activation_cannot_be_high_intensity(value):
    cards = pd.DataFrame({"feature_id": [1], "p99_activation": [value]})
    result = assign_feature_labels(cards)
    assert result.primary_label.eq("unlabeled").all()
    assert result.manual_priority.eq("unreviewed").all()


def test_missing_measurements_do_not_lower_the_intensity_threshold():
    cards = pd.DataFrame({"feature_id": range(12), "p99_activation": [np.nan] * 10 + [1., 10.]})
    result = assign_feature_labels(cards)
    assert result[result.primary_label.eq("high_intensity")].feature_id.tolist() == [11]


def test_recomputed_labels_do_not_keep_stale_automatic_classifications():
    cards = pd.DataFrame([{
        "feature_id": 7, "inspection_labels": "likely_artifact,high_intensity,coactivation_hub,boundary_heavy",
        "artifact_score": 0.1, "p99_activation": np.nan, "max_coactivation_jaccard": 0.,
    }])
    result = assign_feature_labels(cards).iloc[0]
    assert result.inspection_labels == "boundary_heavy"
    assert result.primary_label == "unlabeled"
