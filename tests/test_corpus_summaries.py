from __future__ import annotations

import pandas as pd

from sae_feature_atlas.config.datasets import (
    describe_corpus,
    summarize_text_sources,
    summarize_token_sources,
)


def test_summarize_text_sources_counts_documents_by_source() -> None:
    texts = [
        {"source": "pile-10k", "text": "a"},
        {"source": "pile-10k", "text": "b"},
        {"source": "wikimedia-en", "text": "c"},
    ]

    summary = summarize_text_sources(texts)

    assert summary.to_dict("records") == [
        {"source": "pile-10k", "texts": 2, "text_share": 2 / 3},
        {"source": "wikimedia-en", "texts": 1, "text_share": 1 / 3},
    ]


def test_summarize_token_sources_counts_tokens_after_tokenization() -> None:
    token_meta = pd.DataFrame(
        [
            {"source": "pile-10k", "text_id": 0, "token_id": 1},
            {"source": "pile-10k", "text_id": 0, "token_id": 2},
            {"source": "wikimedia-en", "text_id": 1, "token_id": 3},
        ]
    )

    summary = summarize_token_sources(token_meta)

    assert summary.to_dict("records") == [
        {"source": "pile-10k", "texts": 1, "tokens": 2, "token_share": 2 / 3},
        {"source": "wikimedia-en", "texts": 1, "tokens": 1, "token_share": 1 / 3},
    ]


def test_mixed_broad_notes_do_not_claim_representative_code_math_slices() -> None:
    desc = describe_corpus("mixed-broad")

    assert "Document-balanced" in desc.description
    assert "not token quotas" in desc.notes
    assert "not representative 5% slices" in desc.notes
