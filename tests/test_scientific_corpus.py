import pytest

from sae_feature_atlas.scientific.corpus import select_documents, text_digest


def test_duplicate_detection_crosses_sources_and_normalizes_whitespace():
    text = "A scientific document with enough text. " * 12
    seen = set()
    first, _ = select_documents([{"text": text, "id": "a"}], 1, "first", seen)
    second, rejected = select_documents(
        [{"text": text.replace(" ", "  ")}, {"text": text + "Distinct ending."}],
        1,
        "second",
        seen,
    )
    assert rejected["duplicate"] == 1
    assert first[0]["text"] == text
    assert second[0]["source"] == "second"
    assert text_digest(text) == text_digest(text.replace(" ", "\n"))


def test_source_exhaustion_fails_instead_of_returning_partial_sample():
    with pytest.raises(ValueError, match="required 2"):
        select_documents([{"text": "long document " * 30}, {"text": "short"}], 2, "test", set())
