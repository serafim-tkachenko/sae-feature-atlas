from sae_feature_atlas.config.datasets import describe_corpus, list_supported_corpora


def test_public_corpus_registry_includes_research_presets():
    corpora = set(list_supported_corpora())

    assert {
        "pile-10k",
        "fineweb-edu-sample",
        "wikimedia-en",
        "mixed-research",
        "mixed-broad",
        "mixed-large",
    }.issubset(corpora)


def test_mixed_broad_descriptor_explains_why_it_exists():
    descriptor = describe_corpus("mixed-broad")

    assert descriptor.streaming is True
    assert "web" in descriptor.domains
    assert "encyclopedic" in descriptor.domains
    assert "corpus-specific" in descriptor.notes


def test_jsonl_corpus_descriptor_is_supported_without_registry_entry():
    descriptor = describe_corpus("jsonl:/tmp/custom_corpus.jsonl")

    assert descriptor.name == "jsonl:/tmp/custom_corpus.jsonl"
    assert descriptor.streaming is False
    assert descriptor.domains == ("custom",)