from __future__ import annotations

import pytest

from sae_feature_atlas.inspection.semantic_annotations import (
    AnnotatorProvenance,
    SemanticAnnotation,
    SemanticEvidencePacket,
    validate_annotation_evidence,
)


def test_semantic_annotation_requires_empirical_evidence_references() -> None:
    packet = SemanticEvidencePacket(
        feature_id=7,
        top_context_ids=("top:7:0",),
        held_out_context_ids=("heldout:7:0",),
        supplementary_decoder_neighbor_ids=("decoder:7:9",),
    )
    annotation = SemanticAnnotation(
        feature_id=7,
        description="Appears in contexts about measurement.",
        scope_conditions=("English prose",),
        supporting_evidence_ids=("top:7:0",),
        counterexample_evidence_ids=("heldout:7:0",),
        uncertainty_and_failure_modes=("May track nearby syntax.",),
        confidence=0.6,
        abstain=False,
        provenance=AnnotatorProvenance(
            annotator="test",
            version="1",
            prompt_or_protocol_version="evidence-v1",
        ),
    )

    validate_annotation_evidence(annotation, packet)
    assert annotation.to_dict()["provenance"]["version"] == "1"


def test_semantic_annotation_rejects_unknown_evidence_and_invalid_abstention() -> None:
    packet = SemanticEvidencePacket(feature_id=7, top_context_ids=("top:7:0",))
    annotation = SemanticAnnotation(
        feature_id=7,
        description="A description",
        scope_conditions=(),
        supporting_evidence_ids=("missing",),
        counterexample_evidence_ids=(),
        uncertainty_and_failure_modes=(),
        confidence=0.4,
        abstain=False,
        provenance=AnnotatorProvenance(annotator="test", version="1"),
    )
    with pytest.raises(ValueError, match="unknown evidence"):
        validate_annotation_evidence(annotation, packet)

    with pytest.raises(ValueError, match="must not assert"):
        SemanticAnnotation(
            feature_id=7,
            description="Unsupported assertion",
            scope_conditions=(),
            supporting_evidence_ids=(),
            counterexample_evidence_ids=(),
            uncertainty_and_failure_modes=(),
            confidence=None,
            abstain=True,
            provenance=AnnotatorProvenance(annotator="test", version="1"),
        )
