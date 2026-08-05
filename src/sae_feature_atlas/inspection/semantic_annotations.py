from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class SemanticEvidencePacket:
    """Identifiers for empirical evidence supplied to a semantic annotator.

    Context evidence is primary. Decoder and coactivation evidence is explicitly
    supplementary and must not be treated as a direct language translation of a
    decoder direction.
    """

    feature_id: int
    activation_population: str = "analysis_activations"
    top_context_ids: tuple[str, ...] = ()
    random_positive_context_ids: tuple[str, ...] = ()
    activation_quantile_context_ids: tuple[str, ...] = ()
    source_distribution_evidence_ids: tuple[str, ...] = ()
    target_token_distribution_evidence_ids: tuple[str, ...] = ()
    held_out_context_ids: tuple[str, ...] = ()
    regime_context_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    supplementary_decoder_neighbor_ids: tuple[str, ...] = ()
    supplementary_coactivation_neighbor_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegimeSemanticDescription:
    regime_id: str
    description: str
    scope_conditions: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    counterexample_evidence_ids: tuple[str, ...] = ()
    uncertainty_and_failure_modes: tuple[str, ...] = ()
    confidence: float | None = None


@dataclass(frozen=True)
class AnnotatorProvenance:
    annotator: str
    version: str
    model: str | None = None
    prompt_or_protocol_version: str | None = None


@dataclass(frozen=True)
class SemanticAnnotation:
    feature_id: int
    description: str | None
    scope_conditions: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    counterexample_evidence_ids: tuple[str, ...]
    uncertainty_and_failure_modes: tuple[str, ...]
    confidence: float | None
    abstain: bool
    provenance: AnnotatorProvenance
    regime_descriptions: tuple[RegimeSemanticDescription, ...] = ()

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.abstain and self.description:
            raise ValueError("abstaining annotations must not assert a description")
        if not self.abstain and not self.description:
            raise ValueError("non-abstaining annotations require a description")

    def to_dict(self) -> dict:
        return asdict(self)


class SemanticAnnotator(Protocol):
    """Future annotator boundary; implementations must consume evidence packets."""

    def annotate(self, evidence: SemanticEvidencePacket) -> SemanticAnnotation:
        ...


def validate_annotation_evidence(
    annotation: SemanticAnnotation,
    evidence: SemanticEvidencePacket,
) -> None:
    """Ensure annotation claims refer only to evidence in the supplied packet."""
    if annotation.feature_id != evidence.feature_id:
        raise ValueError("annotation and evidence packet feature IDs differ")

    available = {
        *evidence.top_context_ids,
        *evidence.random_positive_context_ids,
        *evidence.activation_quantile_context_ids,
        *evidence.source_distribution_evidence_ids,
        *evidence.target_token_distribution_evidence_ids,
        *evidence.held_out_context_ids,
        *evidence.supplementary_decoder_neighbor_ids,
        *evidence.supplementary_coactivation_neighbor_ids,
    }
    for ids in evidence.regime_context_ids.values():
        available.update(ids)

    referenced = {
        *annotation.supporting_evidence_ids,
        *annotation.counterexample_evidence_ids,
    }
    for regime in annotation.regime_descriptions:
        referenced.update(regime.supporting_evidence_ids)
        referenced.update(regime.counterexample_evidence_ids)

    missing = referenced - available
    if missing:
        raise ValueError(f"annotation references unknown evidence IDs: {sorted(missing)}")
