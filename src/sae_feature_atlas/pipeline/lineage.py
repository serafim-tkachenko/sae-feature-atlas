from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from sae_feature_atlas.analysis import token_quality
from sae_feature_atlas.config.schema import ExperimentConfig
from sae_feature_atlas.util.io import write_json


ARTIFACT_SCHEMA_VERSION = 2


class StaleArtifactError(RuntimeError):
    pass


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def token_quality_policy_digest() -> str:
    """Hash the classifier implementation as part of analysis population identity."""
    module_path = Path(token_quality.__file__)
    return hashlib.sha256(module_path.read_bytes()).hexdigest()


def collection_fingerprint_payload(cfg: ExperimentConfig) -> dict:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model": asdict(cfg.model),
        "collection": asdict(cfg.collection),
    }


def analysis_fingerprint_payload(cfg: ExperimentConfig) -> dict:
    return {
        "collection_fingerprint": _fingerprint(collection_fingerprint_payload(cfg)),
        "token_quality_policy_digest": token_quality_policy_digest(),
        "activation_row_filter": asdict(cfg.activation_filter),
        "feature_filter": asdict(cfg.feature_filter),
        "analysis": asdict(cfg.analysis),
    }


def fingerprints(cfg: ExperimentConfig) -> dict[str, str]:
    return {
        "collection": _fingerprint(collection_fingerprint_payload(cfg)),
        "analysis": _fingerprint(analysis_fingerprint_payload(cfg)),
    }


def git_provenance(cwd: Path | None = None) -> dict:
    cwd = cwd or Path.cwd()

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit_sha": run("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def build_lineage(cfg: ExperimentConfig, stage: str) -> dict:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "stage": stage,
        "run_name": cfg.collection.run_name,
        "fingerprints": fingerprints(cfg),
        "fingerprint_payloads": {
            "collection": collection_fingerprint_payload(cfg),
            "analysis": analysis_fingerprint_payload(cfg),
        },
        "git": git_provenance(),
    }


def write_lineage(cfg: ExperimentConfig, stage: str) -> dict:
    lineage = build_lineage(cfg, stage)
    write_json(cfg.lineage_path, lineage)
    return lineage


def require_compatible_lineage(
    cfg: ExperimentConfig,
    *,
    require_analysis_match: bool = True,
) -> dict:
    if not cfg.lineage_path.exists():
        raise StaleArtifactError(
            f"Missing lineage metadata at {cfg.lineage_path}. Existing artifacts must be "
            "explicitly migrated or regenerated before reuse."
        )
    lineage = json.loads(cfg.lineage_path.read_text(encoding="utf-8"))
    if lineage.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise StaleArtifactError("Artifact schema version is incompatible with current code.")

    expected = fingerprints(cfg)
    observed = lineage.get("fingerprints", {})
    if observed.get("collection") != expected["collection"]:
        raise StaleArtifactError(
            "Collection fingerprint mismatch; do not reuse token/activation artifacts."
        )
    if require_analysis_match and observed.get("analysis") != expected["analysis"]:
        raise StaleArtifactError(
            "Analysis fingerprint mismatch; derived artifacts must be regenerated."
        )
    return lineage
