from __future__ import annotations

CORE_STEPS = ["collect"]

ATLAS_STEPS = [
    "collect",
    "features",
    "coactivation",
    "geometry",
    "geometry-vs-coactivation",
    "bimodality",
    "inspection",
    "space",
    "cards",
    "report",
]

RESEARCH_STEPS = [
    "collect",
    "features",
    "coactivation",
    "geometry",
    "geometry-vs-coactivation",
    "bimodality",
    "inspection",
    "space",
    "coverage",
    "alignment",
    "cards",
    "report",
]

STEP_PRESETS: dict[str, list[str]] = {
    "core": CORE_STEPS,
    "atlas": ATLAS_STEPS,
    "research": RESEARCH_STEPS,
}

ALL_STEPS = [
    "collect",
    "features",
    "coactivation",
    "geometry",
    "geometry-vs-coactivation",
    "bimodality",
    "inspection",
    "space",
    "coverage",
    "alignment",
    "cards",
    "report",
]


def normalize_steps(steps: str | list[str] = "all", preset: str | None = None) -> list[str]:
    """Normalize user step selection into an executable step list.

    Presets define the public workflow levels:
    - core: collect model activations and SAE activations;
    - atlas: build reusable feature cards and local geometry/coactivation views;
    - research: atlas plus residual-space coverage and graph-alignment metrics.
    """
    if preset is not None:
        if preset not in STEP_PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Available: {sorted(STEP_PRESETS)}")
        if steps in (None, "all"):
            return STEP_PRESETS[preset]

    raw = [s.strip() for s in steps.split(",")] if isinstance(steps, str) else list(steps)
    raw = [s for s in raw if s]

    if not raw or "all" in raw:
        return STEP_PRESETS.get(preset or "atlas", ATLAS_STEPS)

    if len(raw) == 1 and raw[0] in STEP_PRESETS:
        return STEP_PRESETS[raw[0]]

    unknown = sorted(set(raw) - set(ALL_STEPS))
    if unknown:
        raise ValueError(
            f"Unknown steps: {unknown}. Available steps: {ALL_STEPS}. "
            f"Available presets: {sorted(STEP_PRESETS)}"
        )
    return raw
