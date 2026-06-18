from sae_feature_atlas.pipeline.steps import ALL_STEPS, STEP_PRESETS, normalize_steps


def test_public_presets_are_minimal_and_explicit():
    assert sorted(STEP_PRESETS) == ["atlas", "core", "research"]
    assert "paper" not in STEP_PRESETS
    assert "candidates" not in ALL_STEPS


def test_research_preset_expands_to_known_steps():
    steps = normalize_steps("all", preset="research")
    assert steps == STEP_PRESETS["research"]
    assert all(step in ALL_STEPS for step in steps)
