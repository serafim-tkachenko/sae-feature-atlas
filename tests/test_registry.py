from sae_feature_atlas.config.registry import infer_hook_name, make_config


def test_site_aliases_resolve_to_hook_names():
    assert infer_hook_name(13, "res") == "blocks.13.hook_resid_post"
    assert infer_hook_name(13, "resid_post") == "blocks.13.hook_resid_post"
    assert infer_hook_name(13, "mlp") == "blocks.13.hook_mlp_out"
    assert infer_hook_name(13, "mlp_out") == "blocks.13.hook_mlp_out"
    assert infer_hook_name(13, "att") == "blocks.13.hook_attn_out"
    assert infer_hook_name(13, "attn_out") == "blocks.13.hook_attn_out"


def test_make_config_normalizes_site_alias_in_model_config():
    assert make_config(site="res").model.site == "resid_post"
    assert make_config(site="mlp").model.site == "mlp_out"
    assert make_config(site="att").model.site == "attn_out"


def test_inspection_path_property_is_a_path_not_nested_property():
    cfg = make_config()
    assert cfg.inspection_feature_summaries_path.name == "inspection_feature_summaries.parquet"
