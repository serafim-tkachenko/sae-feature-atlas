from sae_feature_atlas.config.registry import make_config
from sae_feature_atlas.pipeline.runner import run_pipeline

if __name__ == "__main__":
    cfg = make_config(model="gemma-3-1b-pt", layer=13)
    run_pipeline(cfg, steps="all")
