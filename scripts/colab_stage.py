"""Execute a notebook stage in the isolated scientific environment."""

from dataclasses import replace
import json
from pathlib import Path
import sys

from sae_feature_atlas.config.schema import PathsConfig
from sae_feature_atlas.scientific.collect import configuration, collect
from sae_feature_atlas.scientific.regimes import RegimeConfig
from sae_feature_atlas.scientific.run import analyze, manifest


def main(stage, settings_path):
    settings = json.loads(Path(settings_path).read_text())
    cfg = configuration(settings["run_name"], settings["max_texts"], settings["max_seq_len"])
    cfg = replace(
        cfg,
        paths=PathsConfig(
            data_root=Path(settings["data_root"]), reports_root=Path(settings["reports_root"])
        ),
    )
    rcfg = RegimeConfig(**settings["regimes"])
    if stage == "sanity":
        import torch
        from huggingface_hub import hf_hub_download
        import psutil

        if not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime, then Run all again.")
        if torch.cuda.get_device_properties(0).total_memory < 11 * 2**30:
            raise RuntimeError("The configured run requires at least 12 GB GPU RAM.")
        if settings["max_texts"] > 1000 and psutil.virtual_memory().total < 45 * 2**30:
            raise RuntimeError(
                "Select a high-RAM runtime (>=48 GB) for the explicit larger configuration."
            )
        print(
            "Python",
            sys.version,
            "Torch",
            torch.__version__,
            "CUDA",
            torch.version.cuda,
            "GPU",
            torch.cuda.get_device_name(),
            flush=True,
        )
        hf_hub_download(cfg.model.model_name, "config.json")
    elif stage == "collect":
        collect(cfg)
    elif stage == "analyze":
        analyze(cfg, rcfg)
    elif stage == "controls":
        from sae_feature_atlas.scientific.controls import weak_controls

        weak_controls(cfg, rcfg)
    elif stage == "robustness":
        from sae_feature_atlas.scientific.robustness import one_per_document

        one_per_document(cfg, rcfg)
    elif stage == "report":
        from sae_feature_atlas.scientific.report import report

        report(cfg, rcfg)
        manifest(cfg, rcfg)
    elif stage == "export":
        from sae_feature_atlas.scientific.export import export_bundle

        export_bundle(cfg)
    else:
        raise ValueError(stage)


if __name__ == "__main__":
    main(*sys.argv[1:])
