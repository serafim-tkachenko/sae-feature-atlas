"""Download pinned resources; consume and remove the temporary authentication file."""

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

credential = Path("/content/.foundation_hf_download_token")
token = credential.read_text().strip()
credential.unlink()
root = Path("/content/sae-foundation")
manifest = json.loads((root / "data/raw/foundation_v1/corpus_manifest.json").read_text())
snapshot_download(
    "google/gemma-3-4b-pt",
    revision=manifest["tokenizer_revision"],
    token=token,
    allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt", "*.jinja"],
)
repo = "google/gemma-scope-2-4b-pt"
revision = HfApi(token=token).model_info(repo).sha
for layer in (17, 22):
    for name in ("config.json", "params.safetensors"):
        hf_hub_download(
            repo,
            f"resid_post/layer_{layer}_width_65k_l0_medium/{name}",
            revision=revision,
            token=token,
        )
print("PINNED WEIGHTS READY", flush=True)
