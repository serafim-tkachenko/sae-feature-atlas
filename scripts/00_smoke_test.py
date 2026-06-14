from __future__ import annotations

import json

from sae_feature_atlas.config import CFG
from sae_feature_atlas.model_utils import (
    get_device,
    load_model,
    load_sae,
    validate_model_sae_compatibility,
)


def main() -> None:
    device = get_device()
    print("Device:", device)

    sae = load_sae(CFG, device)
    print("Loaded SAE")
    print(sae.cfg)

    model = load_model(CFG, device)
    print("Loaded model")

    result = validate_model_sae_compatibility(model, sae, CFG, device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
