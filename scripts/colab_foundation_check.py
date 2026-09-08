"""Inspect runtime resources and gated-model access without printing credentials."""

import subprocess

print(subprocess.check_output(["nvidia-smi"], text=True))
print(subprocess.check_output(["free", "-h"], text=True))
print(
    subprocess.check_output(
        [
            "/content/sae-env/bin/python",
            "-c",
            """
import torch
from huggingface_hub import hf_hub_download
print('TORCH', torch.__version__, 'CUDA', torch.version.cuda)
print('GPU', torch.cuda.get_device_name())
try:
    hf_hub_download('google/gemma-3-4b-pt', 'config.json')
    print('MODEL_ACCESS_OK')
except Exception as exc:
    print('MODEL_ACCESS_FAILED', type(exc).__name__)
""",
        ],
        text=True,
    )
)
