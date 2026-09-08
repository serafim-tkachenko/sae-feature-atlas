"""Executed through the official Colab CLI after uploading the research bundle."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

root = Path("/content/sae-foundation")
root.mkdir(exist_ok=True)
with zipfile.ZipFile("/content/foundation_colab_bundle.zip") as z:
    for name in z.namelist():
        if not (root / name).resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe archive path")
    z.extractall(root)
os.chdir(root)
for name, expected in json.loads(Path("foundation_bundle_manifest.json").read_text()).items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Bundle hash mismatch: {name}")
subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.9.28"], check=True)
subprocess.run(["uv", "python", "install", "3.11"], check=True)
subprocess.run(["uv", "venv", "--python", "3.11", "/content/sae-env"], check=True)
py = "/content/sae-env/bin/python"
subprocess.run(
    [
        "uv",
        "pip",
        "sync",
        "--python",
        py,
        "--require-hashes",
        "requirements/colab.lock",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu126",
        "--index-strategy",
        "unsafe-best-match",
    ],
    check=True,
)
subprocess.run(["uv", "pip", "install", "--python", py, "--no-deps", "-e", "."], check=True)
print("SETUP COMPLETE", flush=True)
subprocess.run(["nvidia-smi"], check=True)
subprocess.run(["free", "-h"], check=True)
