"""Use the prepared Colab environment for a bounded development pilot."""

import json
import os
from pathlib import Path
import subprocess
import zipfile
import hashlib

root = Path("/content/sae-context-pilot")
root.mkdir(exist_ok=True)
with zipfile.ZipFile("/content/context_intervention_bundle.zip") as z:
    for name in z.namelist():
        if not (root / name).resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe bundle path")
    z.extractall(root)
for name, expected in json.loads((root / "intervention_bundle_manifest.json").read_text()).items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise ValueError("Corrupt bundle")
py = "/content/sae-env/bin/python"
# The locked environment already exists. PYTHONPATH selects this exact source bundle.
env = dict(os.environ, PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="4")
script = root / "download_and_smoke.py"
script.write_text("""
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from sae_feature_atlas.scientific.intervention_run import run
root = Path('/content/sae-context-pilot')
plan = json.loads((root/'pilot/plan.json').read_text())
credential = Path('/content/.context_hf_download_token')
token = credential.read_text().strip() if credential.exists() else None
credential.unlink(missing_ok=True)
snapshot_download(plan['model']['model_name'], revision=plan['model_revision'], token=token,
    allow_patterns=['*.json', '*.model', '*.safetensors', '*.txt', '*.jinja'])
run(root/'pilot', root/'full_v3', local_files_only=True)
""")
with (root / "job.log").open("w") as log:
    proc = subprocess.Popen(
        [py, "-u", str(script)],
        cwd=root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print("PILOT_PROCESS", proc.pid, "LOG", root / "job.log")
