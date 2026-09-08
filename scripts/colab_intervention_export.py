"""Verify and export completed pilot data before releasing the Colab runtime."""

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

root = Path("/content/sae-context-pilot")
results = root / "full_v3"
complete = json.loads((results / "complete.json").read_text())
plan = json.loads((root / "pilot/plan.json").read_text())
if complete["prompts"] != plan["prompts"]:
    raise ValueError("Full pilot is incomplete")
for name, expected in complete["files"].items():
    if hashlib.sha256((results / name).read_bytes()).hexdigest() != expected:
        raise ValueError("Corrupt prompt data")
(root / "environment.json").write_text(
    subprocess.check_output(
        [
            "/content/sae-env/bin/python",
            "-c",
            "import importlib.metadata as m,json,sys,torch; from datetime import datetime,timezone; "
            "print(json.dumps(dict(python=sys.version,exported_at_utc=datetime.now(timezone.utc).isoformat(),"
            "gpu=torch.cuda.get_device_name(),packages={d.metadata['Name']:d.version for d in m.distributions()}),indent=2))",
        ],
        text=True,
    )
)
output = Path("/content/context_intervention_results.zip")
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
    for folder in [root / "pilot", results]:
        for path in sorted(folder.iterdir()):
            if path.is_file():
                z.write(path, str(path.relative_to(root)))
    z.write(root / "environment.json", "environment.json")
    z.write(root / "job.log", "job.log")
    z.write("/content/context_intervention_bundle.zip", "executed_bundle.zip")
print("ARCHIVE_SHA256", hashlib.sha256(output.read_bytes()).hexdigest())
print("ARCHIVE_BYTES", output.stat().st_size)
print("PROMPTS", complete["prompts"])
