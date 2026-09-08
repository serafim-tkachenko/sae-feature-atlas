"""Package code and small frozen pilot inputs, excluding credentials and raw corpora."""

import argparse
import json
from pathlib import Path
import zipfile

from sae_feature_atlas.scientific.collect import sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prepared", type=Path, required=True)
parser.add_argument("--out", type=Path, default=Path("outputs/context_intervention_bundle.zip"))
args = parser.parse_args()
plan = json.loads((args.prepared / "plan.json").read_text())
for name, digest in plan["files"].items():
    if sha256(args.prepared / name) != digest:
        raise ValueError("Prepared inputs changed")
files = [
    (p, str(p))
    for folder in ["src", "experiments", "requirements"]
    for p in Path(folder).rglob("*")
    if p.is_file() and "__pycache__" not in p.parts
]
files += [(Path(n), n) for n in ["pyproject.toml", "README.md"]]
files += [(args.prepared / n, "pilot/" + n) for n in ["plan.json", *plan["files"]]]
args.out.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
    for p, name in files:
        z.write(p, name)
    z.writestr(
        "intervention_bundle_manifest.json", json.dumps({name: sha256(p) for p, name in files})
    )
print(args.out, args.out.stat().st_size, "bytes")
