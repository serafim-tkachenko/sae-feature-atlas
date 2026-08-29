"""Package research code and frozen text samples for the user's Colab runtime."""

import hashlib
import json
from pathlib import Path
import zipfile

files = []
for folder in ("src", "tests", "scripts", "requirements", "docs"):
    files.extend(
        p
        for p in Path(folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
files += [Path(name) for name in ("pyproject.toml", "uv.lock", "README.md")]
for folder in ("data/raw/foundation_v1", "data/raw/foundation_smoke_v2"):
    files.extend(p for p in Path(folder).glob("*") if p.is_file())
manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
output = Path("outputs/foundation_colab_bundle.zip")
output.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files):
        z.write(p, str(p))
    z.writestr("foundation_bundle_manifest.json", json.dumps(manifest, indent=2))
print(output, output.stat().st_size, "bytes")
