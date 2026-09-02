"""Package the colleague report separately from multi-gigabyte research data."""

from pathlib import Path
import hashlib
import json
import zipfile


def main():
    root = Path("reports/gemma4b_foundation_v1")
    assert (root / "scientific_report.pdf").exists()
    assert (root / "verification.json").exists()
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.name != "report_manifest.json"]
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (root / "report_manifest.json").write_text(json.dumps(manifest, indent=2))
    output = Path("outputs/gemma4b_foundation_report.zip")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, str(p.relative_to(root)))
        z.write(root / "report_manifest.json", "report_manifest.json")
        z.write("docs/foundation_handoff.md", "reproduction_guide.md")
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
    print(output, output.stat().st_size)


if __name__ == "__main__":
    main()
