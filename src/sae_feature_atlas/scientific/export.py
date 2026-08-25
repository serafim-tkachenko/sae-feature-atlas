"""Portable source archive and complete analysis bundle without duplicate chunks."""

from pathlib import Path
import json
import zipfile

from sae_feature_atlas.pipeline.lineage import git_provenance


def source_archive(path="outputs/sae_feature_atlas_source.zip"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    roots = ["src", "tests", "scripts", "docs", "notebooks", "requirements"]
    files = [Path(f) for f in ["README.md", "pyproject.toml", "uv.lock", ".gitignore"]]
    for root in roots:
        files += [p for p in Path(root).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            archive.write(p, p.as_posix())
        archive.writestr("source_provenance.json", json.dumps(git_provenance(), indent=2))
    return path


def export_bundle(cfg):
    source = source_archive()
    output = Path("outputs") / f"{cfg.collection.run_name}_analysis_bundle.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(source, source.name)
        for path in sorted(cfg.run_data_dir.iterdir()):
            if path.is_file():
                archive.write(path, "data/" + path.name)
        for path in sorted(cfg.run_reports_dir.rglob("*")):
            if path.is_file():
                archive.write(path, "report/" + path.relative_to(cfg.run_reports_dir).as_posix())
    return output
