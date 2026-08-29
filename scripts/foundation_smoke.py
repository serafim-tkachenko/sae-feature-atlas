"""Run a separate 32-document native 4B benchmark before the foundation collection."""

import json
import argparse
import time
from pathlib import Path

import torch

from sae_feature_atlas.scientific.collect import collect, configuration
from sae_feature_atlas.scientific.corpus import text_digest
from sae_feature_atlas.util.io import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--layer", type=int, choices=[17, 22], default=17)
args = parser.parse_args()
smoke = Path("data/raw/foundation_smoke_v2/texts.jsonl")
main = Path("data/raw/foundation_v1/texts.jsonl")
smoke_rows = [json.loads(line) for line in smoke.read_text().splitlines()]
main_hashes = {text_digest(json.loads(line)["text"]) for line in main.read_text().splitlines()}
if any(text_digest(row["text"]) in main_hashes for row in smoke_rows):
    raise ValueError("Engineering smoke documents overlap the research corpus.")
cfg = configuration(
    "gemma4b_foundation_smoke_v1" + ("_l22" if args.layer == 22 else ""),
    len(smoke_rows),
    512,
    model="gemma-3-4b-pt",
    layer=args.layer,
    width="65k",
    l0="medium",
    corpus="foundation-smoke",
)
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
collect(cfg, corpus_path=smoke, local_files_only=True)
elapsed = time.perf_counter() - started
output_bytes = sum(p.stat().st_size for p in cfg.run_data_dir.glob("*.parquet"))
result = {
    "documents": len(smoke_rows),
    "elapsed_seconds_including_load": elapsed,
    "gpu": torch.cuda.get_device_name(),
    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    "consolidated_output_mib": output_bytes / 2**20,
    "linear_12000_document_hours_including_repeated_load_overhead": elapsed
    * 12000
    / len(smoke_rows)
    / 3600,
    "interpretation": "Engineering check only; loading overhead makes this a conservative throughput estimate.",
}
write_json(cfg.run_data_dir / "benchmark.json", result)
print(json.dumps(result, indent=2))
