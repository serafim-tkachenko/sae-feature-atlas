# Inspection flow

Inspection has two layers.

## Automated inspection step

The pipeline step:

```bash
uv run sae-atlas run --steps inspection --run-name <run>
```

reads activation rows, top examples, analysis features, and optionally
coactivation/bimodality artifacts. It writes:

```text
inspection_feature_summaries.parquet
inspection_pair_summaries.parquet
reports/<run_name>/inspection_report.md
reports/<run_name>/inspection_report.json
```

The goal is triage: detect artifacts, formatting-heavy features, token
concentration, suspicious coactivation pairs, and features worth manual review.

## Manual inspection commands

Start with the saved run's data directory to list artifact availability and,
optionally, a feature's saved card and top examples:

```bash
uv run sae-atlas inspect-run data/processed/<run_name>
uv run sae-atlas inspect-run data/processed/<run_name> --feature 1645 --examples 5
```

This command reads saved tables without importing the model runtime. It also
works on partial runs: missing optional tables produce a message. The example
script `examples/inspect_run.py` and notebook-friendly `AtlasRun` API remain
available for the same workflow.

For detailed feature, pair, or activation-regime contexts:

```bash
uv run sae-atlas inspect-feature --run-name <run> --feature-id <id> --n 10
uv run sae-atlas inspect-pair --run-name <run> --feature-i <id1> --feature-j <id2> --n 10
uv run sae-atlas inspect-bimodal-feature --run-name <run> --feature-id <id> --n 6
```

These commands are intentionally read-only. They do not generate new research
claims; they make existing artifacts easier to inspect.
