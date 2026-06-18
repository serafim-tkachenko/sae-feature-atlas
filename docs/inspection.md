# Inspection flow

Inspection has two layers.

## Automated inspection step

The pipeline step:

```bash
uv run sae-atlas run --steps inspection --run-name <run>
```

reads activation rows, top examples, filtered features, and optionally
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

```bash
uv run sae-atlas inspect-feature --run-name <run> --feature-id <id> --n 10
uv run sae-atlas inspect-pair --run-name <run> --feature-i <id1> --feature-j <id2> --n 10
uv run sae-atlas inspect-bimodal-feature --run-name <run> --feature-id <id> --n 6
```

These commands are intentionally read-only. They do not generate new research
claims; they make existing artifacts easier to inspect.
