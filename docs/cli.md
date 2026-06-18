# CLI guide

The project exposes one command:

```bash
uv run sae-atlas <command>
```

## Discovery commands

```bash
uv run sae-atlas list-models
uv run sae-atlas list-sites
uv run sae-atlas list-corpora
uv run sae-atlas list-presets
uv run sae-atlas resolve-sae --model gemma-3-1b-pt --layer 13 --site resid_post
```

Site aliases are accepted:

```text
res  -> resid_post
mlp  -> mlp_out
att  -> attn_out
```

## Planning and validation

```bash
uv run sae-atlas plan --preset research --model gemma-3-1b-pt --layer 13 --max-texts 100
uv run sae-atlas smoke-test --model gemma-3-1b-pt --layer 13
```

`plan` shows selected steps and artifact readiness. `smoke-test` loads the model
and SAE to validate the runtime configuration.

## Running

```bash
uv run sae-atlas run --preset core --max-texts 50
uv run sae-atlas run --preset atlas --max-texts 500
uv run sae-atlas run --preset research --max-texts 1500
```

You can also run selected steps:

```bash
uv run sae-atlas run --steps features,coactivation,geometry
```

Available presets:

```text
core      minimal activation collection
atlas     general feature-atlas workflow
research  atlas plus residual-space coverage and graph-alignment metrics
```

## Inspection commands

```bash
uv run sae-atlas inspect-feature --run-name <run> --feature-id 123
uv run sae-atlas inspect-pair --run-name <run> --feature-i 123 --feature-j 456
uv run sae-atlas inspect-bimodal-feature --run-name <run> --feature-id 123
```
