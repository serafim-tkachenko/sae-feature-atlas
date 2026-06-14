# CLI

The main command is:

```bash
sae-atlas run
```

## Commands

```bash
sae-atlas list-models
sae-atlas resolve-sae --model gemma-3-1b-pt --layer 13
sae-atlas smoke-test --model gemma-3-1b-pt --layer 13
sae-atlas run --model gemma-3-1b-pt --layer 13 --steps all
sae-atlas report --run-name gemma3_1b_l13_res16k_pile10k_top64
```

## Pipeline steps

Available steps:

```text
collect
features
coactivation
geometry
geometry-vs-coactivation
bimodality
report
all
```

Examples:

```bash
sae-atlas run --model gemma-3-1b-pt --layer 13 --steps collect,features
sae-atlas run --model gemma-3-1b-pt --layer 13 --steps coactivation,geometry,report
```
