# Quickstart

## 1. Install

```bash
uv sync
uv run huggingface-cli login
```

## 2. Smoke test

```bash
uv run sae-atlas smoke-test --model gemma-3-1b-pt --layer 13
```

## 3. Small debug run

```bash
uv run sae-atlas run \
  --model gemma-3-1b-pt \
  --layer 13 \
  --max-texts 100 \
  --top-k 32 \
  --steps collect,features,report
```

## 4. Full pilot run

```bash
uv run sae-atlas run \
  --model gemma-3-1b-pt \
  --layer 13 \
  --max-texts 1500 \
  --top-k 64 \
  --steps all
```

Open:

```text
reports/<run_name>/index.html
```
