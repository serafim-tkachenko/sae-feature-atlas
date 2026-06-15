# Corpus guide

Corpus choice strongly affects feature interpretation

## Recommended workflow

Debug:

```bash
--corpus pile-10k --max-texts 100 --top-k 32
```

Local pilot:

```bash
--corpus mixed-research --max-texts 300 --top-k 32
```

Colab/larger run:

```bash
--corpus mixed-research --max-texts 1000 --top-k 64
```

Educational text:

```bash
--corpus fineweb-edu-sample --max-texts 1000 --top-k 64
```

## Presets

### `pile-10k`

Diverse small sample - good default

### `tinystories`

Simple narratives - useful for smoke tests but narrow

### `fineweb-edu-sample`

Streaming educational web text - higher quality but heavier

### `mixed-research`

Mixed corpus combining Pile-style text, educational text when available, and manual technical snippets

### `jsonl:/path/file.jsonl`

Custom corpus. Each row should contain:

```json
{"source": "my-source", "text": "Text goes here."}
```
