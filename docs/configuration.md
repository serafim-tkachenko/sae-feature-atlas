# Configuration

The recommended user-facing configuration is model/layer first:

```bash
sae-atlas run --model gemma-3-1b-pt --layer 13
```

The library automatically resolves the Gemma Scope SAE

## Main parameters

| Parameter | Meaning | Default |
|---|---|---|
| `--model` | Model alias, for example `gemma-3-1b-pt` | `gemma-3-1b-pt` |
| `--layer` | Transformer layer to analyze | model-specific default |
| `--site` | SAE site: `resid_post`, `mlp_out`, `attn_out` | `resid_post` |
| `--width` | SAE width | `16k` |
| `--l0` | SAE sparsity variant | `medium` |
| `--corpus` | Corpus loader | `pile-10k` |
| `--max-texts` | Number of texts | `1500` |
| `--max-seq-len` | Max tokens per text | `256` |
| `--activation-mode` | `topk` or `positive` | `topk` |
| `--top-k` | Number of features per token in top-k mode | `64` |
| `--steps` | Pipeline steps to run | `all` |

## Important caveat about top-k

If `activation_mode=topk`, feature frequency means:

-> how often the feature appears among the saved top-K activations

It does not mean:

-> how often the feature has positive activation.

Use `activation_mode=positive` for more faithful frequency estimates
