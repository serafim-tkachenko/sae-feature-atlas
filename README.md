# SAE Feature Atlas

A Python toolkit and CLI for collecting and inspecting sparse-autoencoder activations. It provides explicit token populations, feature statistics, coactivation, decoder geometry, decoded context evidence and reproducible artifact lineage.

**Research studies have moved to [Model Behavior Research](https://github.com/serafim-tkachenko/model-behavior-research).** That repository contains study reports, experiment runners, notebooks, numerical evidence and coding-forensics pilots. This repository owns the reusable library and its user documentation.

## Install

Python 3.10–3.12 is supported by the package; validation is performed on Python 3.11. Model collection requires a compatible PyTorch device and access to the selected model/SAE. Analysis of saved artifacts can run on CPU.

~~~bash
git clone https://github.com/serafim-tkachenko/sae-feature-atlas.git
cd sae-feature-atlas
uv sync --locked
uv run sae-atlas --help
uv run sae-atlas list-models
uv run sae-atlas list-presets
~~~

Accept any required model access terms on Hugging Face and authenticate with `uv run hf auth login`. Do not put access tokens in source files or notebooks.

## Plan, validate and run

~~~bash
uv run sae-atlas plan --preset atlas --model gemma-3-1b-pt --layer 13 --max-texts 10 --max-seq-len 128 --top-k 32
uv run sae-atlas smoke-test --model gemma-3-1b-pt --layer 13
uv run sae-atlas run --preset atlas --model gemma-3-1b-pt --layer 13 --max-texts 10 --max-seq-len 128 --top-k 32
~~~

`plan` resolves configuration without collection. `smoke-test` loads model/SAE weights; it is not an offline unit test. Start small and measure memory before scaling.

Presets: `core` collects activations; `atlas` adds descriptive analysis, inspection and reports; `research` adds decoder/residual-PC and graph alignment diagnostics. The preset name is retained for API compatibility.

## Python API

~~~python
from sae_feature_atlas import make_config
from sae_feature_atlas.storage import AtlasRun

cfg = make_config(model="gemma-3-1b-pt", layer=13, max_texts=10, top_k=32)
run = AtlasRun.from_dir("data/processed/<run_name>")
print(run.artifact_status())
cards = run.feature_cards()
~~~

See the [runnable example](examples/simple_usage.py), [quickstart](docs/quickstart.md), [CLI reference](docs/cli.md), [configuration](docs/configuration.md) and [documentation index](docs/README.md).

## Artifact and interpretation contract

Collection writes token metadata, source identities, sparse activations and residual samples beneath `data/processed/<run_name>/`. Reports live beneath `reports/<run_name>/`. `lineage.json` records schema, configuration fingerprints, source revision and dirty state; incompatible analysis outputs must be regenerated.

- In `topk` mode, frequencies measure retained memberships, not all positive activations.
- Token metadata defines denominators, including tokens with no retained row for a feature.
- Missing empirical edges are not necessarily observed zeros.
- PCA, UMAP, decoder similarity and bimodality are descriptive diagnostics, not semantic or causal mechanisms.
- The legacy `coverage` command names a decoder/residual-PC alignment diagnostic, not reconstruction coverage.

Read [concepts](docs/concepts.md), [metrics](docs/metrics_guide.md) and [analysis workflow](docs/analysis_workflow.md) before interpreting results.

## Development

~~~bash
uv sync --locked
uv run pytest -q
uv run python -m compileall -q src
~~~

Tests use small fixtures without model downloads. GPU compatibility is checked separately with the device and precision recorded. See [contributing](CONTRIBUTING.md), [migration](docs/migration.md) and [changelog](CHANGELOG.md).

Original code and documentation are MIT licensed. Models and source datasets retain their own licenses; model weights are not distributed here.
