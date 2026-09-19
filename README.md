# SAE Feature Atlas

Collect sparse-autoencoder activations and inspect the text behind them.

The toolkit helps answer three practical questions: how often does a feature activate, in which token contexts, and which other features activate alongside it? It saves token metadata with the activations, so frequency estimates include tokens where a feature has no saved activation. Decoder geometry and residual-PCA diagnostics are available for comparing representation structure.

It was developed for a [study of SAE context and intervention effects](https://github.com/serafim-tkachenko/model-behavior-research). The study code and conclusions live in that repository; this package provides the reusable collection and analysis tools.

## Install

Python 3.10-3.12 is supported; the reference environment uses Python 3.11.

~~~bash
git clone https://github.com/serafim-tkachenko/sae-feature-atlas.git
cd sae-feature-atlas
uv sync --locked
uv run sae-atlas --help
~~~

## Collect a small run

Collection loads the model and SAE through PyTorch. Accept the relevant model access terms and authenticate with `uv run hf auth login` first.

~~~bash
uv run sae-atlas plan --preset atlas --model gemma-3-1b-pt --layer 13 --max-texts 10 --max-seq-len 128 --top-k 32
uv run sae-atlas smoke-test --model gemma-3-1b-pt --layer 13
uv run sae-atlas run --preset atlas --model gemma-3-1b-pt --layer 13 --max-texts 10 --max-seq-len 128 --top-k 32
~~~

`plan` prints the configuration without loading weights. `smoke-test` loads the model and SAE to check compatibility. `run` writes artifacts under `data/processed/<run_name>/` and a report under `reports/<run_name>/`.

## Inspect an existing run

Saved tables can be read on CPU, without loading model weights:

~~~bash
uv run python examples/inspect_run.py data/processed/<run_name> --feature 1645
~~~

Or use the same reader from a notebook:

~~~python
from sae_feature_atlas.storage import AtlasRun

run = AtlasRun.from_dir("data/processed/<run_name>")
print(run.artifact_status())
cards = run.feature_cards()
examples = run.feature_examples(1645, n=5)
pairs = run.coactivation_pairs()
~~~

The feature ID is dictionary-specific; choose one from your run's cards. Optional analysis tables are empty when that stage has not been run. See [inspection](docs/inspection.md) and the [analysis workflow](docs/analysis_workflow.md).

## Choose the collection and analysis scope

| Option | What it does |
| --- | --- |
| `core` preset | Collects activations and token metadata. |
| `atlas` preset | Adds feature statistics, examples, coactivation and reports. |
| `research` preset | Adds decoder/residual-PC and graph-alignment diagnostics. |
| `topk` activation mode | Saves only the largest K activations at each token. Frequency means retained membership. |
| `positive` activation mode | Saves all positive activations; storage and analysis costs can be much larger. |

A close decoder direction is not necessarily a coactivating feature. A missing empirical edge is not necessarily an observed zero. Activation clusters and two-component fits do not establish separate meanings. The [metrics guide](docs/metrics_guide.md) defines the quantities and their denominators.

`lineage.json` records configuration fingerprints, source revision and artifact schema so analysis outputs can be traced to their inputs. The older `coverage` command is retained as an alias for decoder/residual-PC alignment; it does not measure reconstruction coverage.

## Documentation and development

[Quickstart](docs/quickstart.md) · [CLI](docs/cli.md) · [Configuration](docs/configuration.md) · [All documentation](docs/README.md) · [Migration notes](docs/migration.md)

~~~bash
uv run pytest -q
uv run python -m compileall -q src
~~~

Unit tests use small fixtures without model downloads. GPU compatibility is a separate check. See [contributing](CONTRIBUTING.md) for development details.

Code and original documentation are MIT licensed. Model weights and source datasets retain their own licenses.
