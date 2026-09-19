# What the toolkit does

SAE Feature Atlas collects activations from a text model and its pretrained sparse autoencoder, then connects each saved activation to its source text and token position.

Use it to inspect activation examples, count feature occurrence with explicit token denominators, compare coactivation with decoder similarity, and explore activation distributions. The `AtlasRun` reader opens saved tables from Python without model inference.

The toolkit grew out of the [SAE context study](https://github.com/serafim-tkachenko/model-behavior-research). Study-specific hypothesis tests and interventions are maintained there. Atlas supplies measurements and examples; interpreting a feature's meaning or causal role requires a separate experiment.

Start with the [quickstart](quickstart.md), then follow the [analysis workflow](analysis_workflow.md). The [metrics guide](metrics_guide.md) explains denominators and the difference between saved top-K membership and all positive activations.
