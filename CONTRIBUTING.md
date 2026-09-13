# Contributing

Use Python 3.11 for the validated environment. Run `uv sync --locked`, `uv run pytest -q` and `uv run python -m compileall -q src` before submitting a change.

Keep APIs and artifact schemas explicit. Changes to sampling, token populations or collection identity need meaningful tests and migration notes. Never silently reuse incompatible artifacts. Tests should use small fixtures; GPU/network tests must be explicit and record model revision, precision, device and runtime.

Do not commit credentials, model checkpoints, private corpora or coordination notes. Report reproducible software problems through GitHub issues without secrets. Study-specific code belongs in the research repository.
