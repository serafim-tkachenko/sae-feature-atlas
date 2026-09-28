# Contributing

Use Python 3.11 for the validated environment. Run `uv sync --locked`, `uv run pytest -q` and `uv run python -m compileall -q src` before submitting a change.

Keep APIs and artifact schemas explicit. Changes to sampling, token populations or collection identity need meaningful tests and migration notes. Never silently reuse incompatible artifacts. Tests should use small fixtures; GPU/network tests must be explicit and record model revision, precision, device and runtime.

Do not commit credentials, model checkpoints, private corpora or coordination notes. Report reproducible software problems through GitHub issues without secrets.

## Browser explorer

The React/TypeScript source is in `web/`. Development requires Node 22.12+ and npm;
viewers of generated reports do not need either. From `web/`, run:

```bash
npm ci
npm run build
npx playwright install chromium
npm test
```

The build type-checks the frontend and writes deterministic JS/CSS into
`src/sae_feature_atlas/report/static/`. Commit those bundles with source changes;
Python wheels and source distributions include them. CI rebuilds and checks for
drift. Browser tests generate small synthetic Parquet fixtures through the real
exporter, then open HTML directly from disk with networking disabled. They require
the project's `uv` environment. On Linux, browser dependencies can be installed
with `npx playwright install --with-deps chromium`.

Format frontend source with `npx prettier --write src tests/*.ts *.ts` from `web/`.
Keep evidence and metadata contracts explicit in `report/explorer.py` and
`web/src/types.ts`; update both and the browser fixtures when changing the schema.
Use plain SVG charts for the current scatterplot and histogram to keep the
offline bundle small. Do not embed source corpora or real report exports in tests.
