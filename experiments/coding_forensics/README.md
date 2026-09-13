# Coding repair boundary feasibility study

A small, separate behavioral study alongside Feature Atlas. The current result is a failed behavior gate: all 48 paired-development trajectories repaired the task without modifying protected tests. It does not justify a larger run of this assay.

Read protocol.md for the registered development decisions and RESULTS.md for outcomes, deviations and limits.

Run from the repository root on Linux/WSL with the project environment and bubblewrap installed:

    .venv/bin/python experiments/coding_forensics/pilot.py --validate
    .venv/bin/python experiments/coding_forensics/pilot.py --run outputs/coding_forensics/new_run
    .venv/bin/python experiments/coding_forensics/audit.py outputs/coding_forensics/new_run

Model weights must already be cached at the exact revision in pilot.py. The run refuses to overwrite an existing manifest. CPU evaluation uses the system Python inside bubblewrap; generated code never runs directly in the model process. The experiment does not require SAE weights.

A run saves its exact runner, protocol, dataset, source digests, model settings, prompts, token IDs, outputs and per-trajectory results. The audit emits per_rollout.csv, raw_review.json, audit.json, a PNG/SVG figure, and an evidence ZIP beside the run folder. Large/generated outputs remain excluded from Git.

The first aborted development run is outputs/coding_forensics/dev_v1; the completed corrected run is outputs/coding_forensics/dev_v2. The v2 executed runner is preserved byte-for-byte. Subsequent lint cleanup only split two multi-name imports; no experiment logic changed. Use the archived runner and protocol for exact provenance.

No final test set has been created, no larger run is justified by this pilot, and no inference about model intent is made.
