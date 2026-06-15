# Concepts

## SAE feature atlas

A feature atlas is a table/report that summarizes how SAE features behave on a corpus:

- how often they activate,
- where they activate,
- which examples activate them most,
- which features co-activate,
- which decoder directions are geometrically close,
- whether activation strength looks multi-modal

## Feature cards

A feature card is one row per SAE feature - it is the main reusable artifact

## Decoder geometry vs co-activation

Two features can be close in SAE decoder space but not co-activate, or co-activate often while pointing in different decoder directions

## Bimodality

Some features may have weak and strong activation regimes - this can matter for explanations and steering
