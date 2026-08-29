# Activation strength and SAE neighborhoods: foundation experiment

Status: prospective design under implementation; no confirmatory results yet.
The existing 1B/Pile experiment is pilot evidence and informed this design.

## Question and scope

Does a discovery-defined activation-strength regime predict a feature's
coactivation neighborhood on unseen documents, within each of two text sources?
The claim concerns conditional associations in a fixed pretrained model and SAE.
Neither a two-Gaussian fit nor a significant neighborhood difference establishes
semantic polysemy, discrete concepts, or causal interactions.

## Resources and sampling

Use Gemma 3 4B pretrained, residual-stream SAEs at layers 17 and 22 (subject to
native checkpoint verification), width 65,536 (`65k`) and medium sparsity. Layer 17 is the
primary analysis; layer 22 is a prespecified depth sensitivity, not an independent
model replication. Record actual checkpoint L0 and measured activation density.

Use the existing `fineweb-edu-sample` and `wikimedia-en` configuration descriptors:
HuggingFaceFW/fineweb-edu, sample-10BT; wikimedia/wikipedia, 20231101.en.
Target 6,000 documents per source, at most 512 tokens each. This is a source-balanced
sample, not a representative estimate of all language use. Preserve source labels
and report both document and eligible-token counts per source and split.

Resolve and pin dataset revisions before sampling. Seed the streaming shuffle,
record its finite buffer size and source-row identifiers, and save the selected
texts. Do not describe buffer shuffling as uniform sampling over the entire corpus.
Remove exact duplicates after Unicode/whitespace normalization globally. Audit
near duplicates and keep detected related documents in the same split. The
sampling manifest must record exclusions. Source failures stop preparation;
there is no fallback to Pile or substitution of manual probes.

Exclude TinyStories and the manual math/code probes from the primary population;
their synthetic construction defines a different target population. OpenWebText
is available as a preset but is not part of this two-source design.

## Discovery and evaluation

Split duplicate groups before activation analysis using a seeded hash, yielding
approximately half discovery and half evaluation within each source. Freeze
source membership, duplicate groups, random seeds, model/SAE
revisions, token eligibility, and sample size before opening evaluation results.
Exclude all pilot documents and separate engineering smoke documents.

Screen a uniform seeded subset of 2,048 discovery-supported features per layer;
select at most 24 using discovery criteria only. Fit five-start two-component
GMMs to log1p positive activation, delta BIC >=10, component weights >=0.1,
separation >=2. Require discovery support >=500 tokens in >=100 documents.
Preserve ambiguous assignments; confident posterior threshold is 0.9.

Retain every positive SAE activation without top-k pruning. Freeze candidate
selection and fitted parameters before evaluating either source. Analyze sources
separately to prevent a change in corpus mixture from defining the main effect.

The primary endpoint uses one uniformly selected confident observation per
duplicate group per feature and source, sampled without consulting partner structure or regime
labels. Require >=100 documents in each regime within each evaluation source.
Use token-identity, position (64-token bins), and positive-support (16-feature
bins) conditional permutations. Report movable fraction and support failures.
Conditional exchangeability is still an assumption, not a consequence of sampling.

Use JS divergence in bits; save signed partner changes and top-neighbor stability.
Run 9,999 permutations and 1,000 document bootstraps. Include failures as p=1.
For each feature, use the maximum of the two source-specific p-values as a
conjunction test; apply BY across the 24 selected primary-layer candidates.
This tests evidence in both source populations, not identity of the changed
partners. Report partner-direction agreement separately. The second layer is
reported as a distinct prespecified secondary family.

## Controls and interpretation

Prospectively include strict low-BIC controls and weak-separation controls,
matched on discovery activation/document support without replacement. Keep
these populations separate and report unmatched candidates. Lack of matched
controls cannot be interpreted as candidate specificity. Compare frozen quantile
tails at equal evaluation regime counts and report paired effects with uncertainty.

Retain the full-token analysis as a secondary sensitivity. Audit lexical overlap,
position, source, observed sparsity, sample-size effects, and permutation
degeneracy. Export blind, document-disjoint context samples for colleague
annotation; semantic conclusions require completed annotation, not examples
selected after seeing interesting effects.

## Execution and stopping

Run a disjoint smoke collection first to verify native hooks, checkpoint tensors,
finite activations, reconstruction, throughput, peak VRAM, host memory and output
size. Fix the final execution manifest after that benchmark and before evaluation.
Do not reduce precision or sample size silently. Do not extend the sample in
response to significance. Technical failures may be resumed from checked chunks.

Execution uses the user's Colab account, an A100-SXM4 40 GB GPU and 83 GiB host
RAM. The separate layer-17 engineering run completed 32 documents in 20.73 seconds
including loading; peak reserved GPU memory was 9.58 GiB. The final seven-document
chunk took 0.60 seconds, giving a rough steady-state projection of 17 minutes per
12,000-document layer. This short benchmark does not establish a precise runtime
or estimate statistical-analysis cost. The first-document relative reconstruction
MSE was 0.00151 and mean positive support was 58.3, against target L0=60.
The scientific evaluation has not been opened during preparation.

Preparation completed: 12,000 documents, zero normalized exact-text overlap with
the 1B pilot. FineWeb-Edu has 2,965 discovery and 3,035 evaluation documents;
Wikipedia has 3,054 discovery and 2,946 evaluation documents. The token-window
audit found two near-duplicate links. Candidate detection uses 64 MinHash
permutations in 16 bands of four, followed by exact token-5-shingle Jaccard >=0.8.
Detected connected groups share a split. Retrieval is approximate and does not
guarantee that every near duplicate was detected. The native layer-17 checkpoint
was verified to declare 65,536 features, target L0=60 and the expected model/hook.

## Deliverables

Versioned protocol and amendments, pinned source manifest, resumable raw sparse
activations, frozen discovery fits, all selected candidates including failures,
source-specific effects/nulls/uncertainty, controls, blind annotation material,
and a readable report with engineering provenance in a separate appendix.
The final report must distinguish completed evidence from remaining validation.

## Reproduction entry points

The foundation run uses `python -m sae_feature_atlas.scientific.foundation` with
`--layer 17` or `--layer 22`, and explicit `--stage collect`, `analyze`, `confirm`,
or `controls`. Prepared inputs are in `data/raw/foundation_v1`. Model files are
downloaded at the pinned revision first, then loaded locally; no authentication
credential is included in the source/data bundle. The Colab CLI transport uses
google-colab-cli 0.6.0 with jupyter-kernel-client 0.15.0 (the unbounded 1.0.2
dependency was incompatible). Subprocesses set MPLBACKEND=Agg to avoid inheriting
the notebook kernel's inline backend into the isolated environment.
