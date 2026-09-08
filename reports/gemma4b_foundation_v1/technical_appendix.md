# Technical appendix

Reproducibility details are separated from the scientific narrative.

## Layer 17

### collection_provenance.json

```json

{
  "collection_started_at_utc": "2026-09-07T22:05:26.574398+00:00",
  "model_revision": "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
  "sae_repo": "google/gemma-scope-2-4b-pt",
  "sae_revision": "a0ffd6132a985bc84077a66d1a1033e10b604fa8",
  "dataset": "prepared-jsonl",
  "dataset_revision": null,
  "prepared_corpus_sha256": "93907478e7d3b42170a6e65c893ce2090e817b764fd9f91333cee6d4f1de27a7",
  "collection": {
    "run_name": "gemma4b_foundation_v1_l17",
    "corpus": "foundation-two-source",
    "max_texts": 12000,
    "max_seq_len": 512,
    "activation_mode": "positive",
    "top_k_features_per_token": 64,
    "residual_sample_stride": 8,
    "random_seed": 42
  },
  "model": {
    "model_name": "google/gemma-3-4b-pt",
    "sae_release": "gemma-scope-2-4b-pt-res",
    "sae_id": "layer_17_width_65k_l0_medium",
    "layer": 17,
    "hook_name": "blocks.17.hook_resid_post",
    "site": "resid_post",
    "width": "65k",
    "l0": "medium",
    "d_model": null,
    "d_sae": null
  },
  "collector_sha256": "7cd866d02520ffc9d5683da56b0d728d2b03c5319ee1d7e592215aa5bb5fb380",
  "git": {
    "commit_sha": null,
    "dirty": null
  },
  "python": "3.11.14",
  "torch": "2.12.0+cu126",
  "cuda": "12.6",
  "gpu": "NVIDIA A100-SXM4-40GB",
  "packages": {
    "transformers": "5.12.0",
    "sae-lens": "6.44.2",
    "datasets": "5.0.0",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "scikit-learn": "1.9.0",
    "pandas": "3.0.3",
    "pyarrow": "24.0.0",
    "huggingface-hub": "1.19.0"
  },
  "collection_backend": "native HF text backbone; layers[17] output",
  "model_dtype": "bfloat16",
  "sae_dtype": "float32",
  "corpus_selection": "prepared manifest; fixed document ordering and splits",
  "sae_native_config": {
    "hf_hook_point_in": "model.layers.17.output",
    "hf_hook_point_out": "model.layers.17.output",
    "width": 65536,
    "model_name": "google/gemma-3-4b-pt",
    "architecture": "jump_relu",
    "l0": 60,
    "affine_connection": false,
    "type": "sae"
  },
  "sae_weights_sha256": "7cf01279c876ed7d67722a83fb89608d2306556a2aec02253a833387530b0923",
  "sae_config": "JumpReLUSAEConfig(d_in=2560, d_sae=65536, dtype='float32', device='cuda', apply_b_dec_to_input=False, normalize_activations='none', reshape_activations='none', metadata=SAEMetadata({'sae_lens_version': '6.44.2', 'sae_lens_training_version': None, 'model_name': 'google/gemma-3-4b-pt', 'hook_name': 'blocks.17.hook_resid_post', 'hook_head_index': None, 'prepend_bos': True, 'dataset_path': 'monology/pile-uncopyrighted', 'context_size': 1024, 'hf_hook_name': 'model.layers.17.output', 'neuronpedia_id': 'gemma-3-4b/17-gemmascope-2-res-65k'}))",
  "text_count": 12000,
  "source_texts_sha256": "93907478e7d3b42170a6e65c893ce2090e817b764fd9f91333cee6d4f1de27a7"
}

```

### foundation_design.json

```json

{
  "experiment": {
    "model": {
      "model_name": "google/gemma-3-4b-pt",
      "sae_release": "gemma-scope-2-4b-pt-res",
      "sae_id": "layer_17_width_65k_l0_medium",
      "layer": 17,
      "hook_name": "blocks.17.hook_resid_post",
      "site": "resid_post",
      "width": "65k",
      "l0": "medium",
      "d_model": null,
      "d_sae": null
    },
    "collection": {
      "run_name": "gemma4b_foundation_v1_l17",
      "corpus": "foundation-two-source",
      "max_texts": 12000,
      "max_seq_len": 512,
      "activation_mode": "positive",
      "top_k_features_per_token": 64,
      "residual_sample_stride": 8,
      "random_seed": 42
    },
    "activation_filter": {
      "exclude_token_positions": [
        0
      ],
      "exclude_token_positions_ge": null,
      "exclude_token_strings": [
        ""
      ],
      "exclude_token_substrings": [
        "\u00e2",
        "\u20ac",
        "\u2122"
      ],
      "include_sources": null,
      "exclude_sources": [],
      "min_activation": null,
      "require_finite_activation": true,
      "exclude_token_quality_kinds": [
        "space",
        "special",
        "quote",
        "punctuation",
        "symbol",
        "mojibake",
        "control"
      ],
      "keep_token_quality_columns": true
    },
    "feature_filter": {
      "min_feature_token_count": 50,
      "min_feature_text_count": 10,
      "max_feature_token_frequency": 0.2
    },
    "analysis": {
      "coactivation_max_pairs": 100000,
      "coactivation_min_pair_support": 10,
      "coactivation_neighbors_per_feature": 50,
      "decoder_neighbors_top_k": 20,
      "decoder_neighbors_batch_size": 512,
      "bimodality_min_points": 100,
      "bimodality_delta_bic_threshold": 10.0,
      "bimodality_min_component_weight": 0.1,
      "bimodality_min_separation": 2.0,
      "bimodality_gmm_n_init": 5,
      "bimodality_random_seed": 0,
      "bimodality_top_features_for_examples": 50,
      "bimodality_examples_per_peak": 8,
      "top_examples_per_feature": 20,
      "context_window": 20,
      "inspection_top_features": 30,
      "inspection_top_pairs": 30,
      "pca_components": 20,
      "pca_scatter_sample": 5000,
      "umap_neighbors": 30,
      "umap_min_dist": 0.1,
      "umap_metric": "cosine",
      "umap_random_state": 42,
      "pc_alignment_top_components": [
        1,
        5,
        20
      ],
      "graph_alignment_k_values": [
        5,
        10,
        20
      ]
    },
    "paths": {
      "raw_texts_path": "data/raw/texts_sample.jsonl",
      "data_root": "data/processed",
      "reports_root": "reports"
    }
  },
  "regimes": {
    "seed": 20260908,
    "discovery_fraction": 0.5,
    "screen_features": 2048,
    "min_discovery_support": 500,
    "min_discovery_documents": 100,
    "max_candidates": 24,
    "posterior_threshold": 0.9,
    "min_regime_support": 100,
    "min_regime_documents": 100,
    "min_partner_support": 10,
    "neighbor_k": 20,
    "permutations": 9999,
    "bootstraps": 1000,
    "position_bin": 64,
    "support_bin": 16,
    "match_log_caliper": 0.35,
    "gmm_n_init": 5,
    "delta_bic_threshold": 10.0,
    "min_component_weight": 0.1,
    "min_separation": 2.0
  },
  "corpus_manifest": {
    "seed": 20260908,
    "per_source": 6000,
    "shuffle_buffer": 10000,
    "sampling": "seeded finite-buffer streaming shuffle; not full-corpus uniform",
    "deduplication": "global exact NFKC/whitespace-normalized SHA256",
    "near_duplicate_audit": "completed; approximate candidate retrieval",
    "sources": [
      {
        "name": "fineweb-edu-sample",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "count": 6000,
        "rejected": {
          "short_or_invalid": 2,
          "duplicate": 0
        }
      },
      {
        "name": "wikimedia-en",
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.en",
        "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "count": 6000,
        "rejected": {
          "short_or_invalid": 765,
          "duplicate": 0
        }
      }
    ],
    "texts_sha256": "93907478e7d3b42170a6e65c893ce2090e817b764fd9f91333cee6d4f1de27a7",
    "duplicate_audit_sha256": "bd81e8f7433f2a1c4729e37617e95abb72f585ee295b91fe96eef3e46830dffb",
    "split_method": "seeded group hash; approximately half per source",
    "tokenizer_model": "google/gemma-3-4b-pt",
    "tokenizer_revision": "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
    "max_length": 512
  },
  "protocol_sha256": "f31c22c139511809775dc90b4ec7afbb7be52798edfda1375d081dd2e3068b35"
}

```

### geometry/plan.json

```json

{
  "layer": 17,
  "seed": 20260908,
  "primary_ids": [
    72,
    296,
    575,
    665,
    710,
    802,
    863,
    1076,
    1635,
    1645,
    1844,
    1892,
    1903,
    2286,
    2377,
    3136,
    6756,
    7511,
    8532,
    10392,
    12335,
    17857,
    28027,
    35053
  ],
  "quantile_ids": [
    175,
    390,
    932,
    1734,
    2133,
    2264,
    2482,
    2581,
    3096,
    3186,
    3610,
    3781,
    4926,
    5054,
    5105,
    5868,
    6277,
    6339,
    6476,
    6780,
    7986,
    8076,
    8963,
    9222,
    9536,
    9672,
    9844,
    9985,
    10012,
    10223,
    11245,
    11472,
    13290,
    13617,
    15592,
    16634,
    18791,
    19452,
    19677,
    20053,
    20166,
    20532,
    20644,
    20647,
    21268,
    21367,
    21836,
    24107,
    24120,
    24180,
    24553,
    28448,
    30162,
    31578,
    32626,
    36728,
    40051,
    42476,
    48511,
    48879,
    51351,
    55280,
    59930,
    62248
  ],
  "reference_documents_per_source_split": 300,
  "targets_sha256": "3f7a54c9d0f9249df6e507e5f4630cc954bf24575d78fb1a29ec0f8e0583372d",
  "discovery_fits_sha256": "a101c09154989329dbd5426f7e3a0f25ce707c5304e7f36fdfa102b30968df96",
  "geometry_protocol_sha256": "0e788fab60ea4707198e57265fefa900227af8ff00927bab4fc5f78b732273c1",
  "planner_sha256": "8ccac171b55f8faf97d5a9ab67b77ebcf3b7d0d47fcf6ce2c38b826be1f2f508",
  "native_token_count": 214268
}

```

### geometry/analysis_complete.json

```json

{
  "native_locations": 214268,
  "discovery_reference_locations": 12158,
  "dimension": 2560,
  "analysis_sha256": "42d536d029608a922e4672b2a4e5fcfba0775eed462157c368c5755fd6e73cda",
  "plan_sha256": "6b392792e9e3e2afbad95f4119180ffdbd68204c626131d506b2b4801e7194c8"
}

```

### geometry/coverage_complete.json

```json

{
  "screened_features": 2048,
  "screened_pairs": 2096128,
  "reference_evaluation_locations": 12141,
  "centered_reconstruction_r_squared": 0.9367885691458877,
  "uncentered_relative_mse": 0.002199634040413358,
  "pair_population": "All eligible collected tokens; descriptive, not a held-out hypothesis test; restricted to discovery-supported random screen"
}

```

### geometry/encoder_control_design.json

```json

{
  "status": "Exploratory sensitivity added after primary and orthogonal outcomes were inspected",
  "direction": "Remove span of focal decoder and full discovery residual covariance times focal encoder",
  "covariance": "Same frozen discovery reference PCA covariance",
  "permutations": 4999,
  "seed_offset": 314,
  "family": "BY across all selected candidates per layer; source conjunction",
  "code_sha256": "1585dfc3eb79cc23fe19163a2217c95e6d200932d7a249298b5c6ef721e9a5c9",
  "interpretation": "Simple linear covariance alternative, not a complete conditional generative null"
}

```

### geometry/numerical_tie_audit.json

```json

{
  "tolerance": "1e-10 * max(1, abs(observed squared norm))",
  "changes": [
    {
      "feature": 1076,
      "source": "fineweb-edu-sample",
      "old_p": 0.0002,
      "tie_aware_p": 0.0568
    },
    {
      "feature": 12335,
      "source": "fineweb-edu-sample",
      "old_p": 0.4648,
      "tie_aware_p": 0.4678
    }
  ],
  "null_draws_unchanged": true
}

```

### geometry/replay_activation_audit.json

```json

{
  "sampled_locations": 32,
  "seed": 20261025,
  "max_absolute_error": 0.00927734375,
  "maximum_expected_activation": 5934.224609375,
  "positive_membership_disagreements": 0,
  "absolute_tolerance": 0.03,
  "relative_tolerance": 0.0001,
  "passed": true,
  "purpose": "Replay hook and coordinate consistency across all dictionary entries at seeded locations"
}

```

### geometry/consistency_complete.json

```json

{
  "layer": 17,
  "source_sha256": "f46e138973e5b32c9de4d7c4982520b3f1972f57436581c294a74b747317b947",
  "decoder_pca_variance_2d": 0.06489324701964179,
  "status": "Prespecified descriptive checks; no significance test",
  "binary_partner_sum": "Sum decoder vectors weighted by signed changes in positive membership probability; not an amplitude-weighted SAE reconstruction"
}

```

### secondary_complete.json

```json

{
  "layer": 17,
  "verified_complete_at_utc": "2026-09-08T00:10:22.141790+00:00",
  "pooled_candidates": 24,
  "weak_matches": 14,
  "weak_evaluated": 14,
  "weak_supported": 4,
  "completion_check": "All selected pooled candidates and all eligible matched controls have final outputs."
}

```

### regime_weak_control_design.json

```json

{
  "pool_size": 2048,
  "additional_features_fitted": 2048,
  "seed": 20260909,
  "definition": "converged GMM, standardized separation <2",
  "amendment": "Prespecified in the foundation protocol before this run's evaluation.",
  "matching": "original log-support/log-document caliper, no replacement",
  "evaluation": "same frozen discovery tail fractions; exact candidate regime counts",
  "strict_control_arm_preserved": true
}

```

### pooled_execution.json

```json

{
  "source_sha256": "41990376e26949a38b9cd77a526954e7cbbce5544ed7c1172aef22e1b26d5862",
  "method": "Frozen fits and completed checkpoints retained; grouped conditional permutations for remaining pooled comparisons"
}

```

### source_confirmation/execution.json

```json

{
  "method": "Grouped uniform within-stratum permutations; same frozen endpoint and 9999 replicates",
  "source_sha256": "f83cbca5cc3f2b4865cdb3f952cac995e1f352c970ca46b92a93c8b704eea9e3",
  "plan_sha256": "6b392792e9e3e2afbad95f4119180ffdbd68204c626131d506b2b4801e7194c8"
}

```

regime_matched_controls.parquet: 0 matched candidates of 24.

regime_weak_matched_controls.parquet: 14 matched candidates of 24.

## Layer 22

### collection_provenance.json

```json

{
  "collection_started_at_utc": "2026-09-07T22:21:30.924091+00:00",
  "model_revision": "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
  "sae_repo": "google/gemma-scope-2-4b-pt",
  "sae_revision": "a0ffd6132a985bc84077a66d1a1033e10b604fa8",
  "dataset": "prepared-jsonl",
  "dataset_revision": null,
  "prepared_corpus_sha256": "93907478e7d3b42170a6e65c893ce2090e817b764fd9f91333cee6d4f1de27a7",
  "collection": {
    "run_name": "gemma4b_foundation_v1_l22",
    "corpus": "foundation-two-source",
    "max_texts": 12000,
    "max_seq_len": 512,
    "activation_mode": "positive",
    "top_k_features_per_token": 64,
    "residual_sample_stride": 8,
    "random_seed": 42
  },
  "model": {
    "model_name": "google/gemma-3-4b-pt",
    "sae_release": "gemma-scope-2-4b-pt-res",
    "sae_id": "layer_22_width_65k_l0_medium",
    "layer": 22,
    "hook_name": "blocks.22.hook_resid_post",
    "site": "resid_post",
    "width": "65k",
    "l0": "medium",
    "d_model": null,
    "d_sae": null
  },
  "collector_sha256": "7cd866d02520ffc9d5683da56b0d728d2b03c5319ee1d7e592215aa5bb5fb380",
  "git": {
    "commit_sha": null,
    "dirty": null
  },
  "python": "3.11.14",
  "torch": "2.12.0+cu126",
  "cuda": "12.6",
  "gpu": "NVIDIA A100-SXM4-40GB",
  "packages": {
    "transformers": "5.12.0",
    "sae-lens": "6.44.2",
    "datasets": "5.0.0",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "scikit-learn": "1.9.0",
    "pandas": "3.0.3",
    "pyarrow": "24.0.0",
    "huggingface-hub": "1.19.0"
  },
  "collection_backend": "native HF text backbone; layers[22] output",
  "model_dtype": "bfloat16",
  "sae_dtype": "float32",
  "corpus_selection": "prepared manifest; fixed document ordering and splits",
  "sae_native_config": {
    "hf_hook_point_in": "model.layers.22.output",
    "hf_hook_point_out": "model.layers.22.output",
    "width": 65536,
    "model_name": "google/gemma-3-4b-pt",
    "architecture": "jump_relu",
    "l0": 60,
    "affine_connection": false,
    "type": "sae"
  },
  "sae_weights_sha256": "1fd68bc98bf876ec2fb7a3233b424e821351533b0121872e8029219408a0d703",
  "sae_config": "JumpReLUSAEConfig(d_in=2560, d_sae=65536, dtype='float32', device='cuda', apply_b_dec_to_input=False, normalize_activations='none', reshape_activations='none', metadata=SAEMetadata({'sae_lens_version': '6.44.2', 'sae_lens_training_version': None, 'model_name': 'google/gemma-3-4b-pt', 'hook_name': 'blocks.22.hook_resid_post', 'hook_head_index': None, 'prepend_bos': True, 'dataset_path': 'monology/pile-uncopyrighted', 'context_size': 1024, 'hf_hook_name': 'model.layers.22.output', 'neuronpedia_id': 'gemma-3-4b/22-gemmascope-2-res-65k'}))",
  "text_count": 12000,
  "source_texts_sha256": "93907478e7d3b42170a6e65c893ce2090e817b764fd9f91333cee6d4f1de27a7"
}

```

### foundation_design.json

```json

{
  "experiment": {
    "model": {
      "model_name": "google/gemma-3-4b-pt",
      "sae_release": "gemma-scope-2-4b-pt-res",
      "sae_id": "layer_22_width_65k_l0_medium",
      "layer": 22,
      "hook_name": "blocks.22.hook_resid_post",
      "site": "resid_post",
      "width": "65k",
      "l0": "medium",
      "d_model": null,
      "d_sae": null
    },
    "collection": {
      "run_name": "gemma4b_foundation_v1_l22",
      "corpus": "foundation-two-source",
      "max_texts": 12000,
      "max_seq_len": 512,
      "activation_mode": "positive",
      "top_k_features_per_token": 64,
      "residual_sample_stride": 8,
      "random_seed": 42
    },
    "activation_filter": {
      "exclude_token_positions": [
        0
      ],
      "exclude_token_positions_ge": null,
      "exclude_token_strings": [
        ""
      ],
      "exclude_token_substrings": [
        "\u00e2",
        "\u20ac",
        "\u2122"
      ],
      "include_sources": null,
      "exclude_sources": [],
      "min_activation": null,
      "require_finite_activation": true,
      "exclude_token_quality_kinds": [
        "space",
        "special",
        "quote",
        "punctuation",
        "symbol",
        "mojibake",
        "control"
      ],
      "keep_token_quality_columns": true
    },
    "feature_filter": {
      "min_feature_token_count": 50,
      "min_feature_text_count": 10,
      "max_feature_token_frequency": 0.2
    },
    "analysis": {
      "coactivation_max_pairs": 100000,
      "coactivation_min_pair_support": 10,
      "coactivation_neighbors_per_feature": 50,
      "decoder_neighbors_top_k": 20,
      "decoder_neighbors_batch_size": 512,
      "bimodality_min_points": 100,
      "bimodality_delta_bic_threshold": 10.0,
      "bimodality_min_component_weight": 0.1,
      "bimodality_min_separation": 2.0,
      "bimodality_gmm_n_init": 5,
      "bimodality_random_seed": 0,
      "bimodality_top_features_for_examples": 50,
      "bimodality_examples_per_peak": 8,
      "top_examples_per_feature": 20,
      "context_window": 20,
      "inspection_top_features": 30,
      "inspection_top_pairs": 30,
      "pca_components": 20,
      "pca_scatter_sample": 5000,
      "umap_neighbors": 30,
      "umap_min_dist": 0.1,
      "umap_metric": "cosine",
      "umap_random_state": 42,
      "pc_alignment_top_components": [
        1,
        5,
        20
      ],
      "graph_alignment_k_values": [
        5,
        10,
        20
      ]
    },
    "paths": {
      "raw_texts_path": "data/raw/texts_sample.jsonl",
      "data_root": "data/processed",
      "reports_root": "reports"
    }
  },
  "regimes": {
    "seed": 20260908,
    "discovery_fraction": 0.5,
    "screen_features": 2048,
    "min_discovery_support": 500,
    "min_discovery_documents": 100,
    "max_candidates": 24,
    "posterior_threshold": 0.9,
    "min_regime_support": 100,
    "min_regime_documents": 100,
    "min_partner_support": 10,
    "neighbor_k": 20,
    "permutations": 9999,
    "bootstraps": 1000,
    "position_bin": 64,
    "support_bin": 16,
    "match_log_caliper": 0.35,
    "gmm_n_init": 5,
    "delta_bic_threshold": 10.0,
    "min_component_weight": 0.1,
    "min_separation": 2.0
  },
  "corpus_manifest": {
    "seed": 20260908,
    "per_source": 6000,
    "shuffle_buffer": 10000,
    "sampling": "seeded finite-buffer streaming shuffle; not full-corpus uniform",
    "deduplication": "global exact NFKC/whitespace-normalized SHA256",
    "near_duplicate_audit": "completed; approximate candidate retrieval",
    "sources": [
      {
        "name": "fineweb-edu-sample",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "count": 6000,
        "rejected": {
          "short_or_invalid": 2,
          "duplicate": 0
        }
      },
      {
        "name": "wikimedia-en",
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.en",
        "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "count": 6000,
        "rejected": {
          "short_or_invalid": 765,
          "duplicate": 0
        }
      }
    ],
    "texts_sha256": "93907478e7d3b42170a6e65c893ce2090e817b764fd9f91333cee6d4f1de27a7",
    "duplicate_audit_sha256": "bd81e8f7433f2a1c4729e37617e95abb72f585ee295b91fe96eef3e46830dffb",
    "split_method": "seeded group hash; approximately half per source",
    "tokenizer_model": "google/gemma-3-4b-pt",
    "tokenizer_revision": "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
    "max_length": 512
  },
  "protocol_sha256": "f31c22c139511809775dc90b4ec7afbb7be52798edfda1375d081dd2e3068b35"
}

```

### geometry/plan.json

```json

{
  "layer": 22,
  "seed": 20260908,
  "primary_ids": [
    87,
    129,
    141,
    157,
    252,
    839,
    905,
    1199,
    1247,
    1284,
    1772,
    1827,
    2193,
    2241,
    4714,
    6629,
    6846,
    6903,
    7122,
    11908,
    14064,
    21515,
    25249,
    34492
  ],
  "quantile_ids": [
    261,
    583,
    1284,
    2366,
    3007,
    3199,
    3510,
    3753,
    4603,
    4835,
    5634,
    5853,
    7404,
    7462,
    7546,
    8662,
    9320,
    9415,
    9563,
    10043,
    11557,
    11599,
    12922,
    13298,
    13509,
    13732,
    14041,
    14162,
    14197,
    14621,
    15487,
    15599,
    17676,
    18036,
    20343,
    21409,
    23585,
    24046,
    24525,
    25249,
    25320,
    25739,
    25884,
    25909,
    26573,
    26644,
    27090,
    29318,
    29380,
    29407,
    29576,
    33647,
    34762,
    36293,
    37204,
    41050,
    44503,
    46550,
    51464,
    51744,
    53948,
    57594,
    62572,
    64090
  ],
  "reference_documents_per_source_split": 300,
  "targets_sha256": "9ee11f222a945a15d4068854fda57b7171c170622fef68ff9516d282ace4cef0",
  "discovery_fits_sha256": "2e7e8888b5035bf3dd35836abfa77de7ed60bb92f447ffaccd85adf7a36e5d9c",
  "geometry_protocol_sha256": "0e788fab60ea4707198e57265fefa900227af8ff00927bab4fc5f78b732273c1",
  "planner_sha256": "8ccac171b55f8faf97d5a9ab67b77ebcf3b7d0d47fcf6ce2c38b826be1f2f508",
  "native_token_count": 196158
}

```

### geometry/analysis_complete.json

```json

{
  "native_locations": 196158,
  "discovery_reference_locations": 12158,
  "dimension": 2560,
  "analysis_sha256": "42d536d029608a922e4672b2a4e5fcfba0775eed462157c368c5755fd6e73cda",
  "plan_sha256": "8a0ef9881532d75f5d9af16311af055c93e636c7cadfe61c69db8a19b1b7bf91"
}

```

### geometry/coverage_complete.json

```json

{
  "screened_features": 2048,
  "screened_pairs": 2096128,
  "reference_evaluation_locations": 12141,
  "centered_reconstruction_r_squared": 0.8673451317352625,
  "uncentered_relative_mse": 0.004978185621906421,
  "pair_population": "All eligible collected tokens; descriptive, not a held-out hypothesis test; restricted to discovery-supported random screen"
}

```

### geometry/encoder_control_design.json

```json

{
  "status": "Exploratory sensitivity added after primary and orthogonal outcomes were inspected",
  "direction": "Remove span of focal decoder and full discovery residual covariance times focal encoder",
  "covariance": "Same frozen discovery reference PCA covariance",
  "permutations": 4999,
  "seed_offset": 314,
  "family": "BY across all selected candidates per layer; source conjunction",
  "code_sha256": "1585dfc3eb79cc23fe19163a2217c95e6d200932d7a249298b5c6ef721e9a5c9",
  "interpretation": "Simple linear covariance alternative, not a complete conditional generative null"
}

```

### geometry/numerical_tie_audit.json

```json

{
  "tolerance": "1e-10 * max(1, abs(observed squared norm))",
  "changes": [],
  "null_draws_unchanged": true
}

```

### geometry/replay_activation_audit.json

```json

{
  "sampled_locations": 32,
  "seed": 20261025,
  "max_absolute_error": 0.01416015625,
  "maximum_expected_activation": 5092.755859375,
  "positive_membership_disagreements": 0,
  "absolute_tolerance": 0.03,
  "relative_tolerance": 0.0001,
  "passed": true,
  "purpose": "Replay hook and coordinate consistency across all dictionary entries at seeded locations"
}

```

### geometry/consistency_complete.json

```json

{
  "layer": 22,
  "source_sha256": "f46e138973e5b32c9de4d7c4982520b3f1972f57436581c294a74b747317b947",
  "decoder_pca_variance_2d": 0.03603104012471422,
  "status": "Prespecified descriptive checks; no significance test",
  "binary_partner_sum": "Sum decoder vectors weighted by signed changes in positive membership probability; not an amplitude-weighted SAE reconstruction"
}

```

### secondary_complete.json

```json

{
  "layer": 22,
  "verified_complete_at_utc": "2026-09-08T00:10:22.161662+00:00",
  "pooled_candidates": 24,
  "weak_matches": 15,
  "weak_evaluated": 15,
  "weak_supported": 4,
  "completion_check": "All selected pooled candidates and all eligible matched controls have final outputs."
}

```

### regime_weak_control_design.json

```json

{
  "pool_size": 2048,
  "additional_features_fitted": 2048,
  "seed": 20260909,
  "definition": "converged GMM, standardized separation <2",
  "amendment": "Prespecified in the foundation protocol before this run's evaluation.",
  "matching": "original log-support/log-document caliper, no replacement",
  "evaluation": "same frozen discovery tail fractions; exact candidate regime counts",
  "strict_control_arm_preserved": true
}

```

### pooled_execution.json

```json

{
  "source_sha256": "41990376e26949a38b9cd77a526954e7cbbce5544ed7c1172aef22e1b26d5862",
  "method": "Frozen fits and completed checkpoints retained; grouped conditional permutations for remaining pooled comparisons"
}

```

### source_confirmation/execution.json

```json

{
  "method": "Grouped uniform within-stratum permutations; same frozen endpoint and 9999 replicates",
  "source_sha256": "f83cbca5cc3f2b4865cdb3f952cac995e1f352c970ca46b92a93c8b704eea9e3",
  "plan_sha256": "8a0ef9881532d75f5d9af16311af055c93e636c7cadfe61c69db8a19b1b7bf91"
}

```

regime_matched_controls.parquet: 0 matched candidates of 24.

regime_weak_matched_controls.parquet: 15 matched candidates of 24.