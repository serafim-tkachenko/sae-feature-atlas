"""Seeded context excerpts and a separate key for subsequent blinded annotation."""

import argparse
from pathlib import Path
import pandas as pd


def main(layer):
    root = Path(f"data/processed/gemma4b_foundation_v1_l{layer}")
    targets = pd.read_parquet(root / "geometry/targets.parquet")
    targets = targets[targets.purpose == "primary"]
    tokens = pd.read_parquet(root / "token_metadata.parquet")
    grouped = {tid: f.set_index("token_pos").token_str for tid, f in tokens.groupby("text_id")}
    examples = []
    for (fid, source, level), frame in targets.groupby(["feature_id", "source", "level"]):
        for r in frame.sample(n=min(8, len(frame)), random_state=20261021).itertuples():
            words = grouped[r.text_id]
            before = "".join(words.loc[max(0, r.token_pos - 10) : r.token_pos - 1].tolist())
            after = "".join(words.loc[r.token_pos + 1 : r.token_pos + 10].tolist())
            examples.append(
                {
                    "feature_id": fid,
                    "source": source,
                    "regime": int(level),
                    "text_id": r.text_id,
                    "token_pos": r.token_pos,
                    "activation": r.activation,
                    "context": before + " [" + words.loc[r.token_pos] + "] " + after,
                }
            )
    result = pd.DataFrame(examples).sample(frac=1, random_state=20261023).reset_index(drop=True)
    result["example_id"] = [f"L{layer}-{i + 1:04d}" for i in range(len(result))]
    result.to_csv(root / "geometry/context_examples_key.csv", index=False)
    blind = result[["example_id", "context"]].assign(
        description="", category="", confidence="", reviewer_notes=""
    )
    blind.to_csv(root / "geometry/context_examples_blinded.csv", index=False)
    print("BLINDED CONTEXTS", layer, len(blind), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, choices=[17, 22], required=True)
    main(parser.parse_args().layer)
