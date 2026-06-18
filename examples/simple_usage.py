from sae_feature_atlas import make_config
from sae_feature_atlas.runtime.gemma_scope import GemmaScopeRuntime


def main() -> None:
    cfg = make_config(model="gemma-3-1b-pt", layer=13, max_texts=10, top_k=10)
    runtime = GemmaScopeRuntime(cfg).load()

    print(runtime.validate())

    values, indices = runtime.topk_features("London is the capital of Great Britain", top_k=10)
    print("indices:", indices)
    print("values:", values)


if __name__ == "__main__":
    main()
