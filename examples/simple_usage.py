from sae_feature_atlas import GemmaScopeBundle, make_config


def main() -> None:
    cfg = make_config(model="gemma-3-1b-pt", layer=13, max_texts=10, top_k=10)
    bundle = GemmaScopeBundle(cfg).load()

    print(bundle.validate())

    values, indices = bundle.topk_features("London is the capital of Great Britain", top_k=10)
    print("indices:", indices)
    print("values:", values)


if __name__ == "__main__":
    main()
