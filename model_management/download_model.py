import argparse

from model_management.model_zoo import (
    ensure_local_model_artifact,
    get_model_artifact_path,
    list_available_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download model weights into model_management/models"
    )
    parser.add_argument(
        "model_name", choices=list_available_models(), help="model name to download"
    )
    args = parser.parse_args()

    artifact_path = ensure_local_model_artifact(args.model_name)
    print(f"Downloaded {args.model_name} to {artifact_path}")
    print(f"Expected local path: {get_model_artifact_path(args.model_name)}")


if __name__ == "__main__":
    main()
