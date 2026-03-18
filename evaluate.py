import argparse
from utils import set_seed
from trainer import evaluate_reconstruction


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate reconstruction model")

    parser.add_argument("--backbone", type=str, default="AE", choices=["AE", "PTv3"])
    parser.add_argument("--dataset", type=str, default="modelnet40")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to evaluate on",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["cd", "geocd"],
        choices=["cd", "geocd"],
        help="Metrics to compute",
    )

    parser.add_argument("--k_value", type=int, default=5)
    parser.add_argument("--nhops", type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    config = get_args()
    set_seed(config.seed)
    evaluate_reconstruction(config)