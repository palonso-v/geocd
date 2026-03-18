import argparse
from utils import set_seed
from trainer import train_cd

def get_args():
    parser = argparse.ArgumentParser(description="Pretrain reconstruction model with Chamfer Distance")

    parser.add_argument("--backbone", type=str, default="AE", choices=["AE", "PTv3"])
    parser.add_argument("--dataset", type=str, default="modelnet40")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--nepochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=None, help="If None, uses backbone-specific default")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./outputs")

    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    config = get_args()
    set_seed(config.seed)
    train_cd(config)