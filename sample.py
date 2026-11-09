import argparse
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import TinyDiffusion
from diffusion import GaussianDiffusion
from datasets import get_dataset


def validate_checkpoint(checkpoint, args):
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Checkpoint is missing reproducibility metadata. Retrain with the current train.py.")
    config = checkpoint.get("config", {})
    expected = {
        "dataset": args.dataset,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "time_dim": args.time_dim,
        "timesteps": args.timesteps,
    }
    mismatches = [name for name, value in expected.items() if config.get(name) != value]
    if mismatches:
        raise ValueError(f"Checkpoint configuration differs for: {', '.join(mismatches)}")


def sample(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyDiffusion(
        data_dim=2,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        time_dim=args.time_dim,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    validate_checkpoint(checkpoint, args)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    diff = GaussianDiffusion(timesteps=args.timesteps, device=device)
    samples = diff.sample(model, shape=(args.n_samples, 2)).cpu().numpy()

    real = get_dataset(args.dataset, args.n_samples).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(real[:, 0], real[:, 1], s=2, alpha=0.4, c="#4f86f7")
    axes[0].set_title("real data")
    axes[0].axis("equal")
    axes[1].scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.4, c="#f76b4f")
    axes[1].set_title("generated samples")
    axes[1].axis("equal")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="model.pt")
    p.add_argument("--dataset", default="swiss_roll", choices=["swiss_roll", "moons", "circles", "checkerboard"])
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--time_dim", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--output", default="samples.png")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    sample(args)
