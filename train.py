import argparse
import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import TinyDiffusion
from diffusion import GaussianDiffusion
from datasets import get_dataset


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    data = get_dataset(args.dataset, args.n_samples).to(device)

    model = TinyDiffusion(
        data_dim=2,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        time_dim=args.time_dim,
    ).to(device)

    diff = GaussianDiffusion(timesteps=args.timesteps, device=device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        idx = torch.randint(0, len(data), (args.batch_size,))
        x0 = data[idx]
        loss = diff.loss(model, x0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if epoch % 100 == 0 or epoch == 1:
            print(f"epoch {epoch:5d} | loss {loss.item():.6f}")

    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "config": {
                "dataset": args.dataset,
                "n_samples": args.n_samples,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "hidden_dim": args.hidden_dim,
                "depth": args.depth,
                "time_dim": args.time_dim,
                "timesteps": args.timesteps,
                "seed": args.seed,
            },
        },
        args.save_path,
    )
    print(f"model saved to {args.save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="swiss_roll", choices=["swiss_roll", "moons", "circles", "checkerboard"])
    p.add_argument("--n_samples", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--time_dim", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save_path", default="model.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)
