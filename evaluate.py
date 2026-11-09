"""Repeatable synthetic distribution check using unbiased RBF MMD squared."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import get_dataset
from diffusion import GaussianDiffusion
from model import TinyDiffusion


def squared_distances(x, y):
    return np.maximum(
        np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :]
        - 2 * x @ y.T,
        0.0,
    )


def mmd2_unbiased(x, y, bandwidth2):
    """An unbiased estimate; finite-sample values may be negative."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if (
        x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]
        or len(x) < 2 or len(y) < 2 or not np.isfinite(bandwidth2)
        or bandwidth2 <= 0 or not np.isfinite(x).all() or not np.isfinite(y).all()
    ):
        raise ValueError("MMD requires finite 2D samples with matching dimensions and positive bandwidth")
    xx = np.exp(-squared_distances(x, x) / (2 * bandwidth2))
    yy = np.exp(-squared_distances(y, y) / (2 * bandwidth2))
    xy = np.exp(-squared_distances(x, y) / (2 * bandwidth2))
    return float(
        (xx.sum() - np.trace(xx)) / (len(x) * (len(x) - 1))
        + (yy.sum() - np.trace(yy)) / (len(y) * (len(y) - 1))
        - 2 * xy.mean()
    )


def evaluate(checkpoint_path, output_dir, n_samples=512, repeats=3, seed=123):
    if n_samples < 2 or n_samples > 2000 or repeats < 1:
        raise ValueError("Use 2 <= n_samples <= 2000 and repeats >= 1")
    digest = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Expected a version 1 training checkpoint")
    config = checkpoint["config"]
    torch.manual_seed(seed)
    model = TinyDiffusion(
        data_dim=2, hidden_dim=config["hidden_dim"],
        depth=config["depth"], time_dim=config["time_dim"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    diffusion = GaussianDiffusion(timesteps=config["timesteps"], device="cpu")

    np.random.seed(seed)
    reference = get_dataset(config["dataset"], n_samples).numpy().astype(np.float64)
    distances = squared_distances(reference, reference)
    bandwidth2 = float(np.median(distances[np.triu_indices(n_samples, 1)]))
    if bandwidth2 <= 0:
        raise ValueError("Reference distribution has zero median pairwise distance")

    rows = []
    for repeat in range(repeats):
        np.random.seed(seed + repeat + 1)
        torch.manual_seed(seed + repeat + 1)
        heldout = get_dataset(config["dataset"], n_samples).numpy().astype(np.float64)
        independent = get_dataset(config["dataset"], n_samples).numpy().astype(np.float64)
        gaussian = np.random.multivariate_normal(
            reference.mean(axis=0), np.cov(reference, rowvar=False), n_samples,
        )
        generated = diffusion.sample(model, (n_samples, 2)).numpy().astype(np.float64)
        rows.append({
            "repeat": repeat,
            "generated": mmd2_unbiased(heldout, generated, bandwidth2),
            "gaussian": mmd2_unbiased(heldout, gaussian, bandwidth2),
            "independent_reference": mmd2_unbiased(heldout, independent, bandwidth2),
        })

    summary = {
        key: {"mean": float(np.mean([row[key] for row in rows])),
              "std": float(np.std([row[key] for row in rows], ddof=1)) if repeats > 1 else None}
        for key in ("generated", "gaussian", "independent_reference")
    }
    report = {
        "metric": "unbiased_rbf_mmd_squared",
        "lower_is_better": True,
        "bandwidth_squared": bandwidth2,
        "n_samples_per_set": n_samples,
        "repeats": repeats,
        "evaluation_seed": seed,
        "checkpoint_config": config,
        "checkpoint_sha256": digest,
        "per_repeat": rows,
        "summary": summary,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    names = ("generated", "gaussian", "independent_reference")
    ax.boxplot([[row[name] for row in rows] for name in names], tick_labels=names)
    ax.set_ylabel("Unbiased RBF MMD² (lower is better)")
    ax.set_title(f"{config['dataset']} · {repeats} evaluation repeats")
    fig.tight_layout()
    fig.savefig(destination / "mmd.png", dpi=150)
    plt.close(fig)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--output_dir", default="evaluation")
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.output_dir, args.n_samples, args.repeats, args.seed)
    print(json.dumps(result["summary"], indent=2))
