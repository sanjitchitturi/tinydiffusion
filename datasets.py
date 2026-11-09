import torch
import numpy as np
from sklearn.datasets import make_swiss_roll, make_moons, make_circles

def get_dataset(name, n_samples):
    if name == "swiss_roll":
        X, _ = make_swiss_roll(n_samples, noise=0.1)
        X = X[:, [0, 2]]
    elif name == "moons":
        X, _ = make_moons(n_samples, noise=0.05)
    elif name == "circles":
        X, _ = make_circles(n_samples, noise=0.03, factor=0.5)
    elif name == "checkerboard":
        X = _checkerboard(n_samples)
    else:
        raise ValueError(f"unknown dataset: {name}")

    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return torch.tensor(X, dtype=torch.float32)

def _checkerboard(n):
    x = np.random.uniform(-4, 4, size=(n * 4, 2))
    mask = ((np.floor(x[:, 0]) + np.floor(x[:, 1])) % 2 == 0)
    x = x[mask][:n]
    return x
