# Contributing

Keep TinyDiffusion focused on inspectable diffusion diagnostics rather than production
image generation.

## Verification

Run from the repository root:

```sh
python -m pytest -q
python train.py --dataset moons --epochs 2 --n_samples 64 --batch_size 16 --hidden_dim 16 --depth 1 --time_dim 8 --timesteps 8
```

Record seeds, data distribution, schedule, architecture, and runtime before comparing
samples. Do not claim sample quality from a short smoke run. Keep checkpoints, generated
plots, caches, and local environments out of commits.

Use [SECURITY.md](SECURITY.md) for security-sensitive reports.
