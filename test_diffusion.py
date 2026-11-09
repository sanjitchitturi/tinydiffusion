import torch
import pytest
from types import SimpleNamespace

from diffusion import GaussianDiffusion
from model import TinyDiffusion
from sample import validate_checkpoint
from evaluate import mmd2_unbiased


def test_q_sample_matches_clean_input_at_zero_noise() -> None:
    diffusion = GaussianDiffusion(timesteps=8)
    clean = torch.tensor([[1.0, -2.0]])
    timestep = torch.tensor([0])
    noisy, sampled_noise = diffusion.q_sample(clean, timestep, torch.zeros_like(clean))

    torch.testing.assert_close(sampled_noise, torch.zeros_like(clean))
    torch.testing.assert_close(noisy, diffusion.sqrt_alpha_bar[0] * clean)


def test_denoiser_is_deterministic_in_evaluation_mode() -> None:
    model = TinyDiffusion(hidden_dim=16, depth=1, time_dim=8).eval()
    points = torch.randn(4, 2)
    timesteps = torch.tensor([0, 1, 2, 3])

    torch.testing.assert_close(model(points, timesteps), model(points, timesteps))


def test_reverse_sampling_has_requested_shape_and_finite_values() -> None:
    model = TinyDiffusion(hidden_dim=16, depth=1, time_dim=8).eval()
    samples = GaussianDiffusion(timesteps=4).sample(model, (6, 2))

    assert samples.shape == (6, 2)
    assert torch.isfinite(samples).all()


def test_checkpoint_rejects_mismatched_dataset() -> None:
    args = SimpleNamespace(dataset="moons", hidden_dim=16, depth=1, time_dim=8, timesteps=8)
    checkpoint = {"model_state": {}, "config": vars(args).copy()}
    validate_checkpoint(checkpoint, args)
    checkpoint["config"]["dataset"] = "circles"
    with pytest.raises(ValueError, match="dataset"):
        validate_checkpoint(checkpoint, args)


def test_unbiased_mmd_is_symmetric_and_detects_separation() -> None:
    import numpy as np

    x = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    y = x + 4.0
    assert mmd2_unbiased(x, y, 1.0) == pytest.approx(mmd2_unbiased(y, x, 1.0))
    assert mmd2_unbiased(x, y, 1.0) > mmd2_unbiased(x, x, 1.0)


def test_mmd_rejects_nonfinite_or_mismatched_samples() -> None:
    import numpy as np

    valid = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="finite 2D samples"):
        mmd2_unbiased(valid, np.array([[0.0], [1.0]]), 1.0)
    with pytest.raises(ValueError, match="finite 2D samples"):
        mmd2_unbiased(valid, np.array([[0.0, np.nan], [1.0, 1.0]]), 1.0)
    with pytest.raises(ValueError, match="positive bandwidth"):
        mmd2_unbiased(valid, valid, float("nan"))
