"""Configuration builders and bundled experiment configs."""
import os

import jax.numpy as jnp
import pytest
from flax import nnx

from bed.cli import main as climain
from bed.models import DenseNN

os.environ.setdefault("PYTHONBREAKPOINT", "0")


def _model():
    return DenseNN(input_dim=3, hidden_dims=[4], output_dim=1, rngs=nnx.Rngs(0))


def test_he_diagonal_prior_uses_kernel_scale_and_bias_var():
    model = _model()
    cov = climain._build_latent_cov(model, {"type": "he_diagonal", "kernel_scale": 6.0, "bias_var": 0.25})
    diag = jnp.diag(cov)
    for path, meta in model.weight_mapping.items():
        s0, s1 = meta["slice"]
        expected = 0.25 if path[-1] == "bias" else 6.0 / meta["shape"][0]
        assert jnp.allclose(diag[s0:s1], expected)


def test_latent_cov_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown keys"):
        climain._build_latent_cov(_model(), {"type": "he_diagonal", "scale": 2})
    with pytest.raises(ValueError, match="Unsupported"):
        climain._build_latent_cov(_model(), {"type": "banana"})


def test_scaled_identity_prior():
    model = _model()
    cov = climain._build_latent_cov(model, {"type": "scaled_identity", "scale": 0.3})
    assert jnp.allclose(cov, 0.3 * jnp.eye(model.weight_size))


@pytest.mark.parametrize("name", ["experiment_0", "experiment_1", "experiment_2", "experiment_5"])
def test_bundled_synthetic_configs_build(name):
    summary = climain.run_experiment_from_config(name, dry_run=True)
    assert summary["design_pool_size"] > 0
    assert "RAND" not in summary["strategies"]


def test_synthetic_config_falls_back_to_the_model_config_for_the_ground_truth():
    data_cfg = {"kind": "synthetic", "num_train": 4, "num_test": 2, "input_dim": 2, "embedding_dim": 1}
    model_cfg = {"type": "LinearNN", "input_dim": 2, "seed": 0}
    data = climain._build_data(data_cfg, model_cfg, {"enabled": True, "seed": 7, "scale": 2.0})
    assert data.x_train.shape == (4, 1, 1, 2)
    with pytest.raises(ValueError):
        climain._build_data(data_cfg)


def test_short_run_of_experiment_0_completes():
    results = climain.run_experiment_from_config("experiment_0", iterations=2, strategies=["EPIG", "RANDOM"], show_plot=False)
    r = results.experiment_results_dict["ekf"]["EPIG"]
    assert len(r.rmse_pool) == 2
    # experiment_0 optimizes over a continuous grid, so designs are points, not pool indices
    assert len(r.selected_designs) == 2 and r.selected_indices == []
    assert "WARM-START" in results.experiment_results_dict["ekf"]
