"""End-to-end tests of the sequential runner on a small linear problem (fast, CPU only)."""
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from bed.data import create_synthetic_data
from bed.ekf import EKF
from bed.experiments import Experiment
from bed.models import LinearNN, NeuralNetworkRegressor

jax.config.update("jax_enable_x64", True)

INPUT_DIM = 2
NUM_TRAIN = 30
NUM_POOL = 10
NOISE_VAR = 0.05


def _make_experiment(seed=0, warm_start=3):
    flax_model = LinearNN(input_dim=INPUT_DIM, rngs=nnx.Rngs(0))
    truth = deepcopy(flax_model)
    nnx.update(truth, truth.weights_to_state(
        jax.random.normal(jax.random.PRNGKey(1234), (flax_model.weight_size, 1))))
    data = create_synthetic_data(
        truth, num_train=NUM_TRAIN, num_test=NUM_POOL, input_dim=INPUT_DIM, embedding_dim=1,
        embedding_noise_std=0.1, measurement_noise_std=NOISE_VAR**0.5, key=jax.random.PRNGKey(0),
    )
    return Experiment(
        model=NeuralNetworkRegressor(flax_model), data=data,
        latent_cov=0.5 * jnp.eye(flax_model.weight_size), latent_innovation=0.0,
        measurement_error=NOISE_VAR * jnp.eye(1), warm_start=warm_start, seed=seed,
    )


def _run(seed=0, criteria=("EPIG", "EIG", "RANDOM"), iterations=6, filter_types=("ekf", "ukf")):
    exp = _make_experiment(seed=seed)
    return exp.run_experiment(criteria=list(criteria), filter_types=list(filter_types),
                              filter_params=[{}] * len(filter_types), iterations=iterations,
                              optimizer_method="brute_force")


def test_selection_is_without_replacement_and_excludes_the_warm_start():
    results = _run()
    for filter_type, by_criterion in results.experiment_results_dict.items():
        warm = by_criterion["WARM-START"]
        assert len(warm.selected_indices) == 3
        assert len(by_criterion["RANDOM"].selected_indices) == 6
        for criterion in ("EPIG", "EIG"):
            picked = by_criterion[criterion].selected_indices
            assert len(picked) == 6
            assert len(set(picked)) == len(picked), f"{filter_type}/{criterion} re-selected a design"
            assert not set(picked) & set(warm.selected_indices), f"{filter_type}/{criterion} re-selected a warm-start design"


def test_warm_start_is_shared_across_filters_and_runs_are_reproducible():
    first, second = _run(seed=3), _run(seed=3)
    d1, d2 = first.experiment_results_dict, second.experiment_results_dict
    assert d1["ekf"]["WARM-START"].selected_indices == d1["ukf"]["WARM-START"].selected_indices
    for filter_type in ("ekf", "ukf"):
        for criterion in ("EPIG", "EIG", "RANDOM"):
            assert d1[filter_type][criterion].selected_indices == d2[filter_type][criterion].selected_indices
    assert _run(seed=4).experiment_results_dict["ekf"]["WARM-START"].selected_indices != d1["ekf"]["WARM-START"].selected_indices


def test_ekf_and_ukf_select_the_same_designs_on_a_linear_model():
    # The sigma-point moments are exact for a model linear in the weights, so both
    # filters must score every candidate identically and follow the same trajectory.
    d = _run().experiment_results_dict
    for criterion in ("EPIG", "EIG"):
        assert d["ekf"][criterion].selected_indices == d["ukf"][criterion].selected_indices
        assert np.allclose(np.asarray(d["ekf"][criterion].rmse_pool, dtype=float),
                           np.asarray(d["ukf"][criterion].rmse_pool, dtype=float), rtol=1e-8, atol=1e-10)


def test_epig_reduces_pool_uncertainty():
    d = _run(iterations=8).experiment_results_dict["ekf"]
    sd = np.asarray(d["EPIG"].sd_pool, dtype=float)
    assert sd[-1] < sd[0]
    assert np.all(np.diff(sd) <= 1e-9)  # with Q = 0 the predictive variance cannot grow


def test_unknown_criterion_raises():
    exp = _make_experiment()
    filt = EKF(exp.model, exp.state_init_prior[0], exp.state_init_prior[1], 0.0, exp.measurement_error)
    with pytest.raises(ValueError):
        exp.run("RAND", "ekf", {}, filt, epochs=1, optimizer="brute_force")


def test_pool_exhaustion_raises():
    exp = _make_experiment(warm_start=0)
    filt = EKF(exp.model, exp.state_init_prior[0], exp.state_init_prior[1], 0.0, exp.measurement_error)
    with pytest.raises(ValueError):
        exp.run("EPIG", "ekf", {}, filt, epochs=NUM_TRAIN + 1, optimizer="brute_force")


def test_monte_carlo_epig_runs_and_tracks_the_closed_form():
    exp = _make_experiment()
    filt = EKF(exp.model, exp.state_init_prior[0], exp.state_init_prior[1], 0.0, exp.measurement_error)
    x = exp.design_space[:5]
    closed = np.asarray(exp.calculate_epig(x, filt), dtype=float)
    estimates = np.stack([np.asarray(exp.calculate_epig_mc(x, filt, num_latent_samples=2000), dtype=float)
                          for _ in range(20)])
    assert np.all(np.isfinite(estimates))
    mean, sem = estimates.mean(axis=0), estimates.std(axis=0, ddof=1) / np.sqrt(len(estimates))
    # nested Monte Carlo is biased at finite sample size; require agreement within the
    # estimator's own spread plus a small absolute allowance, not exact equality
    assert np.all(np.abs(mean - closed) < 3 * sem + 0.05), (mean, closed, sem)
    # and the estimator must rank the candidates like the closed form
    assert np.corrcoef(mean, closed)[0, 1] > 0.9


def test_ekf_and_ukf_agree_with_the_runner_criterion_values_on_a_linear_model():
    d = _run(iterations=4).experiment_results_dict
    for criterion in ("EPIG", "EIG"):
        assert np.allclose(np.asarray(d["ekf"][criterion].crit_values, dtype=float),
                           np.asarray(d["ukf"][criterion].crit_values, dtype=float), rtol=1e-8, atol=1e-10)
