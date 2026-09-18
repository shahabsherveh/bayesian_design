import os
import tempfile
import numpy as np
import jax.numpy as jnp
import pytest
from bed.data import (
    Data,
    create_synthetic_data,
    create_synthetic_normal_with_outliers_data_1D,
    get_mnist_data,
    get_uci_data,
)


def simple_model(x):
    # x shape: (input_dim, N)
    return jnp.sum(x, axis=[1, 2, 3], keepdims=True).T  # shape: (N, 1)


def test_data_init_and_observe():
    x_train = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y_train = jnp.array([[10.0], [20.0]])
    x_test = jnp.array([[5.0, 6.0]])
    y_test = jnp.array([[30.0]])
    data = Data(x_train, y_train, x_test, y_test)
    # Check attributes
    assert jnp.allclose(data.x_train, x_train)
    assert jnp.allclose(data.y_train, y_train)
    assert jnp.allclose(data.x_test, x_test)
    assert jnp.allclose(data.y_test, y_test)
    # Test observe (should match y_train or y_test)
    obs = data.observe(jnp.array([1.0, 2.0]))
    assert np.isclose(obs, 10.0)
    obs2 = data.observe(jnp.array([5.0, 6.0]))
    assert np.isclose(obs2, 30.0)


def test_data_from_npy_and_to_npy():
    x_train = jnp.array([[1.0, 2.0]])
    y_train = jnp.array([[10.0]])
    x_test = jnp.array([[3.0, 4.0]])
    y_test = jnp.array([[20.0]])
    data = Data(x_train, y_train, x_test, y_test)
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = [
            os.path.join(tmpdir, f)
            for f in ["xtr.npy", "ytr.npy", "xte.npy", "yte.npy"]
        ]
        data.to_npy(*paths)
        loaded = Data.from_npy(*paths)
        assert jnp.allclose(loaded.x_train, x_train)
        assert jnp.allclose(loaded.y_train, y_train)
        assert jnp.allclose(loaded.x_test, x_test)
        assert jnp.allclose(loaded.y_test, y_test)


def test_create_synthetic_data_shapes():
    num_train, num_test, input_dim, embedding_dim = 5, 3, 4, 2
    data = create_synthetic_data(
        simple_model,
        num_train,
        num_test,
        input_dim,
        embedding_dim,
        measurement_noise_std=0.0,
        output_dim=1,
    )
    assert data.x_train.shape == (num_train, 1, 1, input_dim)
    assert data.x_test.shape == (num_test, 1, 1, input_dim)
    assert data.y_train.shape == (num_train, 1)
    assert data.y_test.shape == (num_test, 1)


def test_mnist_data():
    pytest.importorskip("tensorflow_datasets")
    data = get_mnist_data(num_train=10, num_test=5)
    assert data.x_train.shape[0] == 10
    assert data.y_train.shape == (10, 10)


def test_uci_data():
    try:
        data = get_uci_data(dataset="Concrete Compressive Strength", test_size=0.2)
    except Exception as error:  # network or repository unavailable
        pytest.skip(f"UCI repository not reachable: {error}")
    n_total = data.x_train.shape[0] + data.x_test_pool.shape[0] + data.x_test_glob.shape[0]
    assert n_total == 1030
    assert data.x_train.shape[1:] == (1, 1, 8)
    assert data.y_train.shape[1:] == (1, 1, 1)
    # inputs and targets are standardized over the whole dataset
    x_all = jnp.concatenate([data.x_train, data.x_test_pool, data.x_test_glob]).reshape(n_total, -1)
    y_all = jnp.concatenate([data.y_train, data.y_test_pool, data.y_test_glob]).reshape(n_total)
    assert jnp.allclose(x_all.mean(axis=0), 0.0, atol=1e-6) and jnp.allclose(x_all.std(axis=0), 1.0, atol=1e-6)
    assert abs(float(y_all.mean())) < 1e-6 and abs(float(y_all.std()) - 1.0) < 1e-6
    # the pool is the upper quartile of the held-out set along the first principal component
    n_test = data.x_test_pool.shape[0] + data.x_test_glob.shape[0]
    assert abs(data.x_test_pool.shape[0] / n_test - 0.25) < 0.02


def test_create_synthetic_normal_with_outliers_data_1d():
    data = create_synthetic_normal_with_outliers_data_1D(
        simple_model, var=1.0, mean=0.0, outliers=jnp.array([5.0, -5.0]), num_train=6, num_test=3
    )
    assert data.x_train.shape == (6, 1, 1, 1)
    assert data.x_test_pool.shape == (5, 1, 1, 1)
    assert data.y_train.shape == (6, 1)
    assert data.y_test_pool.shape == (5, 1)
