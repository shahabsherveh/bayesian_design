import os
import tempfile
import numpy as np
import jax.numpy as jnp
from bed.data import Data, create_synthetic_data, get_mnist_data, get_uci_data


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
    data = get_mnist_data(num_train=10, num_test=5)
    breakpoint()

def test_uci_data():
    data = get_uci_data(dataset="",test_size=.2)
    breakpoint()
