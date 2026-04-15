from copy import deepcopy
import jax.numpy as jnp
from jax.tree_util import tree_all
from bed.models import LinearModel, NeuralNetworkFlax
import jax
from flax import nnx

from tests.utils import tree_allclose


class TestLinearModel:
    def test_jacobian(self):
        X = jnp.array([[1.0, 0.5], [0.0, 1.0]])
        z = jnp.array([[0.5], [1.0]])
        model = LinearModel(input_dim=2, rngs=nnx.Rngs(0))
        grad = model.gradient(X[0])
        breakpoint()


# class TestNeuralNetworkModel:
#     model = NeuralNetworkModel(5, hidden_dim_0=4, hidden_dim_1=4, rngs=nnx.Rngs(0))
#
#     def test_call_1(self):
#         X = jnp.array([[1], [0], [0], [0], [0]])
#         z = jnp.zeros(shape=49)
#         z.at[1].set(1)
#         output = self.model(z, X)
#         assert output.shape == (1, 1)
#         assert output[0, 0] == 0
#         breakpoint()
#
#     def test_call_2(self):
#         X = jnp.array([[1], [0], [0], [0], [0]])
#         w_0 = jnp.zeros(shape=(4, 5))
#         w_0 = w_0.at[0, 0].set(1)
#         b_0 = jnp.zeros(shape=(4, 1))
#
#         w_1 = jnp.zeros(shape=(4, 4))
#         w_1 = w_1.at[0, 0].set(1)
#         b_1 = jnp.zeros(shape=(4, 1))
#
#         w_2 = jnp.zeros(shape=(4, 1))
#         w_2 = w_2.at[0, 0].set(1)
#         b_2 = jnp.zeros(shape=(1, 1))
#
#         z = jnp.concatenate(
#             [
#                 w_0.flatten(),
#                 b_0.flatten(),
#                 w_1.flatten(),
#                 b_1.flatten(),
#                 w_2.flatten(),
#                 b_2.flatten(),
#             ]
#         )
#         output = self.model(z, X)
#         assert output.shape == (1, 1)
#         assert output[0, 0] == 1
#
#     def test_call_3(self):
#         X = jnp.array([[1], [1], [0], [0], [0]])
#         w_0 = jnp.zeros(shape=(5, 4))
#         w_0 = w_0.at[0, 0].set(1)
#         b_0 = jnp.zeros(shape=(4, 1))
#         b_0 = b_0.at[0, 0].set(-1)
#
#         w_1 = jnp.zeros(shape=(4, 4))
#         w_1 = w_1.at[0, 0].set(1)
#         b_1 = jnp.zeros(shape=(4, 1))
#         b_1 = b_1.at[0, 0].set(-1)
#
#         w_2 = jnp.zeros(shape=(4, 1))
#         w_2 = w_2.at[0, 0].set(1)
#         b_2 = jnp.zeros(shape=(1, 1))
#
#         z = jnp.concatenate(
#             [
#                 w_0.flatten(),
#                 b_0.flatten(),
#                 w_1.flatten(),
#                 b_1.flatten(),
#                 w_2.flatten(),
#                 b_2.flatten(),
#             ]
#         )[:, None]
#         output = self.model(z, X)
#         assert output.shape == (1, 1)
#         assert output[0, 0] == 0
#
#     def test_jacobian(self):
#         X = jnp.array([[1], [0], [0], [0], [0]])
#         w_0 = jnp.zeros(shape=(4, 5))
#         # w_0 = w_0.at[0, 0].set(1)
#         b_0 = jnp.zeros(shape=(4, 1))
#         w_1 = jnp.zeros(shape=(4, 4))
#         # w_1 = w_1.at[0, 0].set(1)
#         b_1 = jnp.zeros(shape=(4, 1))
#         w_2 = jnp.zeros(shape=(4, 1))
#         # w_2 = w_2.at[0, 0].set(1)
#         b_2 = jnp.zeros(shape=(1, 1))
#         z = jnp.concatenate(
#             [
#                 w_0.flatten(),
#                 b_0.flatten(),
#                 w_1.flatten(),
#                 b_1.flatten(),
#                 w_2.flatten(),
#                 b_2.flatten(),
#             ]
#         )[:, None]
#         jacobian = self.model.jacobian(z, X)
#         assert jacobian.shape == (1, 49)
#         breakpoint()
#


class TestNeuralNetworkModelFlax:
    rngs = nnx.Rngs(0)
    prior_cov = 0.1 * jnp.eye(49)
    model = NeuralNetworkFlax(5, hidden_dim_0=4, hidden_dim_1=4, rngs=rngs)

    def test_call(self):
        X = jnp.array([[1, 0, 0, 0, 0]])
        output = self.model(X)
        breakpoint()
        assert output.shape == (1, 1)

    def test_jacobian(self):
        x = jnp.array([[1, 0, 0, 0, 0]])

        grad = self.model.gradient(x)
