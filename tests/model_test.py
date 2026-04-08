import jax.numpy as jnp
from bed.models import LinearModel


class TestLinearModel:
    def test_jacobian(self):
        X = jnp.array([[1.0, 0.5], [0.0, 1.0]])
        z = jnp.array([[0.5], [1.0]])
        model = LinearModel()
        assert model.jacobian(z, X)
