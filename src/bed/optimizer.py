from jax import numpy as jnp
import jax
import optax


class OptimizerBase:
    def __init__(self, objective_fn, search_space) -> None:
        pass
