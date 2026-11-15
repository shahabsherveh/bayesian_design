"""Optimization interfaces used by experimental-design algorithms."""


class OptimizerBase:
    """Base interface for optimizers over a candidate search space."""

    def __init__(self, objective_fn, search_space) -> None:
        """Store the objective and search space for a concrete optimizer."""
