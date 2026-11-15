"""Shared measurement-model base class."""

from __future__ import annotations

from typing import Any


class Model:
    """
    Base class for measurement models used with Extended Kalman Filter.

    Defines the interface for models that map latent states and designs
    to observations. Subclasses must implement __call__ for forward evaluation.

    Note:
        The jacobian method uses JAX automatic differentiation to compute
        gradients needed for EKF linearization.
    """

    def jacobian(self, z, x):
        """Return the Jacobian of the measurement with respect to ``z``."""
        raise NotImplementedError("Subclasses must implement the jacobian method.")

    def train(self, x_train, y_train, rngs: Any, epochs: int = 200, learning_rate: float = 0.1, **kwargs):
        """
        Train the model on given data.

        Args:
            x_train: Training design matrix of shape (n_samples, n_features)
            y_train: Training observations of shape (n_samples,)
            rngs: Optional PRNG key/collection for stochastic training.
            epochs: Number of optimisation epochs.
            learning_rate: Optimiser step size.

        Note:
            This is a placeholder. Subclasses can implement training logic if needed.
        """

    def __call__(self, z, x, **kwargs):
        """
        Evaluate measurement model.

        Args:
            z: Latent state vector
            x: Design vector or matrix

        Returns:
            Predicted measurement(s)

        Note:
            This is a placeholder. Subclasses must override this method.
        """

        raise NotImplementedError("Subclasses must implement the __call__ method.")
