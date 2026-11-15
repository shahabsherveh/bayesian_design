"""Backward-compatible re-export of the model implementations.

The implementations now live in per-family modules under ``bed.models``; this
module is a thin shim so existing ``from bed._models import ...`` imports keep
working.
"""

from .models import (
    CNN,
    GP,
    DenseNN,
    FlaxModel,
    GaussianProcessModel,
    LinearGaussianModel,
    LinearModel,
    LinearNN,
    Model,
    NeuralNetworkBase,
    NeuralNetworkClassifier,
    NeuralNetworkRegressor,
    NNFLax,
    Sinus,
    SinusInverse,
    generate_full_design_matrix,
)

__all__ = [
    "CNN",
    "GP",
    "DenseNN",
    "FlaxModel",
    "GaussianProcessModel",
    "LinearGaussianModel",
    "LinearModel",
    "LinearNN",
    "Model",
    "NNFLax",
    "NeuralNetworkBase",
    "NeuralNetworkClassifier",
    "NeuralNetworkRegressor",
    "Sinus",
    "SinusInverse",
    "generate_full_design_matrix",
]
