"""Model implementations used by Bayesian experimental design.

The implementation is kept behind ``bed._models`` for now so existing
``from bed.models import ...`` imports remain compatible while new code can
import by model family.
"""

from bed._models import (
    CNN,
    DenseNN,
    FlaxModel,
    GP,
    GaussianProcessModel,
    LinearGaussianModel,
    LinearModel,
    LinearNN,
    Model,
    NeuralNetworkBase,
    NeuralNetworkClassifier,
    NeuralNetworkRegressor,
    Sinus,
    SinusInverse,
    generate_full_design_matrix,
)
from scipy.stats import multivariate_normal

__all__ = [
    "CNN",
    "DenseNN",
    "FlaxModel",
    "GP",
    "GaussianProcessModel",
    "LinearGaussianModel",
    "LinearModel",
    "LinearNN",
    "Model",
    "NeuralNetworkBase",
    "NeuralNetworkClassifier",
    "NeuralNetworkRegressor",
    "Sinus",
    "SinusInverse",
    "generate_full_design_matrix",
    "multivariate_normal",
]
