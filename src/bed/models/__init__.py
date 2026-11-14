"""Model implementations used by Bayesian experimental design.

The implementation is kept behind ``bed._models`` for now so existing
``from bed.models import ...`` imports remain compatible while new code can
import by model family.
"""

from scipy.stats import multivariate_normal

from bed._models import (
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
    "multivariate_normal",
]
