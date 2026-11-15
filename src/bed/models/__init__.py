"""Model implementations used by Bayesian experimental design.

The implementations live in per-family modules beside this package
(``classical``, ``gaussian_process``, ``neural``) with the shared ``Model`` base in
``_base``. ``bed._models`` remains a thin re-export shim for backward compatibility.
"""

from ._base import Model
from .classical import (
    LinearGaussianModel,
    LinearModel,
    generate_full_design_matrix,
)
from .gaussian_process import GP, GaussianProcessModel
from .neural import (
    CNN,
    DenseNN,
    FlaxModel,
    LinearNN,
    NeuralNetworkBase,
    NeuralNetworkClassifier,
    NeuralNetworkRegressor,
    NNFLax,
    Sinus,
    SinusInverse,
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
