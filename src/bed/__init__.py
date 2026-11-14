"""Bayesian experimental-design models, filters, and experiment runners."""

# """Bayesian experimental design library."""
#
from .base import BayesianExperimentalDesign
from .ekf import EKF
from .experiments import Experiment, ExperimentResults, MultiExperimentResults
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
    NeuralNetworkClassifier,
    NeuralNetworkRegressor,
    Sinus,
    SinusInverse,
    generate_full_design_matrix,
)

#
__version__ = "0.0.1"
__author__ = "Shahab Sherveh"

__all__ = [
    "CNN",
    "EKF",
    "GP",
    "BayesianExperimentalDesign",
    "DenseNN",
    "Experiment",
    "ExperimentResults",
    "FlaxModel",
    "GaussianProcessModel",
    "LinearGaussianModel",
    "LinearModel",
    "LinearNN",
    "Model",
    "MultiExperimentResults",
    "NeuralNetworkClassifier",
    "NeuralNetworkRegressor",
    "Sinus",
    "SinusInverse",
    "generate_full_design_matrix",
]
