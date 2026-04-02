"""
Bayesian Experimental Design (BED) Library.

This package provides tools for optimal experimental design using Bayesian methods.
It includes implementations of various design criteria and optimization algorithms
for different model types.

Main modules:
    - base: Abstract base classes for experimental design
    - models: Concrete implementations (Linear, GP, EKF)

Typical usage:
    from bed.models import LinearGaussianModel, GaussianProcessModel
    from bed.base import BayesianExperimentalDesign
"""

__version__ = "0.0.1"
__author__ = "Shahab Sherveh"

# Import main classes for convenient access
try:
    from .models import (
        LinearGaussianModel,
        GP,
        GaussianProcessModel,
        EKF,
        Experiment1,
        generate_full_design_matrix,
    )
    from .base import Experiment, BayesianExperimentalDesign
except ImportError:
    # Handle case where dependencies aren't installed yet
    pass

__all__ = [
    "LinearGaussianModel",
    "GP",
    "GaussianProcessModel",
    "EKF",
    "Experiment1",
    "generate_full_design_matrix",
    "Experiment",
    "BayesianExperimentalDesign",
]
