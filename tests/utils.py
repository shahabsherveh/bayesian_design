"""
Utility functions for testing Bayesian experimental design models.

This module provides helper functions for computing expected information gain
and related metrics for linear models, used in unit tests.
"""

from jax import numpy as jnp


def linear_model_epig(x_1, x_0, prior_cov, measurement_cov):
    """
    Compute expected posterior information gain (EPIG) for linear model.
    
    Calculates the expected information gain when adding a new measurement
    at x_1 given an existing measurement at x_0.
    
    Args:
        x_1: New design point (candidate measurement location)
        x_0: Current design point (existing measurement location)
        prior_cov: Prior covariance matrix of parameters
        measurement_cov: Measurement noise covariance
        
    Returns:
        Expected posterior information gain value
    """
    cov_0 = (x_0.T @ prior_cov @ x_0).diagonal() + measurement_cov
    cov_1 = x_1.T @ prior_cov @ x_1 + measurement_cov
    cov_cross = x_0.T @ prior_cov @ x_1
    epig = -0.5 * jnp.log(
        1 - (cov_cross.flatten() ** 2 / (cov_0.flatten() * cov_1.flatten()))
    )
    return epig


def linear_model_eig_crit(x_1, x_0, prior_cov, measurement_cov):
    """
    Compute expected information gain (EIG) criterion for linear model.
    
    Calculates the mutual information between new measurement at x_1 and
    parameters, conditioned on previous measurement at x_0.
    
    This uses the analytical formula for Gaussian models:
        EIG = 0.5 * log(det(cov_1) / det(cov_conditional))
    
    Args:
        x_1: New design point for next measurement
        x_0: Previous design point
        prior_cov: Prior covariance matrix of parameters
        measurement_cov: Measurement noise covariance
        
    Returns:
        Expected information gain criterion value
    """
    cov_0 = x_0.T @ prior_cov @ x_0 + measurement_cov
    cov_1 = x_1.T @ prior_cov @ x_1 + measurement_cov
    eig_crit = 0.5 * jnp.log(cov_1 / (cov_1 - (x_0 @ prior_cov @ x_1.T) ** 2 / cov_0))
    return eig_crit
