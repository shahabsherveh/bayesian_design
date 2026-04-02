from jax import numpy as jnp


def linear_model_epig(x_1, x_0, prior_cov, measurement_cov):
    cov_0 = (x_0.T @ prior_cov @ x_0).diagonal() + measurement_cov
    cov_1 = x_1.T @ prior_cov @ x_1 + measurement_cov
    cov_cross = x_0.T @ prior_cov @ x_1
    epig = -0.5 * jnp.log(
        1 - (cov_cross.flatten() ** 2 / (cov_0.flatten() * cov_1.flatten()))
    )
    breakpoint()
    return epig


def linear_model_eig_crit(x_1, x_0, prior_cov, measurement_cov):
    cov_0 = x_0.T @ prior_cov @ x_0.T + measurement_cov
    cov_1 = x_1 @ prior_cov @ x_1.T + measurement_cov
    eig_crit = 0.5 * jnp.log(cov_1 / (cov_1 - (x_0 @ prior_cov @ x_1.T) ** 2 / cov_0))
