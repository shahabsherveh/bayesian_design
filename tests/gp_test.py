from bed.models import (
    GaussianProcessModel,
    generate_full_design_matrix,
    multivariate_normal,
)
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
import numpy as np


class TestGaussianProcessModel:
    X = generate_full_design_matrix((5, 5))
    np.random.seed(0)
    X_obs_idx = np.random.choice(np.arange(len(X)), size=10, replace=False)
    X_obs = X[X_obs_idx]
    kernel_obs = RBF()
    y_obs = multivariate_normal(
        mean=np.zeros(len(X_obs)), cov=kernel_obs(X_obs, X_obs)
    ).rvs()
    import time

    t = 1000 * time.time()  # current time in milliseconds
    np.random.seed(int(t) % 2**32)
    kernel = RBF(length_scale=1.0)
    model = GaussianProcessModel(kernel=kernel, X=X)

    def test_initialization(self):
        assert model.kernel == self.kernel
        assert np.array_equal(model.X, self.X)

    def test_prior_sample(self):
        prior = self.model.prior()
        sample = prior.rvs()
        self.model.plot_sample(sample)

    def test_predictive_distribution(self):
        pred_dist = self.model.predictive_distribution(self.X_obs_idx, self.y_obs)
        pred_sample = pred_dist.rvs()
        assert np.isclose(self.y_obs, pred_sample[self.X_obs_idx]).all()
        self.model.plot_sample(pred_sample, observed_idx=self.X_obs_idx)

    def test_mutual_information(self):
        mi = self.model.mutual_information(self.X_obs_idx)
        breakpoint()
        assert mi >= 0
