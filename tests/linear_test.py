import numpy as np
import ipdb
from bed.models import LinearGaussianModel


class TestLinearGaussianModel:
    X = np.array(
        [
            -2.1213,
            2.1213,
            -2.2981,
            1.9284,
            -2.4575,
            1.7207,
            -2.5981,
            1.5000,
            -2.7189,
            1.2679,
            -2.8191,
            1.0261,
            -2.8978,
            0.7765,
            -2.9544,
            0.5209,
            -2.9886,
            0.2615,
            -3.0000,
            0.0000,
            1.5000,
            0.0000,
            1.4772,
            -0.2605,
            1.4095,
            -0.5130,
            1.2990,
            -0.7500,
            1.1491,
            -0.9642,
            0.9642,
            -1.1491,
            0.7500,
            -1.2990,
            0.5130,
            -1.4095,
            0.2605,
            -1.4772,
            0.0000,
            -1.5000,
        ]
    ).reshape(
        20,
        2,
    )
    d_points = np.array([0, 9])
    d_allocation = np.array([0.5, 0.5])
    a_points = np.array([0, 9, 19])
    a_allocation = np.array([0.30, 0.38, 0.32])

    n = 20
    observable_designs = np.random.choice(np.arange(X.shape[0]), size=n, replace=False)

    def test_d_optimality(self):
        np.random.seed(42)
        model = LinearGaussianModel(
            theta=np.array([1.0, 2.0]),
            sigma=1.0,
            X=self.X,
            R=np.zeros((2, 2)),
        )
        problem = model.D_opt_DCP()
        model.plot_optimal_design(problem)
        allocations = problem.var_dict["eta"].value
        assert np.equal(self.d_points, np.argwhere(allocations).flatten()).all()
        assert np.allclose(allocations[self.d_points], self.d_allocation, atol=1e-2)

    def test_a_optimality(self):
        np.random.seed(42)
        model = LinearGaussianModel(
            theta=np.array([1.0, 2.0]),
            sigma=1.0,
            X=self.X,
            R=np.zeros((2, 2)),
        )
        problem = model.A_opt_DCP()
        model.plot_optimal_design(problem)
        allocations = problem.var_dict["eta"].value
        assert np.equal(self.a_points, np.argwhere(allocations).flatten()).all()
        assert np.allclose(allocations[self.a_points], self.a_allocation, atol=1e-2)
