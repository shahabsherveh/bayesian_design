import numpy as np


class Experiment:
    def likelihood(self, params, data, design):
        """Compute the likelihood of the data given parameters and design."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def simulate(self, params, design):
        """Simulate data given parameters and design."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class BayesianExperimentalDesign:
    def __init__(self, model: Experiment, prior):
        self.model = model
        self.prior = prior

    def experiment(self, design, num_trials=1):
        """Simulate an experiment given a design."""
        params = self.prior.sample(num_trials)
        data = self.model.simulate(params, design)
        return data, params

    def expected_information_gain(self, design, num_samples=1000):
        """Estimate the expected information gain for a given design."""
        raise NotImplementedError("This method should be implemented by subclasses.")
