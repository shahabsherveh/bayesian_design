"""
Base classes for Bayesian experimental design.

This module provides abstract base classes that define the interface for
experimental design problems and optimization strategies.
"""

import copy as _copy


class Experiment:
    """
    Abstract base class for experimental models.
    
    This class defines the interface that all experimental models must implement.
    Subclasses should implement methods for computing likelihoods and simulating
    experimental outcomes.
    """

    def likelihood(self, params, data, design):
        """
        Compute the likelihood of observed data given parameters and design.
        
        Args:
            params: Model parameters (e.g., regression coefficients, GP hyperparameters)
            data: Observed experimental outcomes
            design: Experimental design specification (e.g., measurement locations)
            
        Returns:
            Likelihood value p(data | params, design)
            
        Raises:
            NotImplementedError: This is an abstract method that must be implemented
        """
        """Return the measurement Jacobian with respect to latent state ``z``."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def simulate(self, params, design):
        """
        Simulate experimental data given parameters and design.
        
        Args:
            params: True model parameters
            design: Experimental design specification
            
        Returns:
            Simulated data from the model
            
        Raises:
            NotImplementedError: This is an abstract method that must be implemented
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class BayesianExperimentalDesign:
    """
    Bayesian experimental design framework.
    
    This class provides utilities for running Bayesian optimal experimental design,
    including simulation and information gain estimation.
    
    Attributes:
        model: An Experiment instance defining the forward model
        prior: Prior distribution over model parameters
    """

    def __init__(self, model: Experiment, prior):
        """
        Initialize Bayesian experimental design.
        
        Args:
            model: Experiment instance with likelihood and simulate methods
            prior: Prior distribution with a sample() method
        """
        self.model = model
        self.prior = prior

    def experiment(self, design, num_trials=1):
        """
        Simulate an experiment given a design.
        
        Samples parameters from the prior and simulates outcomes using the model.
        
        Args:
            design: Experimental design specification
            num_trials: Number of independent experimental trials to simulate
            
        Returns:
            tuple: (data, params) where data is simulated outcomes and params
                   are the sampled parameters
        """
        params = self.prior.sample(num_trials)
        data = self.model.simulate(params, design)
        return data, params

    def expected_information_gain(self, design, num_samples=1000):
        """
        Estimate the expected information gain (EIG) for a given design.
        
        The EIG measures the expected reduction in uncertainty about model
        parameters after observing data from the proposed design.
        
        Args:
            design: Experimental design to evaluate
            num_samples: Number of Monte Carlo samples for estimation
            
        Returns:
            Estimated expected information gain
            
        Raises:
            NotImplementedError: This method should be implemented by subclasses
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class FilterSnapshotMixin:
    """Belief-only copies of a state-estimation filter for ``trace`` recording.

    A sequential run records the filter's belief before each epoch. Deep-copying
    the whole filter would clone the measurement ``model`` every epoch even though
    the model is set once and never mutated during a run — for a moderately sized
    network that is tens of megabytes per epoch. :meth:`snapshot` instead returns a
    new filter that *shares* ``model`` and copies only the mutable belief
    attributes listed in ``_snapshot_copied_attrs``.

    Subclasses set ``_snapshot_copied_attrs`` to the attributes that must be
    independent of the live filter (e.g. ``("state_prior", "measurement_error")``).
    Attributes are copied in the declared order, so list cheap immutable scalars
    before any property whose setter depends on them.
    The snapshot is an ordinary instance of the subclass, so it supports the same
    methods (``get_state_posterior``, ``measurement_prior``, ...) and can itself be
    copied by callers.
    """

    # Attributes copied into a snapshot. ``model`` is deliberately excluded: it is
    # shared because it is immutable for the duration of a run.
    _snapshot_copied_attrs: tuple[str, ...] = ("state_prior", "measurement_error")

    def snapshot(self):
        """Return a cheap copy sharing ``model`` and copying the recorded belief."""
        clone = type(self).__new__(type(self))
        clone.model = self.model
        for name in self._snapshot_copied_attrs:
            # Deep-copy only the declared belief attributes (arrays / tuples of
            # arrays), never the model graph.
            setattr(clone, name, _copy.deepcopy(getattr(self, name)))
        return clone
