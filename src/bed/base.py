"""
Base classes for Bayesian experimental design.

This module provides abstract base classes that define the interface for
experimental design problems and optimization strategies.
"""

import numpy as np


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
