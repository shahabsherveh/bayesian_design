# Experiment guide

## Basic workflow

From the repository root:

```bash
source .venv/bin/activate
bed list-configs
bed experiment --config experiment_0
```

Configurations are defined in `experiments/configs/*.toml`. The script files
in `experiments/` are wrappers that call the same CLI entrypoint.

To inspect a config without running optimization:

```bash
bed experiment --config experiment_0 --dry-run
```

## Choosing a strategy

`Experiment.run_experiment()` accepts a list of strategy labels:

```python
results = experiment.run_experiment(
    experiments=["EPIG", "EIG", "EPIG-MC", "RAND"],
    iterations=10,
    optimizer_method="grid_search",
    optimizer_params={"num_samples": 200},
)
```

`optimizer_method` can be:

- `brute_force`: score every point in the candidate design pool
- `grid_search`: sample points uniformly within the pool bounds
- `gradient_ascent`: optimize a continuous design with JAX gradients

For `grid_search`, use `num_samples`. For `gradient_ascent`, use `lr` and
`max_iters`. `brute_force` is usually the simplest baseline when the pool is
small.

## Building an experiment

The sequential runner expects:

1. A `Data` instance containing training and test designs and measurements.
2. A `Model` implementation with both `__call__(z, x)` and `jacobian(z, x)`.
3. A latent covariance (`latent_cov`) and measurement covariance
   (`measurement_error`).

For neural models, construct a Flax/NNX model first and wrap it:

```python
from flax import nnx
from bed.data import create_synthetic_data
from bed.experiments import Experiment
from bed.models import LinearNN, NeuralNetworkRegressor

model = LinearNN(input_dim=2, rngs=nnx.Rngs(0))
measurement_model = NeuralNetworkRegressor(model)
```

`Experiment` builds its candidate pool by concatenating `data.x_train` and
`data.x_test`. `Data.observe()` uses the same ordering, so custom data
providers must preserve that relationship.

## Initialising the belief by empirical Bayes

By default the filter processes the warm-start observations itself, starting
from `latent_cov` and `measurement_error` as configured. With
`init = "empirical_bayes"` the runner instead fits the *scale* of the prior
covariance (and, with `fit_noise = true`, of the measurement covariance) by
maximising the marginal likelihood of the model linearised at the MAP of the
warm-start observations, and starts every criterion from the Laplace belief
at that MAP (`bed.empirical_bayes.empirical_bayes_init`). The configured
matrices act as shapes; only their scales are fitted, which is all a handful
of warm-start points can support. For a linear model the procedure is exact.

```toml
[experiment]
init = "empirical_bayes"      # or "filter" (default)

[empirical_bayes]             # optional
fit_noise = false             # also fit the measurement-covariance scale
iterations = 3                # MAP / hyperparameter alternations
```

The fitted scales and the evidence are stored on the warm-start result as
`init_info` (an `EmpiricalBayesResult`). The same function can be called
directly on any `Model` with `__call__` and `jacobian`, e.g. to set
`latent_cov` from pilot data before building an `Experiment`.

## Runtime and reproducibility

The research scripts use both JAX PRNG keys and NumPy/SciPy random sampling.
Keep the existing seeds when comparing strategies. Plotting is enabled in
some configurations and requires a Matplotlib backend; disable intermediate
plots with `plot_inter_results=False` when running headlessly.

For a quick iteration, lower:

- `epochs` or `iterations`
- neural-network `training_kwargs["epochs"]`
- `optimizer_params["num_samples"]`
- Monte Carlo `num_latent_samples`
- `num_train` and `num_test`

The MNIST experiment downloads data through TensorFlow Datasets on its first
run and is substantially heavier than the synthetic experiments.
