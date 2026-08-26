# Copilot instructions

## Project commands

- Install the package and its dependencies in editable mode: `pip install -e .`
- Run the test suite: `pytest tests/`
- Run one test by node ID, for example:
  `pytest tests/data_test.py::test_data_from_npy_and_to_npy`
  or `pytest tests/ekf_test.py::TestEKF::test_pos`
- Run the documented experiment tests with:
  `pytest tests/experiment_test.py::TestExperiment1::test_run`
  and `pytest tests/experiment_test.py::TestExperiment1::test_epig_monte_carlo`
- The pytest configuration enables `--pdb` and IPython's terminal debugger and disables output capture (`-s`). A failing test may therefore enter an interactive debugger rather than simply exit.
- No lint or formatter command is configured in `pyproject.toml`; do not assume one exists.
- The package exposes the `bed` console script (`bed = bed.cli.main:app`), although the research workflows are primarily the scripts under `experiments/` and the tests.

## Architecture

- This is a Python 3.12+ package using a `src/` layout. The installable package is `src/bed`; `build/` and `src/bed.egg-info/` are generated artifacts and are ignored by Git.
- The public model namespace is `bed.models`, organized into `bed.models.classical`, `bed.models.gaussian_process`, and `bed.models.neural`. Their current implementations remain in the private compatibility module `bed._models` so existing imports continue to work during the refactor.
- `bed.models` contains the main model layer:
  - `LinearGaussianModel` and `GP` implement classical optimal-design criteria and CVXPy-based or exhaustive allocation methods.
  - `GaussianProcessModel` implements GP prior/posterior prediction and mutual information using scikit-learn kernels and SciPy distributions.
  - `Model` is the measurement-model interface used by the EKF. `LinearModel` is the analytical baseline.
  - `FlaxModel` and its `LinearNN`, `DenseNN`, `CNN`, and sinusoidal variants flatten Flax/NNX parameters into latent vectors. `NeuralNetworkRegressor` and `NeuralNetworkClassifier` adapt those networks to the `Model` interface and provide JAX Jacobians/training.
- `src/bed/data.py` provides the `Data` container, NPY serialization, synthetic distributions, and MNIST loading. Neural-network inputs conventionally use batched image-like shapes such as `(N, 1, 1, design_dim)`; labels/measurements are kept as JAX arrays.
- `src/bed/ekf.py` implements the EKF used for sequential design. It predicts from the current latent mean/covariance, linearizes the measurement model with `model.jacobian`, and performs measurement posterior and predictive covariance updates.
- `bed.experiments` orchestrates sequential design. Use `bed.experiments.runner` for the runner and `bed.experiments.results` for result containers; the implementation currently remains in private compatibility module `bed._experiments`:
  1. Optionally pre-train the wrapped Flax model.
  2. Build a candidate design pool by concatenating training and test inputs.
  3. Score candidates with analytical EIG, linearized EPIG, Monte Carlo EPIG, or random selection.
  4. Select a design via brute force, gradient ascent, or grid search.
  5. Observe data, update the EKF, and record RMSE/plots in `ExperimentResults` and `MultiExperimentResults`.
- `src/bed/base.py` contains minimal abstract interfaces for generic experiments and Bayesian design; the actively used sequential implementation is `bed.experiments.Experiment`.
- `bed.filters` is the public namespace for state-estimation filters and currently exposes `EKF` from `src/bed/ekf.py`.
- `experiments/` contains executable research configurations that construct models/data and call `run_experiment`; these are parameterized studies, not a separate application layer. `notebooks/` contains exploratory work.

## Repository-specific conventions

- Keep model parameter ordering stable. `FlaxModel.weight_mapping` defines the flattened latent-vector layout from the NNX state; use `weights_to_state()` and `state_to_weights()` rather than manually reshaping network parameters.
- Measurement models must support both forward evaluation `model(z, x)` and `model.jacobian(z, x)`, because EKF updates and EIG/EPIG calculations depend on both. Preserve the expected output shapes when adding models.
- Treat the design pool as the concatenation of `Data.x_train` and `Data.x_test`; `Data.observe()` uses the same indexing convention to retrieve the corresponding measurement.
- Use JAX arrays and JAX transformations for differentiable criteria and neural-network computations. NumPy/SciPy are used for classical models, distributions, plotting, and Monte Carlo sampling; avoid changing array types across these boundaries without checking shape and conversion behavior.
- Information criteria are maximized by `Experiment.optimize()` (gradient ascent), while the classical A-/D-optimal allocation criteria in `models.py` are formulated as minimization problems in CVXPy.
- Sequential experiment comparisons should use `Experiment.run_experiment()`, which deep-copies the experiment so each criterion gets an independent EKF state trajectory.
- Existing experiment tests can be computationally heavy and may display plots or contain debugger breakpoints. Prefer focused unit tests for model/data/EKF changes and invoke the long experiment tests deliberately.
- Preserve the fixed/random-seed patterns in existing experiments when reproducibility matters; JAX keys and NumPy/SciPy random state are both used in this codebase.
