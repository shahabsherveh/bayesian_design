# Bayesian Experimental Design

`bed` is a research-oriented Python library for Bayesian experimental design.
It contains classical linear and Gaussian-process methods, neural-network
measurement models, and sequential design driven by an extended Kalman filter.

## Requirements

- Python 3.12 or newer
- A working C/C++ toolchain may be required by scientific Python wheels
- CPU execution is supported; JAX uses CPU when CUDA-enabled packages are not installed

## Setup

Create and activate the repository virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The package imports TensorFlow for the MNIST data loader, so `tensorflow` is
included in `requirements.txt` even when running only synthetic experiments.

## Run tests

Run the complete suite:

```bash
python -m pytest tests/
```

Run a focused test:

```bash
python -m pytest tests/data_test.py::test_data_from_npy_and_to_npy
python -m pytest tests/ekf_test.py::TestEKF::test_pos
```

The pytest configuration enables the IPython debugger and disables output
capture. Some research-oriented tests are intentionally slow, display plots,
or contain breakpoints; run focused unit tests when iterating on library code.

## Run experiments

Experiments are now configuration-driven and executed through the CLI.

List bundled configurations:

```bash
bed list-configs
```

Run a configured experiment:

```bash
bed experiment --config experiment_0
```

Run without plotting (useful on headless environments):

```bash
bed experiment --config experiment_0 --no-plot
```

Quick validation without running optimization:

```bash
bed experiment --config experiment_0 --dry-run
```

You can still run the convenience wrappers in `experiments/`; they now delegate
to the same CLI config execution:

```bash
python experiments/experiment_0.py
python experiments/experiment_1.py
python experiments/experiment_2.py
python experiments/experiment_3.py
```

Configurations live under `experiments/configs/*.toml` and define model, data,
latent covariance, training, and optimizer settings. Runs are not lightweight
smoke tests: reduce `iterations`, training epochs, dataset sizes, or Monte
Carlo sample counts in the config when exploring interactively.

Available experiment variants include:

| Script | Purpose |
| --- | --- |
| `experiment_0.py` | Linear neural measurement model with a high-dimensional design |
| `experiment_1.py` | Dense neural regressor with grid-search design selection |
| `experiment_2.py` and `experiment_2_1.py`–`experiment_2_4.py` | Dense-network and distribution variations |
| `experiment_2_skew.py` | Skew-normal mixture inputs |
| `experiment_3.py` | MNIST classification using the CNN model |
| `experiment_mc.py` | Monte Carlo EPIG exploration |
| `computation_cost.py` | Timing comparison as the design pool grows |

Each sequential run compares some subset of:

- `EPIG`: linearized expected posterior predictive information gain
- `EIG`: expected information gain about latent parameters
- `EPIG-MC`: Monte Carlo EPIG approximation
- `RAND`: random design selection

The experiment API is also available directly:

```python
from bed.data import create_synthetic_data
from bed.experiments import Experiment
from bed.models import LinearNN, NeuralNetworkRegressor
```

Pass `filter_type="ukf"` to `Experiment` (or set
`experiment.filter_type` in a configuration) to use the unscented Kalman
filter; the default remains `"ekf"`.

Create a `Data` object, wrap a Flax model in a neural measurement model, and
pass both to `Experiment`. Use `run()` for one criterion or
`run_experiment()` to compare independent strategies. The latter deep-copies
the initial experiment so each strategy receives its own EKF state trajectory.
See the scripts in `experiments/` for complete configurations.

## Package layout

- `src/bed/data.py`: dataset container, synthetic data generation, and MNIST loading
- `src/bed/models/`: classical, Gaussian-process, and neural model namespaces
- `src/bed/ekf.py`, `src/bed/ukf.py`, and `src/bed/filters/`: EKF/UKF state estimation
- `src/bed/experiments/`: sequential runner and result containers
- `src/bed/base.py`: minimal generic experiment interfaces
- `experiments/`: reproducible research configurations
- `tests/`: model, data, EKF, GP, and experiment tests

The legacy-compatible imports remain available:

```python
from bed.models import GaussianProcessModel, LinearGaussianModel
from bed.experiments import Experiment
```

## License

MIT
