# Bayesian Experimental Design (BED)

A Python library for Bayesian experimental design, providing tools for optimal experimental design using information-theoretic criteria.

## Overview

This library implements various methods for Bayesian experimental design, including:

- **Linear Gaussian Models**: Design optimization for linear models with Gaussian noise
- **Gaussian Process Models**: Non-parametric Bayesian experimental design
- **Extended Kalman Filter (EKF)**: Sequential experimental design with state-space models
- **Information-theoretic criteria**: A-optimal, D-optimal, and Expected Information Gain (EIG)

## Installation

### From source

```bash
git clone https://github.com/shahabsherveh/bayesian_design.git
cd bayesian_design
pip install -e .
```

## Quick Start

## Testing

Run tests with pytest:

```bash
pytest tests/
```

Run Experimenta 1 with:

```
pytest -s --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb  tests/ekf_test.py::TestExperimentalEKF::test_run

```

## License

MIT License

## Authors

- Shahab Sherveh (<shahab.sherveh.0781@student.uu.se>)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
