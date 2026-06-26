# AstroBayesDemo
A small demo for the Code/Astro workshop to run the complete workflow from
coding, documentation, testing, and releasing code on GitHub. The scientific
goal is to build a small framework for understanding Bayesian inference before
extending these ideas toward later PhD work.

## Install

For development, install the local project in editable mode from the project
root:

```bash
python -m pip install -e ".[test]"
```

The `-e` flag means edits to files in `src/astro_bayes_demo/` are immediately
available in the installed package. The `[test]` extra installs `pytest`.

To install the test release from TestPyPI:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  astro-bayes-demo
```

After a future release to the main PyPI index, install with:

```bash
python -m pip install astro-bayes-demo
```

The package is installed as `astro-bayes-demo`, but imported in Python as
`astro_bayes_demo`.

## Quick Start

```python
import numpy as np

from astro_bayes_demo import compute_posterior_grid, generate_synthetic_data

data = generate_synthetic_data(n_points=50, random_seed=42)

posterior = compute_posterior_grid(
    time=data.time,
    flux=data.flux,
    uncertainty=data.uncertainty,
    amplitudes=np.linspace(0.0, 3.0, 101),
    offsets=np.linspace(-1.0, 1.0, 101),
)

print(posterior.best_amplitude)
print(posterior.best_offset)
```

The first version of the package generates a noisy sinusoidal signal with known
truth. This gives us a controlled dataset for testing Bayesian inference methods
before moving to more complex astronomy or VLBI-like examples.

## Run The Demo

```bash
python examples/basic_demo.py
```

The demo saves two figures in `outputs/`:

- `data_with_best_model.png`
- `posterior_grid.png`

## Run Tests

```bash
python -m pytest
```
