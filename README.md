# AstroBayesDemo
A small demo for the Code/Astro workshop to run the complete workflow from
coding, documentation, testing, and releasing code on GitHub. The scientific
goal is to build a small framework for understanding Bayesian inference before
extending these ideas toward later PhD work.

## Install

From the project root:

```bash
python -m pip install -e ".[test]"
```

## Quick Start

```python
from astro_bayes_demo import generate_synthetic_data

data = generate_synthetic_data(n_points=50, random_seed=42)

print(data.time[:3])
print(data.flux[:3])
```

The first version of the package generates a noisy sinusoidal signal with known
truth. This gives us a controlled dataset for testing Bayesian inference methods
before moving to more complex astronomy or VLBI-like examples.

## Run Tests

```bash
python -m pytest
```
