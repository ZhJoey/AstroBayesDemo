"""Small Bayesian inference demos for astronomy-like synthetic data."""

from .data import SyntheticData, generate_synthetic_data
from .model import sinusoid_model
from .posterior import (
    PosteriorGrid,
    compute_posterior_grid,
    log_likelihood,
    log_posterior,
    log_prior,
)
from .plotting import plot_data_with_model, plot_posterior_grid

__all__ = [
    "PosteriorGrid",
    "SyntheticData",
    "compute_posterior_grid",
    "generate_synthetic_data",
    "log_likelihood",
    "log_posterior",
    "log_prior",
    "plot_data_with_model",
    "plot_posterior_grid",
    "sinusoid_model",
]
