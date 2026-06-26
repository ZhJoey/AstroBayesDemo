"""Plotting helpers for the Bayesian demo."""

import numpy as np
from matplotlib import pyplot as plt

from .data import SyntheticData
from .model import sinusoid_model
from .posterior import PosteriorGrid


def plot_data_with_model(
    data: SyntheticData,
    amplitude: float,
    offset: float,
    frequency: float = 1.0,
    phase: float = 0.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot noisy data with a sinusoidal model curve.

    Parameters
    ----------
    data
        Synthetic dataset to plot.
    amplitude
        Model amplitude.
    offset
        Model offset.
    frequency
        Model frequency in cycles per unit time.
    phase
        Model phase in radians.

    Returns
    -------
    tuple
        Matplotlib figure and axes.
    """

    model_time = np.linspace(np.min(data.time), np.max(data.time), 500)
    model_flux = sinusoid_model(
        time=model_time,
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        offset=offset,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.errorbar(
        data.time,
        data.flux,
        yerr=data.uncertainty,
        fmt="o",
        color="tab:blue",
        ecolor="0.65",
        elinewidth=1.0,
        capsize=2.0,
        label="Synthetic data",
    )
    ax.plot(model_time, model_flux, color="tab:red", linewidth=2.0, label="Model")
    ax.plot(data.time, data.truth, color="0.2", linestyle="--", label="Truth")

    ax.set_xlabel("Time")
    ax.set_ylabel("Flux")
    ax.set_title("Synthetic Data And Sinusoidal Model")
    ax.legend()

    return fig, ax


def plot_posterior_grid(
    posterior: PosteriorGrid,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the normalized posterior probability as a heatmap."""

    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    image = ax.pcolormesh(
        posterior.amplitudes,
        posterior.offsets,
        posterior.probability,
        shading="auto",
        cmap="viridis",
    )
    ax.plot(
        posterior.best_amplitude,
        posterior.best_offset,
        marker="x",
        markersize=9,
        markeredgewidth=2,
        color="white",
        label="Best grid point",
    )

    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Offset")
    ax.set_title("Posterior Probability Map")
    ax.legend()
    fig.colorbar(image, ax=ax, label="Posterior probability")

    return fig, ax
