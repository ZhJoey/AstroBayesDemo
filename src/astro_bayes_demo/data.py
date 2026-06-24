"""Synthetic datasets for Bayesian inference demos."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticData:
    """Container for one noisy sinusoidal dataset.

    Parameters
    ----------
    time
        Observation times.
    flux
        Noisy observed values.
    uncertainty
        One-sigma Gaussian uncertainty for each observation.
    truth
        Noise-free signal evaluated at ``time``.
    """

    time: np.ndarray
    flux: np.ndarray
    uncertainty: np.ndarray
    truth: np.ndarray


def generate_synthetic_data(
    n_points: int = 50,
    amplitude: float = 1.5,
    frequency: float = 1.0,
    phase: float = 0.0,
    offset: float = 0.2,
    noise_std: float = 0.2,
    random_seed: int | None = None,
) -> SyntheticData:
    """Generate a noisy sinusoidal dataset with known truth.

    The signal is

    ``y(t) = amplitude * sin(2 pi frequency t + phase) + offset``.

    Parameters
    ----------
    n_points
        Number of observations to simulate.
    amplitude
        Signal amplitude.
    frequency
        Signal frequency in cycles per unit time.
    phase
        Signal phase in radians.
    offset
        Constant vertical offset.
    noise_std
        Standard deviation of the Gaussian observational noise.
    random_seed
        Optional seed for reproducible random noise.

    Returns
    -------
    SyntheticData
        Time, noisy flux, per-point uncertainty, and noise-free truth.
    """

    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    if noise_std <= 0:
        raise ValueError("noise_std must be positive.")

    rng = np.random.default_rng(random_seed)
    time = np.linspace(0.0, 1.0, n_points)
    truth = amplitude * np.sin(2.0 * np.pi * frequency * time + phase) + offset
    uncertainty = np.full(n_points, noise_std)
    flux = truth + rng.normal(0.0, noise_std, size=n_points)

    return SyntheticData(
        time=time,
        flux=flux,
        uncertainty=uncertainty,
        truth=truth,
    )
