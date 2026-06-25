"""Likelihood, prior, and posterior functions for the Bayesian demo."""

from dataclasses import dataclass

import numpy as np

from .model import sinusoid_model


@dataclass(frozen=True)
class PosteriorGrid:
    """Posterior evaluated on a regular amplitude-offset grid.

    Parameters
    ----------
    amplitudes
        Trial amplitude values.
    offsets
        Trial offset values.
    log_probability
        Log-posterior values with shape ``(n_offsets, n_amplitudes)``.
    probability
        Normalized posterior probability with the same shape as
        ``log_probability``.
    """

    amplitudes: np.ndarray
    offsets: np.ndarray
    log_probability: np.ndarray
    probability: np.ndarray

    @property
    def best_amplitude(self) -> float:
        """Amplitude at the highest posterior grid point."""

        offset_index, amplitude_index = np.unravel_index(
            np.argmax(self.probability),
            self.probability.shape,
        )
        return float(self.amplitudes[amplitude_index])

    @property
    def best_offset(self) -> float:
        """Offset at the highest posterior grid point."""

        offset_index, amplitude_index = np.unravel_index(
            np.argmax(self.probability),
            self.probability.shape,
        )
        return float(self.offsets[offset_index])


def log_likelihood(
    time: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    amplitude: float,
    offset: float,
    frequency: float = 1.0,
    phase: float = 0.0,
) -> float:
    """Calculate the Gaussian log-likelihood for a sinusoidal model.

    The likelihood answers this question:

    "If these model parameters were true, how probable are the observed data?"

    We assume each data point has independent Gaussian noise, so larger
    residuals compared with the uncertainty produce a lower log-likelihood.
    """

    model_flux = sinusoid_model(
        time=time,
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        offset=offset,
    )
    residual = flux - model_flux

    return float(
        -0.5
        * np.sum((residual / uncertainty) ** 2 + np.log(2.0 * np.pi * uncertainty**2))
    )


def log_prior(
    amplitude: float,
    offset: float,
    amplitude_bounds: tuple[float, float] = (0.0, 3.0),
    offset_bounds: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Calculate a simple uniform prior for amplitude and offset.

    The prior encodes what parameter values are allowed before seeing the data.
    This first demo uses a flat prior inside the allowed range and zero
    probability outside it.
    """

    amplitude_min, amplitude_max = amplitude_bounds
    offset_min, offset_max = offset_bounds

    amplitude_is_allowed = amplitude_min <= amplitude <= amplitude_max
    offset_is_allowed = offset_min <= offset <= offset_max

    if amplitude_is_allowed and offset_is_allowed:
        return 0.0

    return -np.inf


def log_posterior(
    time: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    amplitude: float,
    offset: float,
    frequency: float = 1.0,
    phase: float = 0.0,
    amplitude_bounds: tuple[float, float] = (0.0, 3.0),
    offset_bounds: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Calculate the log-posterior for amplitude and offset.

    Bayes' theorem says

    ``posterior proportional to likelihood * prior``.

    In log-space this becomes

    ``log_posterior = log_likelihood + log_prior``.
    """

    prior = log_prior(
        amplitude=amplitude,
        offset=offset,
        amplitude_bounds=amplitude_bounds,
        offset_bounds=offset_bounds,
    )

    if not np.isfinite(prior):
        return -np.inf

    likelihood = log_likelihood(
        time=time,
        flux=flux,
        uncertainty=uncertainty,
        amplitude=amplitude,
        offset=offset,
        frequency=frequency,
        phase=phase,
    )

    return prior + likelihood


def compute_posterior_grid(
    time: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    amplitudes: np.ndarray,
    offsets: np.ndarray,
    frequency: float = 1.0,
    phase: float = 0.0,
    amplitude_bounds: tuple[float, float] = (0.0, 3.0),
    offset_bounds: tuple[float, float] = (-1.0, 1.0),
) -> PosteriorGrid:
    """Evaluate and normalize the posterior on an amplitude-offset grid.

    This is the simplest way to see the posterior distribution. The function
    tries every combination of ``amplitudes`` and ``offsets`` and records the
    posterior score for each pair.
    """

    log_probability = np.empty((len(offsets), len(amplitudes)))

    for offset_index, offset in enumerate(offsets):
        for amplitude_index, amplitude in enumerate(amplitudes):
            log_probability[offset_index, amplitude_index] = log_posterior(
                time=time,
                flux=flux,
                uncertainty=uncertainty,
                amplitude=float(amplitude),
                offset=float(offset),
                frequency=frequency,
                phase=phase,
                amplitude_bounds=amplitude_bounds,
                offset_bounds=offset_bounds,
            )

    finite_log_probability = log_probability[np.isfinite(log_probability)]
    if finite_log_probability.size == 0:
        raise ValueError("No finite posterior values found on this grid.")

    # Subtract the maximum before exponentiating to avoid numerical overflow.
    probability = np.exp(log_probability - np.max(finite_log_probability))
    probability[~np.isfinite(probability)] = 0.0
    probability = probability / np.sum(probability)

    return PosteriorGrid(
        amplitudes=amplitudes,
        offsets=offsets,
        log_probability=log_probability,
        probability=probability,
    )
