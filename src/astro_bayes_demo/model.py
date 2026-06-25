"""Model functions used by the Bayesian demo."""

import numpy as np


def sinusoid_model(
    time: np.ndarray,
    amplitude: float,
    frequency: float,
    phase: float,
    offset: float,
) -> np.ndarray:
    """Evaluate a sinusoidal signal model.

    The model is

    ``y(t) = amplitude * sin(2 pi frequency t + phase) + offset``.

    Parameters
    ----------
    time
        Observation times.
    amplitude
        Height of the sine wave.
    frequency
        Number of cycles per unit time.
    phase
        Horizontal shift of the sine wave, in radians.
    offset
        Constant vertical shift.

    Returns
    -------
    numpy.ndarray
        Model values evaluated at each input time.
    """

    return amplitude * np.sin(2.0 * np.pi * frequency * time + phase) + offset
