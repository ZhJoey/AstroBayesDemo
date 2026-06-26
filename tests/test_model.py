import numpy as np

from astro_bayes_demo import sinusoid_model


def test_sinusoid_model_returns_expected_shape():
    time = np.linspace(0.0, 1.0, 30)
    model = sinusoid_model(
        time=time,
        amplitude=1.5,
        frequency=1.0,
        phase=0.0,
        offset=0.2,
    )

    assert model.shape == time.shape


def test_sinusoid_model_with_zero_amplitude_returns_offset():
    time = np.linspace(0.0, 1.0, 30)
    model = sinusoid_model(
        time=time,
        amplitude=0.0,
        frequency=1.0,
        phase=0.0,
        offset=0.2,
    )

    np.testing.assert_allclose(model, 0.2)
