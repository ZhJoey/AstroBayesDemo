import numpy as np
import pytest

from astro_bayes_demo import SyntheticData, generate_synthetic_data


def test_generate_synthetic_data_shapes():
    data = generate_synthetic_data(n_points=20, random_seed=1)

    assert isinstance(data, SyntheticData)
    assert data.time.shape == (20,)
    assert data.flux.shape == (20,)
    assert data.uncertainty.shape == (20,)
    assert data.truth.shape == (20,)


def test_generate_synthetic_data_is_reproducible_with_seed():
    first = generate_synthetic_data(n_points=20, random_seed=12)
    second = generate_synthetic_data(n_points=20, random_seed=12)

    np.testing.assert_allclose(first.flux, second.flux)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_points": 0}, "n_points must be positive"),
        ({"noise_std": 0.0}, "noise_std must be positive"),
    ],
)
def test_generate_synthetic_data_validates_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        generate_synthetic_data(**kwargs)
