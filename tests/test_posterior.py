import numpy as np

from astro_bayes_demo import (
    PosteriorGrid,
    compute_posterior_grid,
    generate_synthetic_data,
    log_likelihood,
    log_posterior,
    log_prior,
)


def test_log_prior_accepts_values_inside_bounds():
    assert log_prior(amplitude=1.5, offset=0.2) == 0.0


def test_log_prior_rejects_values_outside_bounds():
    assert np.isneginf(log_prior(amplitude=-1.0, offset=0.2))
    assert np.isneginf(log_prior(amplitude=1.5, offset=2.0))


def test_log_likelihood_prefers_true_parameters_for_low_noise_data():
    data = generate_synthetic_data(
        n_points=80,
        amplitude=1.5,
        offset=0.2,
        noise_std=0.05,
        random_seed=4,
    )

    true_loglike = log_likelihood(
        time=data.time,
        flux=data.flux,
        uncertainty=data.uncertainty,
        amplitude=1.5,
        offset=0.2,
    )
    bad_loglike = log_likelihood(
        time=data.time,
        flux=data.flux,
        uncertainty=data.uncertainty,
        amplitude=0.3,
        offset=-0.8,
    )

    assert true_loglike > bad_loglike


def test_log_posterior_rejects_parameters_outside_prior():
    data = generate_synthetic_data(random_seed=4)

    posterior = log_posterior(
        time=data.time,
        flux=data.flux,
        uncertainty=data.uncertainty,
        amplitude=-1.0,
        offset=0.2,
    )

    assert np.isneginf(posterior)


def test_compute_posterior_grid_returns_normalized_probability():
    data = generate_synthetic_data(random_seed=4)
    amplitudes = np.linspace(0.0, 3.0, 25)
    offsets = np.linspace(-1.0, 1.0, 21)

    result = compute_posterior_grid(
        time=data.time,
        flux=data.flux,
        uncertainty=data.uncertainty,
        amplitudes=amplitudes,
        offsets=offsets,
    )

    assert isinstance(result, PosteriorGrid)
    assert result.log_probability.shape == (21, 25)
    assert result.probability.shape == (21, 25)
    np.testing.assert_allclose(np.sum(result.probability), 1.0)


def test_compute_posterior_grid_finds_parameters_near_truth():
    data = generate_synthetic_data(
        n_points=80,
        amplitude=1.5,
        offset=0.2,
        noise_std=0.05,
        random_seed=4,
    )
    amplitudes = np.linspace(0.0, 3.0, 61)
    offsets = np.linspace(-1.0, 1.0, 81)

    result = compute_posterior_grid(
        time=data.time,
        flux=data.flux,
        uncertainty=data.uncertainty,
        amplitudes=amplitudes,
        offsets=offsets,
    )

    assert abs(result.best_amplitude - 1.5) < 0.1
    assert abs(result.best_offset - 0.2) < 0.1
