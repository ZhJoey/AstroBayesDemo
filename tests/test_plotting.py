import matplotlib
import numpy as np

matplotlib.use("Agg")

from astro_bayes_demo import (  # noqa: E402
    compute_posterior_grid,
    generate_synthetic_data,
    plot_data_with_model,
    plot_posterior_grid,
)


def test_plot_data_with_model_returns_figure_and_axes():
    data = generate_synthetic_data(random_seed=2)

    fig, ax = plot_data_with_model(data=data, amplitude=1.5, offset=0.2)

    assert fig is not None
    assert ax.get_xlabel() == "Time"


def test_plot_posterior_grid_returns_figure_and_axes():
    data = generate_synthetic_data(random_seed=2)
    posterior = compute_posterior_grid(
        time=data.time,
        flux=data.flux,
        uncertainty=data.uncertainty,
        amplitudes=np.linspace(0.0, 3.0, 25),
        offsets=np.linspace(-1.0, 1.0, 21),
    )

    fig, ax = plot_posterior_grid(posterior)

    assert fig is not None
    assert ax.get_xlabel() == "Amplitude"
