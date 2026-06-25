"""Run the complete AstroBayesDemo workflow.

This script:

1. Generates synthetic sinusoidal data.
2. Computes a posterior grid for amplitude and offset.
3. Saves a data/model plot.
4. Saves a posterior probability map.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from astro_bayes_demo import (
    compute_posterior_grid,
    generate_synthetic_data,
    plot_data_with_model,
    plot_posterior_grid,
)


def main() -> None:
    """Run the basic Bayesian inference demo."""

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    data = generate_synthetic_data(
        n_points=60,
        amplitude=1.5,
        frequency=1.0,
        phase=0.0,
        offset=0.2,
        noise_std=0.2,
        random_seed=42,
    )

    amplitudes = np.linspace(0.0, 3.0, 101)
    offsets = np.linspace(-1.0, 1.0, 101)

    posterior = compute_posterior_grid(
        time=data.time,
        flux=data.flux,
        uncertainty=data.uncertainty,
        amplitudes=amplitudes,
        offsets=offsets,
    )

    print(f"Best amplitude: {posterior.best_amplitude:.3f}")
    print(f"Best offset: {posterior.best_offset:.3f}")

    fig, _ = plot_data_with_model(
        data=data,
        amplitude=posterior.best_amplitude,
        offset=posterior.best_offset,
    )
    fig.savefig(output_dir / "data_with_best_model.png", dpi=200)
    plt.close(fig)

    fig, _ = plot_posterior_grid(posterior)
    fig.savefig(output_dir / "posterior_grid.png", dpi=200)
    plt.close(fig)

    print(f"Saved figures to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
