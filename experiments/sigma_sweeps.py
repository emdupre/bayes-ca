from itertools import product

import click
import jax.numpy as jnp
from jax import jit, vmap
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from tensorflow_probability.substrates import jax as tfp

from bayes_ca.prox_grad import pgd_jaxopt

tfd = tfp.distributions


def stagger_data(gap, num_timesteps, num_features):
    """
    Hardcoding a two subject model with a single, staggered jump between
    two Gaussian states with means at -1 and +1.
    """
    offset_one = (num_timesteps // 2) - (gap.astype(int) // 2)
    means_one = jnp.ones((num_timesteps, num_features))
    mask = jnp.arange(num_timesteps) >= offset_one
    means_one = jnp.where(mask[:, None], means_one, -1)

    offset_two = (num_timesteps // 2) + (gap.astype(int) // 2)
    means_two = jnp.ones((num_timesteps, num_features))
    mask = jnp.arange(num_timesteps) >= offset_two
    means_two = jnp.where(mask[:, None], means_two, -1)

    subj_means = jnp.stack((means_one, means_two), axis=0)

    # calculate and return the average of these timeseries
    # to use as the initialization for PGD
    x0 = jnp.average(subj_means, axis=0)

    return subj_means, x0


def sample_mu0(params, num_timesteps, num_features, mu_pri, sigmasq_pri, hazard_rates):
    """
    For provided params, generate sample data and perform PGD to sample $\mu_0$.
    """
    sigmasq_subj, gap = params
    means, x0 = stagger_data(gap, num_timesteps, num_features)
    results = pgd_jaxopt(x0, means, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates)
    return results.params


def sample_mu0_true_x0(
    params, x0, num_timesteps, num_features, mu_pri, sigmasq_pri, hazard_rates
):
    """
    For provided params, generate sample data and perform PGD to sample $\mu_0$.
    """
    sigmasq_subj, gap = params
    means, _ = stagger_data(gap, num_timesteps, num_features)
    results = pgd_jaxopt(x0, means, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates)
    return results.params


def plot_mu0s(
    ax,
    mu_pri,
    sigma_pri,
    num_timesteps,
    num_features,
    hazard_rates,
    max_gap,
    sigma_val,
    n_samples,
):
    """ """
    gaps = jnp.linspace(0, max_gap, n_samples)
    params = jnp.asarray(list(product([sigma_val**2], gaps)))

    mu0s = jit(
        vmap(sample_mu0, in_axes=(0, None, None, None, None, None)), static_argnums=(1, 2)
    )(params, num_timesteps, num_features, mu_pri, sigma_pri**2, hazard_rates)

    ax.set_title(f"Sampled $\mu_0$ at $\sigma^2_{{subj}}$ = {sigma_val**2}")
    colors = plt.cm.viridis(jnp.linspace(0, 1, n_samples))
    for i, mu0 in enumerate(mu0s):
        p = ax.plot(mu0, c=colors[i], alpha=0.8, label=f"Sampled $\mu_0$, {gaps[i]} stagger")
    ax.set_xlabel("Time", labelpad=10)
    ax.set_yticks([-1, 0, 1])
    ax.spines[["right", "top"]].set_visible(False)

    return ax


def plot_param_sweep(
    ax,
    vline,
    mu_pri,
    sigma_pri,
    num_timesteps,
    num_features,
    hazard_rates,
    max_gap,
    max_sigmasq,
    n_samples,
):
    """ """
    gaps = jnp.linspace(1, max_gap, n_samples)
    sigmasqs = jnp.linspace(0.01, max_sigmasq, n_samples)
    params = jnp.asarray(list(product(sigmasqs, gaps)))

    mu0s = jit(
        vmap(sample_mu0, in_axes=(0, None, None, None, None, None)), static_argnums=(1, 2)
    )(params, num_timesteps, num_features, mu_pri, sigma_pri**2, hazard_rates)

    def count_changepoints(mu0):
        """ """
        _, counts = jnp.unique(mu0, return_counts=True)
        return jnp.count_nonzero(counts) > 2

    count_cp = jnp.asarray([count_changepoints(mu0) for mu0 in mu0s])
    sigma_by_gap = jnp.reshape(count_cp, (n_samples, n_samples))

    # check, for each sigma value, whether we still have 2 states when
    # increasing stagger distance...
    diff_dist = sigma_by_gap[:, :-1] != sigma_by_gap[:, 1:]
    diff_dist = jnp.insert(diff_dist, 0, False, axis=1)

    # ...and at which index the numer of cp's changes, if it occurs.
    idx = [jnp.nonzero(d, size=1, fill_value=False) for d in diff_dist]
    gap_threshold = gaps[jnp.asarray(idx)].squeeze()

    hazard_prob = hazard_rates[0]
    beta = -jnp.log(hazard_prob / (1 - hazard_prob))

    ax.plot(sigmasqs, gap_threshold, c="#bc3978")
    ax.plot(sigmasqs, [(beta * sigmasq) for sigmasq in sigmasqs], c="#52abef")
    ax.axvline(x=(vline**2), color="black", linestyle=(5, (10, 3)), linewidth=0.75)

    ax.set_xlabel("$\sigma^2_{{subj}}$ value", labelpad=10)
    ax.set_ylabel("Stagger distance", labelpad=15)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_title(f"Transition to 2 changepoints")

    return ax


@click.command()
@click.option("--mu_pri", default=0.0, help="")
@click.option("--sigma_pri", default=1.0, help="")
@click.option("--sigma", default=2.0, help="")
@click.option("--hazard_prob", default=0.01, help="")
@click.option("--num_features", default=1, help="")
@click.option("--num_timesteps", default=300, help="")
@click.option("--x0_strategy", default="average", help="")
def main(mu_pri, sigma_pri, sigma, hazard_prob, num_features, num_timesteps, x0_strategy):
    """ """
    # hardcoded params
    max_duration = num_timesteps
    hazard_rates = hazard_prob * jnp.ones(max_duration)
    hazard_rates = hazard_rates.at[-1].set(1.0)

    fig, axs = plt.subplot_mosaic([["A", "B"]], layout="constrained", figsize=(9, 4), dpi=300)

    for label, ax in axs.items():
        # label physical distance to the left and up:
        trans = mtransforms.ScaledTranslation(-40 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(
            0.0,
            1.0,
            label,
            transform=ax.transAxes + trans,
            fontsize="xx-large",
            va="bottom",
            fontfamily="sans-serif",
            fontweight="bold",
        )

    panel_a = plot_param_sweep(
        axs["A"],
        sigma,
        mu_pri,
        sigma_pri,
        num_timesteps,
        num_features,
        hazard_rates,
        max_gap=50,
        max_sigmasq=9.0,
        n_samples=50,
    )
    panel_b = plot_mu0s(
        axs["B"],
        mu_pri,
        sigma_pri,
        num_timesteps,
        num_features,
        hazard_rates,
        max_gap=50,
        sigma_val=sigma,
        n_samples=25,
    )

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    cbar = fig.colorbar(sm, ax=axs["B"], location="right")
    cbar.set_ticks(ticks=[0, 0.5, 1], labels=[0, 50 // 2, 50])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("Stagger distance", rotation=270)

    plt.show()


if __name__ == "__main__":
    main()
