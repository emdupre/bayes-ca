from itertools import product

import click
from jax import vmap
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

from bayes_ca.prox_grad import pgd, pgd_jaxopt

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

    return subj_means


def sample_mu0(gap, x0, params):
    """ """
    (num_timesteps, num_features, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates) = params
    means, _ = stagger_data(gap, num_timesteps, num_features)
    results = pgd_jaxopt(x0, means, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates)
    return results


def plot_mu0s(
    x0, mu_pri, num_timesteps, num_features, hazard_rates, max_gap, sigma_val, n_samples
):
    """ """
    gaps = jnp.linspace(0, max_gap, n_samples)

    mu0s = []
    means = vmap(stagger_data, in_axes=(0, None, None))(gaps, num_timesteps, num_features)
    for m in means:
        results = pgd(x0, m, mu_pri, sigma_val**2, sigma_val**2, hazard_rates)
        mu0s.append(results.x)

    fig = plt.figure()
    ax = plt.subplot(111)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    colors = plt.cm.viridis(jnp.linspace(0, 1, n_samples))
    for i, mu0 in enumerate(mu0s):
        p = ax.plot(mu0, c=colors[i], alpha=0.8, label=f"sampled $\mu_0$, {gaps[i]} stagger")
    cbar = fig.colorbar(sm, ax=ax, location="right")
    cbar.set_ticks(ticks=[0, 0.5, 1], labels=[0, 50 // 2, 50])

    return fig


def plot_param_sweep(
    x0, mu_pri, num_timesteps, num_features, hazard_rates, max_gap, max_sigma, n_samples
):
    """ """
    n_samples = 50
    gaps = jnp.linspace(0, max_gap, n_samples)
    sigmas = jnp.linspace(0.01, max_sigma, n_samples)

    mu0s = []
    muns = []
    for sigma in sigmas:
        means = vmap(stagger_data, in_axes=(0, None, None))(gaps, num_timesteps, num_features)
        muns.append(means)
        for m in means:
            results = pgd(x0, m, mu_pri, sigma**2, sigma**2, hazard_rates)
            mu0s.append(results.x)

    muns = jnp.vstack(muns)
    mu0s = jnp.asarray(mu0s)
    params = product(sigmas, gaps)

    split_changepoints = jnp.full(n_samples**2, False)
    for i, mu0 in enumerate(mu0s):
        _, counts = jnp.unique(mu0, return_counts=True)
        if len(counts) != 2:
            split_changepoints = split_changepoints.at[i].set(True)
            if len(counts) != 3:
                print("weird ! ", len(counts))
                print(list(params)[i], jnp.unique(mu0))

    change = jnp.repeat(jnp.inf, n_samples)
    reshape_bin = jnp.reshape(split_changepoints, (n_samples, n_samples))
    for i, r in enumerate(reshape_bin):
        try:
            change = change.at[i].set(jnp.where(jnp.diff(r, axis=0))[0][0])
        except IndexError:  # we never see a switch
            pass

    # NOTE : This works but I'm not sure why....
    [gaps[c.astype(int)] for c in change if c is not jnp.inf]

    fig, ax = plt.subplots()
    # define the colors
    cmap = mpl.colors.ListedColormap(["w", "k"])
    # create a normalize object the describes the limits of
    # each color
    bounds = [0.0, 0.5, 1.0]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(
        jnp.reshape(split_changepoints, (n_samples, n_samples)),
        interpolation="none",
        cmap=cmap,
        norm=norm,
    )
    ax.set_yticks(jnp.arange(n_samples)[::2], jnp.around(sigmas, 2)[::2])
    ax.set_xticks(jnp.arange(n_samples)[::2], jnp.around(gaps, 0).astype(int)[::2])
    ax.set_ylabel("Sigma values")
    ax.set_xlabel("Stagger distance")
    return fig


@click.command()
@click.option("--mu_pri", default=0.0, help="")
@click.option("--sigmasq", default=2.0, help="")
@click.option("--hazard_prob", default=0.01, help="")
@click.option("--num_features", default=1, help="")
@click.option("--num_timesteps", default=300, help="")
def main(mu_pri, sigmasq, hazard_prob, num_features, num_timesteps):
    """ """
    # temporal params
    max_duration = num_timesteps
    hazard_rates = hazard_prob * jnp.ones(max_duration)
    hazard_rates = hazard_rates.at[-1].set(1.0)

    # the true changepoint
    x0 = jnp.concatenate(
        (
            -1 * jnp.ones((num_timesteps // 2, num_features)),
            jnp.ones((num_timesteps // 2, num_features)),
        )
    )

    fig1 = plot_mu0s(
        x0,
        mu_pri,
        num_timesteps,
        num_features,
        hazard_rates,
        max_gap=50,
        sigma_val=sigmasq,
        n_samples=25,
    )

    fig2 = plot_param_sweep(
        x0,
        mu_pri,
        num_timesteps,
        num_features,
        hazard_rates,
        max_gap=50,
        max_sigma=3.0,
        n_samples=50,
    )

    plt.show()


if __name__ == "__main__":
    main()
