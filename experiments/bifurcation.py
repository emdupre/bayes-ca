import click
import jax.numpy as jnp
from jax import jit, vmap
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


@jit
def sample_mu0(
    gap, sigmasq_subj, x0, num_timesteps, num_features, mu_pri, sigmasq_pri, hazard_rates
):
    """ """
    means = stagger_data(gap, num_timesteps, num_features)
    results = pgd_jaxopt(x0, means, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates)
    return results


def plot_mu0s(
    x0_strategy,
    mu_pri,
    sigmasq_pri,
    num_timesteps,
    num_features,
    hazard_rates,
    max_gap,
    sigma_val,
    n_samples,
):
    """ """
    gaps = jnp.linspace(0, max_gap, n_samples)

    mu0s = []
    means = vmap(stagger_data, in_axes=(0, None, None))(gaps, num_timesteps, num_features)

    for m in means:

        if x0_strategy == "true":
            # the true changepoint
            x0 = jnp.concatenate(
                (
                    -1 * jnp.ones((num_timesteps // 2, num_features)),
                    jnp.ones((num_timesteps // 2, num_features)),
                )
            )
        elif x0_strategy == "average":
            x0 = jnp.average(m, axis=0)

        results = pgd(x0, m, mu_pri, sigmasq_pri**2, sigma_val**2, hazard_rates)
        mu0s.append(results.x)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_title(f"sampled $\mu_0$ at $\sigma_{{subj}}$ = {sigma_val}")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    colors = plt.cm.viridis(jnp.linspace(0, 1, n_samples))
    for i, mu0 in enumerate(mu0s):
        p = ax.plot(mu0, c=colors[i], alpha=0.8, label=f"sampled $\mu_0$, {gaps[i]} stagger")

    cbar = fig.colorbar(sm, ax=ax, location="right")
    cbar.set_ticks(ticks=[0, 0.5, 1], labels=[0, 50 // 2, 50])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("stagger distance", rotation=270)

    return fig


def plot_param_sweep(
    x0_strategy, mu_pri, num_timesteps, num_features, hazard_rates, max_gap, max_sigma, n_samples
):
    """ """
    x0_strategy = "average"
    max_sigmasq = 9.0
    max_gap = 50
    n_samples = 50
    gaps = jnp.linspace(0, max_gap, n_samples)
    sigmasqs = jnp.linspace(0.01, max_sigmasq, n_samples)
    sigmas = [jnp.sqrt(s) for s in sigmasqs]

    mu0s = vmap(sample_mu0, in_axes=(0, 0, None, None, None, None, None, None))(
        gaps, sigmasqs, x0, num_timesteps, num_features, mu_pri, 1.0**2, hazard_rates
    )

    mu0s = []
    # muns = []
    for sigmasq in sigmasqs:
        means = vmap(stagger_data, in_axes=(0, None, None))(gaps, num_timesteps, num_features)
        # muns.append(means)
        for m in means:
            if x0_strategy == "true":
                # the true changepoint
                x0 = jnp.concatenate(
                    (
                        -1 * jnp.ones((num_timesteps // 2, num_features)),
                        jnp.ones((num_timesteps // 2, num_features)),
                    )
                )
            elif x0_strategy == "average":
                x0 = jnp.average(m, axis=0)

            results = pgd(x0, m, mu_pri, 1.0**2, sigmasq, hazard_rates)
            mu0s.append(results.x)

    # muns = jnp.vstack(muns)
    mu0s = jnp.asarray(mu0s)
    split_changepoints = jnp.full(n_samples**2, False)

    for i, mu0 in enumerate(mu0s):
        _, counts = jnp.unique(mu0, return_counts=True)
        if len(counts) > 2:
            split_changepoints = split_changepoints.at[i].set(True)

    change = jnp.repeat(jnp.inf, n_samples)
    reshape_bin = jnp.reshape(split_changepoints, (n_samples, n_samples))
    for i, r in enumerate(reshape_bin):
        try:
            change = change.at[i].set(jnp.where(jnp.diff(r, axis=0))[0][0])
        except IndexError:  # we never see a switch
            pass

    # NOTE : This works but I'm not sure why....
    split = [gaps[c.astype(int)] for c in change if c is not jnp.inf]

    hazard_prob = hazard_rates[0]
    b = -jnp.log(hazard_prob / (1 - hazard_prob))

    fig, ax = plt.subplots()
    ax.plot(sigmas, split)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.set_xlabel("$\sigma_{{subj}}$ value", labelpad=10)
    ax.set_ylabel("Stagger distance", labelpad=15)
    ax.spines[["left", "top"]].set_visible(False)
    ax.set_title(f"Transition from 1 to 2 changepoints")

    # # define the colors
    # cmap = mpl.colors.ListedColormap(["w", "k"])
    # # create a normalize object the describes the limits of
    # # each color
    # bounds = [0.0, 0.5, 1.0]
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # ax.imshow(
    #     jnp.reshape(split_changepoints, (n_samples, n_samples)),
    #     interpolation="none",
    #     cmap=cmap,
    #     norm=norm,
    # )

    return fig


@click.command()
@click.option("--mu_pri", default=0.0, help="")
@click.option("--sigmasq", default=2.0, help="")
@click.option("--hazard_prob", default=0.01, help="")
@click.option("--num_features", default=1, help="")
@click.option("--num_timesteps", default=300, help="")
@click.option("--x0_strategy", default="average", help="")
def main(mu_pri, sigmasq, hazard_prob, num_features, num_timesteps, x0_strategy):
    """ """
    # hardcoded params
    sigmasq_pri = 1.0
    max_duration = num_timesteps

    hazard_rates = hazard_prob * jnp.ones(max_duration)
    hazard_rates = hazard_rates.at[-1].set(1.0)

    fig1 = plot_mu0s(
        x0_strategy,
        mu_pri,
        sigmasq_pri,
        num_timesteps,
        num_features,
        hazard_rates,
        max_gap=50,
        sigma_val=sigmasq,
        n_samples=25,
    )

    # fig2 = plot_param_sweep(
    #     x0_strategy,
    #     mu_pri,
    #     num_timesteps,
    #     num_features,
    #     hazard_rates,
    #     max_gap=50,
    #     max_sigma=3.0,
    #     n_samples=50,
    # )

    plt.show()


if __name__ == "__main__":
    main()
