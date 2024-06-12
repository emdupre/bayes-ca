import click
import jax.random as jr
import jax.numpy as jnp
from jax import jit, vmap
import matplotlib.pyplot as plt
from fastprogress import progress_bar
from matplotlib.transforms import ScaledTranslation
from tensorflow_probability.substrates import jax as tfp

import bayes_ca.inference as core
from bayes_ca.prox_grad import pgd_jaxopt

tfd = tfp.distributions


def gibbs_sample_subject_means(
    key, subj_obs, sigmasq_obs, global_means, sigmasq_subj, mu_pri, sigmasq_pri, hazard_rates
):
    """ """
    num_subjects = subj_obs.shape[0]
    effective_sigmasq = 1 / (1 / sigmasq_obs + 1 / sigmasq_subj)
    effective_emissions = effective_sigmasq * (
        subj_obs / sigmasq_obs + global_means / sigmasq_subj
    )

    _sample_one = lambda key, y: core.gaussian_cp_posterior_sample(
        key, y, hazard_rates, mu_pri, sigmasq_pri, effective_sigmasq
    )[1]
    return vmap(_sample_one)(jr.split(key, num_subjects), effective_emissions)


@jit
def step(
    key, subj_obs, sigmasq_obs, global_means, sigmasq_subj, mu_pri, sigmasq_pri, hazard_rates
):
    """ """
    # Sample new subject means
    subj_means = gibbs_sample_subject_means(
        key, subj_obs, sigmasq_obs, global_means, sigmasq_subj, mu_pri, sigmasq_pri, hazard_rates
    )
    result = pgd_jaxopt(  # 55m
        global_means, subj_means, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates
    )
    global_means = result.params

    joint_lp = core.joint_lp(
        global_means,
        subj_means,
        subj_obs,
        mu_pri,
        sigmasq_pri,
        sigmasq_subj,
        sigmasq_obs,
        hazard_rates,
    )

    return global_means, subj_means, joint_lp


def stagger_data(key, sigmasq_obs, min_val=-0.40, mid_val=0.30, max_val=0.80):
    """
    A three subject model with hardcoded number of time steps and features.
    """
    this_key, key = jr.split(key)

    signal_one = jnp.concatenate(
        (
            jnp.ones((100, 1)) * min_val,
            jnp.ones((100, 1)) * mid_val,
            jnp.ones((100, 1)) * max_val,
        )
    )
    obs_one = tfd.Normal(signal_one, jnp.sqrt(sigmasq_obs)).sample(seed=this_key)

    signal_two = jnp.concatenate(
        (
            jnp.ones((100, 1)) * min_val,
            jnp.ones((140, 1)) * mid_val,
            jnp.ones((60, 1)) * max_val,
        )
    )
    obs_two = tfd.Normal(signal_two, jnp.sqrt(sigmasq_obs)).sample(seed=key)

    this_key, key = jr.split(key)
    signal_three = jnp.concatenate(
        (
            jnp.ones((240, 1)) * (min_val + mid_val) / 2,
            jnp.ones((60, 1)) * max_val,
        )
    )
    obs_three = tfd.Normal(signal_three, jnp.sqrt(sigmasq_obs)).sample(seed=key)

    return jnp.stack((signal_one, signal_two, signal_three)), jnp.stack(
        (obs_one, obs_two, obs_three)
    )


@click.command()
@click.option("--seed", default=0, help="")
@click.option("--mu_pri", default=0.0, help="")
@click.option("--sigma_pri", default=1.0, help="")
@click.option("--sigma_subj", default=1.0, help="")
@click.option("--sigma_obs", default=0.25, help="")
@click.option("--num_timesteps", default=300, help="")
@click.option("--hazard_prob", default=0.01, help="")
def main(seed, mu_pri, sigma_pri, sigma_subj, sigma_obs, num_timesteps, hazard_prob):
    """ """
    # hardcoded params
    max_duration = num_timesteps
    hazard_rates = hazard_prob * jnp.ones(max_duration)
    hazard_rates = hazard_rates.at[-1].set(1.0)

    key = jr.PRNGKey(seed=seed)
    signals, obs = stagger_data(key, sigma_obs**2)
    x0 = jnp.mean(obs, axis=0)

    lps = []

    for _ in progress_bar(range(7500)):  # approx 6h run time
        this_key, key = jr.split(key)
        global_means, subj_means, train_lp = step(
            this_key,
            obs,
            sigmasq_obs=0.25**2,
            global_means=x0,
            sigmasq_subj=2.0**2,
            mu_pri=0.0,
            sigmasq_pri=1.0**2,
            hazard_rates=hazard_rates,
        )
        lps.append(train_lp)

    _, _, transition_probs, _ = core.gaussian_cp_smoother(
        global_means, hazard_rates, mu_pri, sigma_pri**2, sigma_subj**2
    )

    fig, axs = plt.subplot_mosaic(
        [["A", "B", "C", "D"], ["E", "F", "G", "H"]],
        sharex=True,
        sharey=True,
        layout="constrained",
        dpi=300,
        figsize=(4.5, 3),
    )
    fig.supxlabel("Time")

    for _, ax in axs.items():
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_ylim(-1, 1.25)
        ax.set_yticks((-1, -0.5, 0, 0.5, 1))

    for i, (ax, c, title) in enumerate(
        zip(
            [axs["A"], axs["B"], axs["C"]],
            ["#59B3A9", "#4298B5", "#007C92"],
            ["$\mu^{n_1}$", "$\mu^{n_2}$", "$\mu^{n_3}$"],
        )
    ):
        ax.plot(signals[i], c=c, alpha=0.5)
        ax.plot(obs[i], ".", color=c, alpha=0.9, markersize=0.9)
        ax.set_title(title, size="x-large")

    axs["A"].text(
        0.0,
        0.5,
        "True",
        transform=(
            axs["A"].transAxes + ScaledTranslation(-50 / 72, +2 / 72, fig.dpi_scale_trans)
        ),
        size="large",
        va="center",
        rotation=90,
    )
    axs["D"].plot(jnp.average(signals, axis=0), ls=(0, (5, 1)), c="#7F7776")
    axs["D"].set_title("$\mu^0$", size="x-large")

    for i, (ax, c) in enumerate(
        zip([axs["E"], axs["F"], axs["G"]], ["#017E7C", "#016895", "#006B81"])
    ):
        ax.plot(subj_means[i], c=c)

    axs["E"].text(
        0.0,
        0.5,
        "Sampled",
        transform=(
            axs["E"].transAxes + ScaledTranslation(-50 / 72, +2 / 72, fig.dpi_scale_trans)
        ),
        size="large",
        va="center",
        rotation=90,
    )
    axs["H"].plot(global_means, c="#544948")
    plt.savefig("sampled_means.png", format="png", transparent=True)

    fig, ax = plt.subplots(layout="constrained", dpi=300, figsize=(3.5, 3))
    ax.spines[["left", "top"]].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("Time", size="large")
    ax.imshow(jnp.log(transition_probs.T), aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("Transition probabilities", size="large")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    cbar = fig.colorbar(sm, ax=ax, location="right")
    cbar.set_ticks(ticks=[0, 0.5, 1], labels=[0.0, 0.5, 1.0])
    cbar.ax.get_yaxis().labelpad = 15
    plt.savefig("Transition_probabilities.png", format="png", transparent=True)

    plt.show()

    return
