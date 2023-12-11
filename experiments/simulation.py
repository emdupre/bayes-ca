import click
import jax.numpy as jnp
import jax.random as jr
from jax.lax import conv
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

from bayes_ca.inference import (
    sample_cp_prior,
    compute_pred_log_likes,
    compute_conditional_means,
    hmm_filter,
    hmm_smoother,
)

tfd = tfp.distributions


@click.command()
@click.option("--num_timesteps", default=1000, help="Number of time steps.")
@click.option("--max_length", "K", default=100, help="Max run length, K")
@click.option("--lmbda", default=0.1, help="Change point probability.")
@click.option("--prior_mean", "mu0", default=0.0, help="Prior mean, mu0")
@click.option("--prior_variance", "sigmasq0", default=3**2, help="Prior variance, sigmasq0")
@click.option(
    "--observation_variance", "sigmasq", default=3**2, help="Observation variance, sigmasq"
)
@click.option("--seed", default=0, help="Random seed.")
def main(
    num_timesteps,
    K,
    lmbda,
    mu0,
    sigmasq0,
    sigmasq,
    seed,
):
    # Generate simulated data
    key = jr.PRNGKey(seed)
    # K = max_length  # max run length
    # lmbda = 0.1  # prob changepoint
    # mu0 = 0.0  # prior mean
    # sigmasq0 = 3**2  # prior variance
    # sigmasq = 0.5**2  # observation variance

    # Compute hazard rates under a geometric duration distribution
    # Note: we can generalize this for other changepoint distributions
    hazard_rates = lmbda * jnp.ones(K + 1)
    hazard_rates = hazard_rates.at[-1].set(1.0)

    # Sample the prior
    this_key, key = jr.split(key)
    zs, mus = sample_cp_prior(this_key, num_timesteps, hazard_rates, mu0, sigmasq0)

    # Sample and visualize noisy observations
    this_key, key = jr.split(key)
    xs = mus + jnp.sqrt(sigmasq) * jr.normal(this_key, mus.shape)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axs[0].plot(zs)
    axs[0].set_ylabel(r"$z_t$")
    axs[1].plot(mus)
    axs[1].plot(xs, "r.", alpha=0.5)
    axs[1].set_ylabel(r"$\mu_t$")
    axs[1].set_xlim(0, num_timesteps)

    # Experiment 1: Compute log-likelihood of simulated data
    lls = compute_pred_log_likes(xs, K, mu0, sigmasq0, sigmasq)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(mus)
    im = axs[1].imshow(lls.T, aspect="auto")
    plt.colorbar(im)

    log_normalizer, filtered_probs, predicted_probs = hmm_filter(hazard_rates, lls)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(mus)
    im = axs[1].imshow(filtered_probs.T, aspect="auto", origin="lower", cmap="Greys")
    axs[1].set_xlim(800, 1000)

    # Experiment 2: Backward filtering
    log_normalizer, smoothed_probs, transition_probs = hmm_smoother(hazard_rates, lls)
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axs[0].plot(mus)
    axs[0].plot(xs, "r.", alpha=0.5)
    axs[1].imshow(smoothed_probs.T, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=1)
    axs[2].imshow(transition_probs.T, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=1)
    axs[2].set_xlim(800, 1000)

    # Experiment 3: Posterior means
    conditional_means = compute_conditional_means(xs, K)
    # Make a convolution kernel to compute the posterior marginal means
    kernel = jnp.tril(jnp.ones((K + 1, K + 1)))[None, None, :, :]  # OIKK
    inpt = (transition_probs * conditional_means).T  # KT
    inpt = inpt[None, None, :, :]  # NCKT
    posterior_means = conv(inpt, kernel, (1, 1), [(0, 0), (0, K)])[0, 0, 0]  # T

    plt.plot(mus, lw=3, label="true mu")
    plt.plot(xs, "r.", alpha=0.5)
    plt.plot(posterior_means, "k-", lw=1, label="posterior mean")
    plt.xlim(800, 1000)

    post_mode = hmm_posterior_mode(initial_dist, hazard_rates, lls)

    fig, axs = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    axs[0].plot(mus)
    axs[0].plot(xs, "r.", alpha=0.5)
    axs[0].grid(True)
    axs[1].plot(zs)
    axs[1].plot(post_mode, "r.", alpha=0.6)
    axs[1].set_xlim(400, 600)
    axs[1].grid(True)

    plt.show()


if __name__ == "__main__":
    main()
