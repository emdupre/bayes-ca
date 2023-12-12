import jax.numpy as jnp
import jax.random as jr
from jax.lax import conv

from bayes_ca.inference import (
    _compute_gaussian_stats,
    _compute_gaussian_lls,
    cp_filter,
    cp_backward_filter,
    cp_smoother,
    gaussian_cp_smoother,
    sample_gaussian_cp_model,
)

# Test
key = jr.PRNGKey(0)
num_timesteps = 1000
K = 100  # max run length
lmbda = 0.1  # prob changepoint
mu0 = 0.0  # prior mean
sigmasq0 = 3**2  # prior variance
sigmasq = 0.5**2  # observation variance

# Compute hazard rates under a geometric duration distribution
# Note: we can generalize this for other changepoint distributions
hazard_rates = lmbda * jnp.ones(K + 1)
hazard_rates = hazard_rates.at[-1].set(1.0)

# Sample the prior
this_key, key = jr.split(key)
zs, mus = sample_gaussian_cp_model(this_key, num_timesteps, hazard_rates, mu0, sigmasq0)

# Sample noisy observations
this_key, key = jr.split(key)
xs = mus + jnp.sqrt(sigmasq) * jr.normal(this_key, mus.shape)
partial_sums, partial_counts = _compute_gaussian_stats(xs, K)
lls = _compute_gaussian_lls(xs, partial_sums, partial_counts, mu0, sigmasq0, sigmasq)
_, _, transition_probs = cp_smoother(hazard_rates, lls)


def test_log_normalizers():
    """ """
    forward_normalizer, _, _ = cp_filter(hazard_rates, lls)
    backward_normalizer, _ = cp_backward_filter(hazard_rates, lls)
    assert jnp.isclose(forward_normalizer, backward_normalizer, atol=1.0)


def test_kernel_conv():
    """ """
    _, _, transition_probs = cp_smoother(hazard_rates, lls)
    kernel = jnp.tril(jnp.ones((K + 1, K + 1)))[None, None, :, :]  # OIKK
    assert jnp.allclose(
        conv(
            transition_probs.T[None, None, :, :],  # NCKT
            kernel,  # 11KK
            (1, 1),  # window strides
            [(0, 0), (0, K)],  # padding
        )[
            0, 0, 0
        ],  # T output
        1.0,
    )


def test_posterior_means():
    """ """
    sigmasq_post = 1 / (1 / sigmasq0 + partial_counts / sigmasq)
    mu_post = sigmasq_post * (mu0 / sigmasq0 + partial_sums / sigmasq)
    _, _, _, posterior_means = gaussian_cp_smoother(xs, hazard_rates, mu0, sigmasq0, sigmasq)

    # Test against naive implementation
    t = 60
    foo = 0.0
    for k in range(K + 1):
        for i in range(min(k + 1, num_timesteps - 1)):
            foo += transition_probs[t + i, k] * mu_post[t + i, k]
    assert jnp.allclose(posterior_means[t], foo)
