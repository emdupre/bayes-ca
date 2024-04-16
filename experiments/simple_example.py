import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from jax import vmap
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


sigma = 1.0
mu_pri = 0.0
sigma_pri = 1.0
key = jr.PRNGKey(0)


def log_prior(mu0):
    return tfd.Normal(mu_pri, sigma_pri).log_prob(mu0)


def single_log_lkhd(mun, mu0):
    ll = jnp.log(
        0.5 * jnp.exp(tfd.Normal(-1, sigma_pri).log_prob(mun))
        + 0.5 * jnp.exp(tfd.Normal(+1, sigma_pri).log_prob(mun))
    )
    ll += tfd.Normal(mun, sigma).log_prob(mu0)

    ll -= jnp.log(
        0.5 * jnp.exp(tfd.Normal(-1, jnp.sqrt(sigma**2 + sigma_pri**2)).log_prob(mu0))
        + 0.5 * jnp.exp(tfd.Normal(+1, jnp.sqrt(sigma**2 + sigma_pri**2)).log_prob(mu0))
    )
    return ll


muns = jnp.array([-1.0, -1.0, -1, -1, -1])
mu0s = jnp.linspace(-2.0, 2.0, 10000)


def log_lkhd(mu0):
    return vmap(single_log_lkhd, in_axes=(0, None))(muns, mu0).sum()


def log_post(mu0):
    return log_prior(mu0) + log_lkhd(mu0)


lls = vmap(log_lkhd)(mu0s)
lps = vmap(log_post)(mu0s)
imax = jnp.argmax(lps)
mu0_star = mu0s[imax]

plt.figure()
plt.plot(mu0s, lps)
# plt.plot(mu0s, lls)
plt.xlabel("mu0")
plt.ylabel("p(mu0 | muns)")
plt.axvline(-1, color="k")
plt.axvline(0, color="k")
plt.axvline(+1, color="k")
plt.plot(mu0_star, lps[imax], "r*")

# plt.figure()
# plt.plot(mu0s, lls)
# plt.xlabel("mu0")
# plt.ylabel("p(muns | mu0)")
# plt.axvline(-1, color="k")
plt.show()
