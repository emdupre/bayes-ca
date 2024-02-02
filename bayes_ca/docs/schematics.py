import itertools

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


# set hyper params
key = jr.PRNGKey(0)
sigmasq_obs = 0.15**2

# set underlying means and generate observations for first subplot
true_means = jnp.concatenate(
    (
        jnp.ones((5, 1)) * -0.40,
        jnp.ones((7, 1)) * 0.30,
        jnp.ones((3, 1)) * 0.80,
    )
)
obs = tfd.Normal(true_means, jnp.sqrt(sigmasq_obs)).sample(seed=key)

# generate line segments for time, run lengths in second subplot
time = jnp.arange(1, 16)
run_lengths = jnp.concatenate([jnp.arange(5), jnp.arange(7), jnp.arange(3)])
rl_coords = [(x, y) for x, y in zip(time, run_lengths)]

# figure generation ; set sizing and x-, y-lims
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(7, 5), sharex=True)
plt.setp(ax1, xlim=(0.75, 15.5), ylim=(-1.0, 1.0))
plt.setp(ax2, ylim=(-0.1, 6.5))

# set first sub-plot
ax1.plot(jnp.arange(1, 16), obs, "o", alpha=1, color="dimgray")
lc = mc.LineCollection(
    [((1, -0.4), (5, -0.4)), ((6, 0.3), (12, 0.3)), ((13, 0.8), (15, 0.8))],
    color="black",
    linestyle="dotted",
)
ax1.add_collection(lc)
ax1.axvline(x=5.5, color="black", linestyle=(5, (10, 3)), linewidth=0.75)
ax1.axvline(x=12.5, color="black", linestyle=(5, (10, 3)), linewidth=0.75)
ax1.set_ylabel(ylabel="$y_t$", labelpad=5.0, fontsize="xx-large", rotation="horizontal")
ax1.yaxis.set_tick_params(labelleft=False)
ax1.set_yticks([])
ax1.yaxis.set_label_coords(-0.05, 0.85)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_position("zero")
ax1.plot(1, 0, ">k", transform=ax1.get_yaxis_transform(), clip_on=False)
ax1.plot(0.75, 1, "^k", transform=ax1.get_xaxis_transform(), clip_on=False)
ax1.plot(0.75, 0, "vk", transform=ax1.get_xaxis_transform(), clip_on=False)


# set second sub-plot
linestyle = [
    "solid",
    "solid",
    "solid",
    "solid",
    (5, (10, 3)),
    "solid",
    "solid",
    "solid",
    "solid",
    "solid",
    "solid",
    (5, (10, 3)),
    "solid",
    "solid",
]
lc = mc.LineCollection(
    itertools.pairwise(rl_coords), color="black", linestyle=linestyle, linewidth=0.75
)
ax2.set_ylabel(ylabel="$z_t$", labelpad=5.0, fontsize="xx-large", rotation="horizontal")
ax2.set_xlabel(xlabel="$t$", fontsize="xx-large", loc="right")
ax2.add_collection(lc)
ax2.scatter(time, run_lengths, color="black", marker=".")
ax2.yaxis.set_label_coords(-0.05, 0.85)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.plot(1, -0.1, ">k", transform=ax2.get_yaxis_transform(), clip_on=False)
ax2.plot(0.75, 1, "^k", transform=ax2.get_xaxis_transform(), clip_on=False)

# render and show
plt.tight_layout()
plt.show()
