import itertools

import jax.numpy as jnp
import jax.random as jr
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import collections as mc
import matplotlib.transforms as mtransforms
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
fig, axs = plt.subplot_mosaic([["a)"], ["b)"]], layout="constrained", sharex=True, figsize=(8, 6))

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

ax1 = axs["a)"]
ax2 = axs["b)"]
# fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(7, 5), sharex=True)
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
ax1.set_ylabel(ylabel="$y_t$", labelpad=5.0, fontsize="x-large", rotation="horizontal")
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
ax2.set_xticks(jnp.arange(1, 16, 2))
ax2.set_ylabel(ylabel="$z_t$", labelpad=5.0, fontsize="x-large", rotation="horizontal")
ax2.set_xlabel(xlabel="$t$", fontsize="x-large", loc="right")
ax2.add_collection(lc)
ax2.scatter(time, run_lengths, color="black", marker=".")
ax2.yaxis.set_label_coords(-0.05, 0.85)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.plot(1, -0.1, ">k", transform=ax2.get_yaxis_transform(), clip_on=False)
ax2.plot(0.75, 1, "^k", transform=ax2.get_xaxis_transform(), clip_on=False)

# render and show
# plt.tight_layout()
plt.show()


#######################################
#### SECOND SCHEMATIC FIGURE
#######################################

# set hyper params
key = jr.PRNGKey(111)
sigmasq_obs = 0.15**2

fig, axs = plt.subplot_mosaic(
    [["a)"], ["b)"], ["c)"]], layout="constrained", sharex=True, figsize=(7, 10)
)
# fig, axs = plt.subplots(nrows=3, figsize=(5, 8), sharex=True)
plt.setp(ax1, xlim=(0, 15.5), ylim=(-1.0, 1.0))
# fig.supylabel("$y_t$", fontsize="x-large", rotation="horizontal")
# fig.supxlabel("$t$", fontsize="x-large", x=0.9, y=0.075, ha="right")


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
    ax.set_xlim((0, 15.5))
    ax.set_ylim((-1.0, 1.0))
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.plot(0, 0, "vk", transform=ax.get_xaxis_transform(), clip_on=False)

ax1 = axs["a)"]
ax2 = axs["b)"]
ax3 = axs["c)"]

# set first sub-plot
signal_one = jnp.concatenate(
    (
        jnp.ones((5, 1)) * -0.40,
        jnp.ones((7, 1)) * 0.30,
        jnp.ones((3, 1)) * 0.80,
    )
)
obs_one = tfd.Normal(signal_one, jnp.sqrt(sigmasq_obs)).sample(seed=key)

ax1.plot(jnp.arange(1, 16), obs_one, "o", alpha=1, color="dimgray")
lc_one = mc.LineCollection(
    [((1, -0.4), (5, -0.4)), ((6, 0.3), (12, 0.3)), ((13, 0.8), (15, 0.8))],
    color="black",
    linestyle="dotted",
)
ax1.add_collection(lc_one)
ax1.axvline(x=5.5, color="black", linestyle=(5, (10, 3)), linewidth=0.75)
ax1.axvline(x=12.5, color="black", linestyle=(5, (10, 3)), linewidth=0.75)

# set second sub-plot
this_key, key = jr.split(key)
signal_two = jnp.concatenate(
    (
        jnp.ones((5, 1)) * -0.40,
        jnp.ones((5, 1)) * 0.30,
        jnp.ones((5, 1)) * 0.80,
    )
)
obs_two = tfd.Normal(signal_two, jnp.sqrt(sigmasq_obs)).sample(seed=this_key)

ax2.plot(jnp.arange(1, 16), obs_two, "o", alpha=1, color="dimgray")
lc_two = mc.LineCollection(
    [((1, -0.4), (5, -0.4)), ((6, 0.3), (10, 0.3)), ((11, 0.8), (15, 0.8))],
    color="black",
    linestyle="dotted",
)
ax2.add_collection(lc_two)
ax2.axvline(x=5.5, color="black", linestyle=(5, (10, 3)), linewidth=0.75)
ax2.axvline(x=10.5, color="black", linestyle=(5, (10, 3)), linewidth=0.75)
ax2.set_ylabel(ylabel="$y_t$", labelpad=10.0, fontsize="x-large", rotation="horizontal")
ax2.yaxis.set_label_coords(-0.05, 0.5)

# set third sub-plot
this_key, key = jr.split(key)
signal_three = jnp.concatenate(
    (
        jnp.ones((12, 1)) * -0.40,
        jnp.ones((3, 1)) * 0.80,
    )
)
obs_three = tfd.Normal(signal_three, jnp.sqrt(sigmasq_obs)).sample(seed=key)

ax3.plot(jnp.arange(1, 16), obs_three, "o", alpha=1, color="dimgray")
lc_three = mc.LineCollection(
    [((1, -0.4), (12, -0.4)), ((13, 0.8), (15, 0.8))],
    color="black",
    linestyle="dotted",
)
ax3.add_collection(lc_three)
ax3.axvline(x=12.5, color="black", linestyle=(5, (10, 3)), linewidth=0.75)
ax3.set_xticks(jnp.arange(1, 16, 2))
ax3.tick_params(axis="x", direction="in", pad=-15.0)
ax3.set_xlabel(xlabel="$t$", labelpad=10.0, fontsize="x-large", loc="right")
ax3.xaxis.set_label_coords(0.95, 0.0)

plt.show()


#######################################
#### THIRD SCHEMATIC FIGURE
#######################################


x = jnp.asarray([4, 4, 11])
y = jnp.asarray([0, 7, 7])

fig, ax = plt.subplots()
ax.set_xlim((0, 14.0))
ax.set_ylim((0, 7.0))
ax.set_aspect("equal", adjustable="box")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.scatter(x, y, color="black", marker="")
ax.fill(x, y, linewidth=0.5, edgecolor="black", facecolor="none")

ax.set_xticks(ticks=[4, 11], labels=["$t$", "$t + K$"])
ax.set_yticks(ticks=[0, 7], labels=[0, "$K$"])

ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.set_xlabel(xlabel="$t$", labelpad=10.0, fontsize="x-large", loc="right")

grad = jnp.atleast_2d(jnp.linspace(0, 1, 256)).T
img = ax.imshow(
    jnp.flip(grad),
    extent=[jnp.min(x), jnp.max(x), jnp.min(y), jnp.max(y)],
    interpolation="bicubic",
    # aspect="auto",
    cmap="gist_gray_r",
)
polygon = Polygon([*zip(x, y)], closed=True, facecolor="none", edgecolor="none")
ax.add_patch(polygon)
img.set_clip_path(polygon)

plt.show()
