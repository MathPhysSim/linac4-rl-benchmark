"""Shared plotting utilities for Linac4 RL benchmark experiments.

Provides reusable functions for visualizing training results (reward
convergence, episode lengths) and agent diagnostics (TD-loss, value
function estimates) across all algorithm scripts.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_results(env, label):
    """Plot training performance: episode iterations and final rewards.

    Generates a two-panel figure showing:
    - Top: number of steps per episode over training.
    - Bottom: final reward per episode vs. initial state RMS.

    Also produces a scatter plot of initial-state RMS vs. final reward.

    Parameters
    ----------
    env : simpleEnv
        The environment instance (must have ``rewards``,
        ``initial_conditions``, and ``TOTAL_COUNTER`` attributes).
    label : str
        Title / label used in the figure and the saved PDF filename.
    """
    rewards = env.rewards
    initial_states = env.initial_conditions

    iterations = []
    finals = []
    starts = []

    for i in range(len(rewards)):
        if len(rewards[i]) > 0:
            finals.append(rewards[i][-1])
            starts.append(-np.sqrt(np.mean(np.power(initial_states[i], 2))))
            iterations.append(len(rewards[i]))

    plot_suffix = (
        f", number of iterations: {env.TOTAL_COUNTER}, "
        f"Linac4 time: {env.TOTAL_COUNTER / 600:.1f} h"
    )

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    axs[0].plot(iterations)
    axs[0].set_title("Iterations" + plot_suffix)
    fig.suptitle(label, fontsize=12)

    axs[1].plot(finals, "r--")
    axs[1].plot(starts, c="lime")
    axs[1].set_title("Final reward per episode")
    axs[1].set_xlabel("Episodes (1)")
    plt.savefig(label + ".pdf")
    plt.show()

    plt.figure()
    plt.scatter(
        -np.array(starts),
        -np.array(finals),
        c="g",
        alpha=0.5,
        marker=r"$\clubsuit$",
        label="Luck",
    )
    plt.ylim(0, 3)
    plt.title(label)
    plt.show()


def plot_convergence(agent, label):
    """Plot TD-loss and value-function estimates over training.

    Generates a dual-axis figure with:
    - Left axis (blue): TD loss per episode.
    - Right axis (green): mean value-function estimate per episode.

    Parameters
    ----------
    agent : NAF
        Trained agent (must have ``losses`` and ``vs`` attributes).
    label : str
        Title / label used in the figure and the saved PDF filename.
    """
    losses, vs = agent.losses, agent.vs
    fig, ax = plt.subplots()
    ax.set_title(label)
    ax.set_xlabel("Episodes")

    color = "tab:blue"
    ax.plot(losses, color=color)
    ax.tick_params(axis="y", labelcolor=color)
    ax.set_ylabel("TD loss", color=color)
    ax.set_ylim(0, 1)

    ax1 = plt.twinx(ax)
    ax1.set_ylim(-2, 1)
    color = "lime"
    ax1.set_ylabel("V", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.plot(vs, color=color)
    plt.savefig(label + "convergence.pdf")
    plt.show()
