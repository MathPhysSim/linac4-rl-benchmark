"""Benchmark classical optimisation algorithms on the Linac4 environment.

Compares derivative-free optimisers (COBYLA, Powell, Bayesian Optimisation)
for beam-steering correction as non-RL baselines.
"""

from bayes_opt import BayesianOptimization

from simple_environment_linac4 import simpleEnv
import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


class EnvironmentWrapper:
    """Wrapper around ``simpleEnv`` for use with SciPy optimisers.

    Provides an ``objective`` method compatible with ``scipy.optimize``
    interfaces, and records the full action/reward history for plotting.
    """

    def __init__(self, **kwargs):
        self.env = simpleEnv(**kwargs)
        self.env.reset()
        self.action_space_dimensions = self.env.action_space.shape[0]
        self.action_bounds_high = self.env.action_space.high
        self.action_bounds_low = self.env.action_space.low
        self.action_names = range(self.env.action_space.shape[0])
        self.history = pd.DataFrame(columns=self.action_names)
        self.counter = -1

    def set_history(self, action, r, s):
        self.counter += 1
        size = self.history.shape[0]
        self.history.loc[size, self.action_names] = action
        self.history.loc[size, "objective"] = r

    def plot_optimization(self):
        self.ax1.set_title("Actors")
        self.ax2.set_title("Objective")
        self.history.loc[:, self.action_names].plot(ax=self.ax1, legend=False)
        self.history.loc[:, "objective"].plot(ax=self.ax2)

    def objective(self, action):
        """Evaluate the environment and return the negative reward (for minimisation)."""
        s, r, d, _ = self.env.step(action=action)
        self.set_history(action, r=r, s=s)
        return -r


ALGORITHM_LIST = ["BAYESIAN", "BOBYQA", "Powell", "COBYLA"]


if __name__ == "__main__":
    algorithm_name = ALGORITHM_LIST[-1]
    print("Starting the algorithm:", algorithm_name)

    environment_instance = EnvironmentWrapper()
    start_vector = np.zeros(environment_instance.action_space_dimensions)

    if algorithm_name == "COBYLA":
        def constr(action):
            if any(action > environment_instance.action_bounds_high):
                return -1
            elif any(action < environment_instance.action_bounds_low):
                return -1
            else:
                return 1

        rhobeg = 0.5 * environment_instance.action_bounds_high[0]
        solution = opt.fmin_cobyla(
            environment_instance.objective, start_vector,
            [constr], rhobeg=rhobeg, rhoend=0.01,
        )

    elif algorithm_name == "Powell":
        solution = opt.fmin_powell(
            environment_instance.objective, start_vector,
            ftol=0.1, xtol=0.1,
            direc=0.5 * environment_instance.action_bounds_high[0],
        )

    print(environment_instance.env.step(solution))
    environment_instance.fig = plt.figure()
    environment_instance.ax1 = environment_instance.fig.add_subplot(211)
    environment_instance.ax2 = environment_instance.fig.add_subplot(
        212, sharex=environment_instance.ax1,
    )
    environment_instance.plot_optimization()
    plt.show()
