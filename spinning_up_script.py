"""Train a DDPG agent on the Linac4 environment using OpenAI SpinningUp.

Demonstrates the SpinningUp DDPG baseline on the Linac4 beam-steering
environment.  Results are logged to ``logging/awake/NAF/`` and plotted
after training.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from plotting import plot_results
from simple_environment_linac4 import simpleEnv
from spinup import ddpg

RANDOM_SEED = 888

tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

env = simpleEnv()
env.seed(RANDOM_SEED)
env.reset()
env_fn = lambda: env

OUTPUT_DIR = "logging/awake/NAF/"

ac_kwargs = dict()
logger_kwargs = dict(output_dir=OUTPUT_DIR, exp_name="transport_awake")

agent = ddpg(
    env_fn=env_fn, epochs=10, steps_per_epoch=100,
    ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs,
    start_steps=int(1e6), seed=RANDOM_SEED,
)

data = pd.read_csv(OUTPUT_DIR + "/progress.txt", sep="\t")
data.index = data["TotalEnvInteracts"]
data_plot = data[["EpLen", "MinEpRet", "AverageEpRet"]]
data_plot.plot(secondary_y=["MinEpRet", "AverageEpRet"])

label = "Classic DDPG on: " + env.__name__
plt.title(label=label)
plt.ylim(-10, 0)
plt.show()

plot_results(env, label)