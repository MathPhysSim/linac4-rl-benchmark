"""Train the new NAF agent (with PER support) on the Linac4 environment.

Uses the refactored NAF implementation from ``NAF_new`` which supports
Prioritized Experience Replay (PER) via a configurable ``prio_info`` dict.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from NAF_new.src.naf import NAF
from NAF_new.src.statistic import Statistic
from plotting import plot_convergence, plot_results
from simple_environment_linac4 import simpleEnv

RANDOM_SEED = 888
CHECKPOINT_DIR = "checkpoints/awake_test_1/"

tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

env = simpleEnv()
env.seed(RANDOM_SEED)
for _ in range(10):
    env.reset()


def main(_):
    discount = 0.999
    batch_size = 10
    learning_rate = 1e-3
    max_steps = 500
    update_repeat = 7
    max_episodes = 100
    tau = 1 - 0.999

    nafnet_kwargs = dict(
        hidden_sizes=[16, 16],
        activation=tf.nn.tanh,
        weight_init=tf.random_uniform_initializer(-0.05, 0.05),
    )
    prio_info = dict(alpha=0.75)

    with tf.Session() as sess:
        stat = Statistic(
            sess=sess, env_name=env.__name__,
            model_dir=CHECKPOINT_DIR, max_update_per_step=update_repeat,
        )
        agent = NAF(
            sess=sess, env=env, stat=stat,
            discount=discount, batch_size=batch_size,
            learning_rate=learning_rate, max_steps=max_steps,
            update_repeat=update_repeat, max_episodes=max_episodes,
            tau=tau, prio_info=prio_info, **nafnet_kwargs,
        )
        agent.run(is_train=True)

    label = "New NAF on: " + env.__name__
    plot_convergence(agent=agent, label=label)
    plot_results(env, label)


if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    else:
        for f in os.listdir(CHECKPOINT_DIR):
            print("Deleting: ", CHECKPOINT_DIR + "/" + f)
            os.remove(CHECKPOINT_DIR + "/" + f)
        time.sleep(3)
    tf.app.run()
    plt.show()
