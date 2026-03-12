"""Train the original NAF agent on the Linac4 environment.

Uses the NAF implementation from ``NAF_old`` (carpedm20 fork) with
hard-coded hyperparameters and no exploration strategy.
"""

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from NAF_old.src.naf import NAF
from NAF_old.src.network import Network
from NAF_old.src.statistic import Statistic
from NAF_old.utils_old import get_model_dir, preprocess_conf
from plotting import plot_convergence, plot_results
from simple_environment_linac4 import simpleEnv

flags = tf.app.flags
conf = flags.FLAGS

logger = logging.getLogger()
logger.propagate = False
logger.setLevel("INFO")

RANDOM_SEED = 888

tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

env = simpleEnv()
env.seed(RANDOM_SEED)


def main(_):
    model_dir = get_model_dir(
        conf, ["is_train", "random_seed", "monitor", "display", "log_level"]
    )
    preprocess_conf(conf)

    with tf.Session() as sess:
        shared_args = {
            "sess": sess,
            "input_shape": env.observation_space.shape,
            "action_size": env.action_space.shape[0],
            "hidden_dims": [16, 16],
            "use_batch_norm": False,
            "use_seperate_networks": False,
            "hidden_w": tf.random_uniform_initializer(-0.05, 0.05),
            "action_w": tf.random_uniform_initializer(-0.05, 0.05),
            "hidden_fn": tf.tanh,
            "action_fn": tf.tanh,
            "w_reg": None,
        }

        logger.info("Creating prediction network...")
        pred_network = Network(scope="pred_network", **shared_args)

        logger.info("Creating target network...")
        target_network = Network(scope="target_network", **shared_args)

        discount = 0.999
        batch_size = 10
        learning_rate = 1e-3
        max_steps = 500
        update_repeat = 7
        max_episodes = 700
        tau = 1 - 0.999

        target_network.make_soft_update_from(pred_network, tau)

        stat = Statistic(
            sess, "default", model_dir, pred_network.variables, update_repeat
        )
        agent = NAF(
            sess, env, None, pred_network, target_network, stat,
            discount, batch_size, learning_rate,
            max_steps, update_repeat, max_episodes,
        )
        agent.run(monitor=False, display=False, is_train=True)

        label = "Orig. NAF on: " + env.__name__
        plot_convergence(agent=agent, label=label)
        plot_results(env, label)


if __name__ == "__main__":
    directory = "checkpoints/awake_test_1/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for f in os.listdir(directory):
            print("Deleting: ", directory + "/" + f)
            os.remove(directory + "/" + f)
        time.sleep(3)
    tf.app.run()
    plt.show()
