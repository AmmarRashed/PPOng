import itertools

import tensorflow as tf
import numpy as np
from joblib import Parallel, delayed, cpu_count

MAX_EPS_LEN = 4096
TRAJECTORIES_PER_UPDATE = 16
MAX_STEPS_PER_TRAJECTORY = 1024
DISCOUNT_RATE = 0.99  # reward discount factor
UPDATE_STEP = 128  # loop update operation n-steps
EPSILON = 0.2  # for clipping surrogate objective
HIDDEN_UNITS = 128
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-4

HE_INIT = tf.contrib.layers.variance_scaling_initializer()  # variance works better with (R)eLU activation
XAVIER = tf.contrib.layers.xavier_initializer()  # xavier works better with sigmoid activation


# taken from Aurelion Geron implementation:
# https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb


def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]


class PPO(object):
    def __init__(self, env):
        self.env = env
        self.sess = tf.Session()
        self.state_space = env.observation_space
        self.action_space = env.action_space
        self.in_state = tf.placeholder(tf.float32, [None, self.state_space])

        # Actor
        self.pi, self.pi_params = self.build_actor_network("LiveActor", trainable=True)
        self.old_pi, self.old_pi_params = self.build_actor_network("FrozenActor", trainable=False)
        self.actions = tf.placeholder(tf.int32, [None, ], name="actor_action")
        self.advantage_placeholder = tf.placeholder(tf.float32, [None, ], name="actor_advantage")
        self.actor_loss = self.calculate_clipped_surrogate_loss()
        self.actor_train_op = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE).minimize(self.actor_loss)
        self.update_frozen_actor_op = [self.old_pi.assign(p) for p, prev_p in zip(self.pi_params, self.old_pi_params)]

        # Critic
        self.v, self.critic_dc_rew, self.advantage, self.critic_loss, self.critic_train_op \
            = self.build_critic_network()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def calculate_clipped_surrogate_loss(self):
        """
        :return: clipped surrogate loss
        """
        actions_indices = tf.stack([tf.range(tf.shape(self.actions)[0], dtype=tf.int32), self.actions], axis=1)
        # get probabilities of actions according to the corresponding policies
        pi_prob = tf.gather_nd(params=self.pi, indices=actions_indices)
        old_pi_prob = tf.gather_nd(params=self.old_pi, indices=actions_indices)
        # calculate the ratio between the new and old policies
        ratio = pi_prob / tf.maximum(old_pi_prob, 1e-12)
        surrogate_loss = tf.multiply(ratio, self.advantage_placeholder)
        return -tf.reduce_mean(
            tf.minimum(
                surrogate_loss,
                tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.advantage_placeholder
            )
        )

    def build_actor_network(self, name: str, trainable: bool):
        """
        :param name: The scope of the network
        :param trainable: boolean, used to freeze training of the old policy actor
        :return: the node of actions probabilities, and the parameters of the network
        """
        with tf.variable_scope(name):
            h = tf.layers.dense(self.in_state, HIDDEN_UNITS, activation=tf.nn.elu,
                                kernel_initializer=HE_INIT, name=f"{name}_H", trainable=trainable)
            actions_probs = tf.layers.dense(h, self.action_space, activation=tf.nn.softmax,
                                            kernel_initializer=XAVIER, name=f"{name}_actions", trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return actions_probs, params

    def build_critic_network(self):
        """
        :return: The nodes of:
        v: state value,
        dc_rew: discounted reward,
        advantage: the difference between discounted reward and predicted state value,
        loss: the mean square error MSE of the advantage
        train_op: the training operation of the network to be run in a session
        """
        with tf.variable_scope("Critic"):
            h = tf.layers.dense(self.in_state, HIDDEN_UNITS, activation=tf.nn.elu,
                                kernel_initializer=HE_INIT, name="Critic_H")
            v = tf.layers.dense(h, 1, name="V")
            dc_rew = tf.placeholder(tf.float32, [None, 1], name="discounted_R")
            advantage = tf.subtract(dc_rew, v, name="critic_advantage")
            loss = tf.reduce_mean(tf.square(advantage))
            train_op = tf.train.AdamOptimizer(CRITIC_LEARNING_RATE).minimize(loss)
        return v, dc_rew, advantage, loss, train_op

    def sample_action(self, state: np.array) -> int:
        """
        :param state: observed environment space
        :return: an action index sampled based on the probability distribution of predicted actions
        """
        actions_probs = self.sess.run(self.pi, feed_dict={self.in_state: state})
        return np.random.choice(self.action_space, p=actions_probs.ravel())

    def calculate_state_value(self, state: np.array) -> float:
        """
        :param state: observed environment space
        :return: predicted state value based on the critic network
        """
        state = state.reshape(1, -1)
        return self.sess.run(self.v, feed_dict={self.in_state: state})[0, 0]

    def run_trajectory(self):
        steps = 0
        buffer_a = list()
        buffer_adv = list()
        all_states = list()
        all_rewards = list()
        ep_count = 0
        while steps < MAX_STEPS_PER_TRAJECTORY:
            done = False
            states = list()
            rewards = list()
            s = self.env.reset()
            while not done:
                a = self.sample_action(s)
                new_s, rew, done = self.env.step(a)  # actions are the same as their indices, i.e; 0, 1, 2
                states.append(new_s)
                rewards.append(rew)

                buffer_a.append(a)

                s = new_s
                steps += 1
            ep_count += 1
            dc_rews = discount_rewards(rewards, DISCOUNT_RATE)
            for state, dc_rew in zip(states, dc_rews):
                # calculate advantage
                buffer_adv.append(self.sess.run(self.advantage, {self.in_state: state, self.critic_dc_rew: dc_rew}))

            all_states.append(states)
            all_rewards.append(rewards)

        # discount and normalize rewards of each episode
        dc_rewards = discount_and_normalize_rewards(all_rewards, DISCOUNT_RATE)
        for episode_states, episode_rewards in zip(all_states, dc_rewards):
            for state, dc_rew in zip(episode_states, episode_rewards):
                # calculate advantage
                buffer_adv.append(self.sess.run(self.advantage, {self.in_state: state, self.critic_dc_rew: dc_rew}))

        return ep_count, \
               np.concatenate(all_states), \
               np.array(buffer_a), \
               np.concatenate(all_rewards), \
               np.array(buffer_adv)

    def update(self):
        trajectories = Parallel(n_jobs=-1)(delayed(self.run_trajectory())() for _ in range(TRAJECTORIES_PER_UPDATE))
        rollout = list()
        episodes_count = 0
        rewards = list()
        for t in trajectories:
            episodes_count += t[0]
            rewards.append(t[-1][-1])
            rollout += t[1:]

        self._update_actor_critic_networks(rollout)

        # update frozen actor network
        self.sess.run(self.update_frozen_actor_op)

        return episodes_count, np.mean(rewards)

    def _update_actor_critic_networks(self, rollout):
        """
        :param rollout: a list of trajectories
        each trajectory is represented by:
        s: array of states
        a: array of actions
        dc_r: array of discounted rewards
        adv: array of advantage values
        """
        # update actor critic network
        for s, a, adv, dc_rew in rollout:
            for _ in range(UPDATE_STEP):
                self.sess.run(self.actor_train_op, {self.in_state: s, self.actions: a, self.advantage_placeholder: adv})
                self.sess.run(self.critic_train_op, {self.in_state: s, self.critic_dc_rew: dc_rew})
