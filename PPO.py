from collections import deque
from queue import Queue

import tensorflow as tf
import numpy as np

BATCH_SIZE = 256
MEMORY_SIZE = 1024
MAX_EPS_LEN = int(1e4)
ROLLOUT_STEPS = 1024  # number of episodes per update
DISCOUNT_RATE = 0.95  # reward discount factor
UPDATE_EPOCHS = 2  # decreasing this ensures stable results, but slower learning
EPSILON = 0.2  # for clipping surrogate objective
HIDDEN_UNITS = 32
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-4
BOLTZMANN_TEMP = 0.1  # boltzmann distribution temprature

HE_INT  = tf.contrib.layers.variance_scaling_initializer()  # variance works better with (R)eLU activation
XAVIER = tf.contrib.layers.xavier_initializer()  # xavier works better with sigmoid activation


#tf.random.set_random_seed(42)
#np.random.seed(42)
# taken from Aurelion Geron implementation:
# https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb


def batch_generator(a, b, c, d):
    start = 0
    while True:
        batch_a, batch_b, batch_c, batch_d = a[start:start+BATCH_SIZE], b[start:start+BATCH_SIZE], \
                                             c[start:start+BATCH_SIZE], d[start:start+BATCH_SIZE]
        if len(batch_a) < BATCH_SIZE:
            remaining = BATCH_SIZE - len(batch_a)
            batch_a = np.concatenate([batch_a, batch_a[:remaining]])
            batch_b = np.concatenate([batch_b, batch_b[:remaining]])
            batch_c = np.concatenate([batch_c, batch_c[:remaining]])
            batch_d = np.concatenate([batch_d, batch_d[:remaining]])
            start = remaining
        else:
            start += BATCH_SIZE
        yield batch_a, batch_b, batch_c, batch_d


def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards


def normalize_rewards(all_rewards):
    flat_rewards = np.concatenate(all_rewards)
    m, std = flat_rewards.mean(), flat_rewards.std()
    return [(r-m)/std for r in all_rewards]



class PPO(object):
    def __init__(self, env, num_states, actions):
        self.memory_states = deque(maxlen=MEMORY_SIZE)  # used as a queue not as an offline learning method
        self.memory_actions = deque(maxlen=MEMORY_SIZE)
        self.memory_rewards = deque(maxlen=MEMORY_SIZE)
        self.env = env
        self.sess = tf.Session()
        self.state_space = num_states
        self.action_space = len(actions)
        self.possible_actions = actions
        self.in_state = tf.placeholder(tf.float32, [None, self.state_space])

        # Actor
        self.pi, self.pi_params = self.build_actor_network("PI", trainable=True)
        self.pi_old, self.pi_old_params = self.build_actor_network("PIold", trainable=False)
        self.actions_placeholder = tf.placeholder(tf.int32, [None, ], name="actor_action")
        self.advantage_placeholder = tf.placeholder(tf.float32, [None, ], name="actor_advantage")
        self.actor_loss = self.calculate_clipped_surrogate_loss()
        self.actor_train_op = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE).minimize(self.actor_loss)
        self.update_old_actor_op = [p_old.assign(p) for p, p_old in zip(self.pi_params, self.pi_old_params)]

        self.action = tf.random.categorical(tf.log(self.pi), num_samples=1)
        #self.act = tf.random.categorical(self.pi, num_samples=1)
        self.action = tf.reshape(self.action, shape=[-1])

        # Critic
        self.v, self.critic_dc_rew, self.advantage, self.critic_loss, self.critic_train_op \
            = self.build_critic_network()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def calculate_clipped_surrogate_loss(self):
        """
        :return: clipped surrogate loss
        """
        actions_indices = tf.stack([tf.range(tf.shape(self.actions_placeholder)[0], dtype=tf.int32), self.actions_placeholder], axis=1)
        # get probabilities of actions according to the corresponding policies
        pi_prob = tf.gather_nd(params=self.pi, indices=actions_indices)
        pi_old_prob = tf.gather_nd(params=self.pi_old, indices=actions_indices)
        # calculate the ratio between the new and old policies
        #ratio = pi_prob / tf.maximum(pi_old_prob, 1e-12)

        ratio = tf.exp(tf.log(pi_prob+1e-12) - tf.log(pi_old_prob+1e-12))
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
            h1 = tf.layers.dense(self.in_state, HIDDEN_UNITS, activation=tf.nn.sigmoid,
                                 kernel_initializer=XAVIER, name=f"{name}_H1", trainable=trainable)
            h2 = tf.layers.dense(h1, HIDDEN_UNITS, activation=tf.nn.sigmoid,
                                 kernel_initializer=XAVIER, name=f"{name}_H2", trainable=trainable)
            l = tf.layers.dense(h2, self.action_space, activation=tf.nn.sigmoid, trainable=trainable, kernel_initializer=XAVIER)
            actions_probs = tf.layers.dense(tf.divide(l, BOLTZMANN_TEMP), self.action_space, activation=tf.nn.softmax,
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
            h1 = tf.layers.dense(self.in_state, HIDDEN_UNITS, activation=tf.nn.sigmoid,
                                 kernel_initializer=XAVIER, name="Critic_H1")
            h2 = tf.layers.dense(h1, HIDDEN_UNITS, activation=tf.nn.sigmoid,
                                 kernel_initializer=XAVIER, name="Critic_H2")
            v = tf.layers.dense(h2, 1, name="V", activation=None)
            dc_rew = tf.placeholder(tf.float32, [None, 1], name="discounted_R")
            advantage = tf.subtract(dc_rew, v, name="critic_advantage")
            loss = tf.reduce_mean(tf.square(advantage))
            train_op = tf.train.AdamOptimizer(CRITIC_LEARNING_RATE).minimize(loss)
        return v, dc_rew, advantage, loss, train_op


    def calculate_state_value(self, state: np.array) -> float:
        """
        :param state: observed environment space
        :return: predicted state value based on the critic network
        """
        state = state.reshape(-1, self.state_space)
        return self.sess.run(self.v, feed_dict={self.in_state: state})[0, 0]

    def rollout(self):
        steps = 0
        ep_count = 0
        actions_set = set()
        mean_rews = 0
        filling = False
        if len(self.memory_states) == 0:
            print("Filling memory...")
            filling = True
        while len(self.memory_states) < MEMORY_SIZE or steps < ROLLOUT_STEPS:
            states = list()
            actions = list()
            rewards = list()
            s = self.env.reset()
            for _ in range(MAX_EPS_LEN):
                a = self.sess.run(self.action, {self.in_state: s.reshape(-1, self.state_space)})[0]
                actions_set.add(a)
                #a = self.sample_action(s.reshape(-1, self.state_space))
                try:
                    new_s, rew, done = self.env.step(a)  # actions are the same as their indices, i.e; 0, 1, 2
                except ValueError:
                    # for gym environment
                    new_s, rew, done, _ = self.env.step(a)  # actions are the same as their indices, i.e; 0, 1, 2
                states.append(new_s)
                rewards.append(rew)

                actions.append(a)

                s = new_s
                steps += 1
                if done or steps >= ROLLOUT_STEPS:
                    break
            ep_count += 1
            dc_rews = discount_rewards(rewards, DISCOUNT_RATE)
            mean_rews += np.sum(dc_rews)
            self.memory_states.append(states)
            self.memory_actions.append(actions)
            self.memory_rewards.append(dc_rews)
        if filling:
            print("Memory filled!")
        return actions_set, mean_rews/ep_count, steps, ep_count

    def update(self):
        #episodes_count, states, actions, rewards, advantages, untransformed_rewards, mean_ep_len = self.rollout()
        actions_set, avg_rews, steps, ep_count = self.rollout()

        # update old actor network
        # print("Updating old actor network")
        self.sess.run(self.update_old_actor_op)
        self._update_actor_critic_networks()
        return actions_set, avg_rews, steps, ep_count

    def _update_actor_critic_networks(self):
        """
        for a set of trajectories (i.e. rollout)
        each trajectory is represented by:
        s: array of states
        a: array of actions
        dc_r: array of discounted rewards
        adv: array of advantage values
        """
        # update actor critic network
        # print("Updating Actor Critic Networks")
        permutations = np.random.permutation(MEMORY_SIZE)
        states = np.concatenate(self.memory_states).reshape(-1, self.state_space)[permutations, :]
        actions = np.concatenate(self.memory_actions).reshape(-1)[permutations]
        norm_dc_rews = np.concatenate(normalize_rewards(self.memory_rewards))[permutations].reshape(-1, 1)

        advs = self.sess.run(self.advantage, {self.in_state: states,
                                              self.critic_dc_rew: norm_dc_rews})
        generator = batch_generator(states,
                                    actions,
                                    norm_dc_rews,
                                    advs)
        for _ in range(UPDATE_EPOCHS * MEMORY_SIZE // BATCH_SIZE):
            s, a, adv, dc_rew = generator.__next__()
            self.sess.run(self.actor_train_op,
                          {self.in_state: s, self.actions_placeholder: a, self.advantage_placeholder: adv.reshape(-1)})
            self.sess.run(self.critic_train_op,
                          {self.in_state: s, self.critic_dc_rew: dc_rew})
