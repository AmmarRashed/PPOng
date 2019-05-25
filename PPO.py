import tensorflow as tf
import numpy as np

BATCH_SIZE = 512
MAX_EPS_LEN = 50_000
ROLLOUT_STEPS = 5000  # update gradients after every rollout (all episodes are complete in each trajectory)
DISCOUNT_RATE = 0.99  # reward discount factor
UPDATE_EPOCHS = 30  # decreasing this ensures stable results, but slower learning
EPSILON = 0.2  # for clipping surrogate objective
HIDDEN_UNITS = 128
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-4

HE_INIT = tf.contrib.layers.variance_scaling_initializer()  # variance works better with (R)eLU activation
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


def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    # normalize for stability purposes
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]


class PPO(object):
    def __init__(self, env, num_states, actions):
        self.env = env
        self.sess = tf.Session()
        self.state_space = num_states
        self.action_space = len(actions)
        self.possible_actions = actions
        self.in_state = tf.placeholder(tf.float32, [None, self.state_space])

        # Actor
        self.pi, self.pi_params = self.build_actor_network("PI", trainable=True)
        self.pi_old, self.pi_old_params = self.build_actor_network("PIold", trainable=False)
        self.actions = tf.placeholder(tf.int32, [None, ], name="actor_action")
        self.advantage_placeholder = tf.placeholder(tf.float32, [None, ], name="actor_advantage")
        self.actor_loss = self.calculate_clipped_surrogate_loss()
        self.actor_train_op = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE).minimize(self.actor_loss)
        self.update_old_actor_op = [p_old.assign(p) for p, p_old in zip(self.pi_params, self.pi_old_params)]

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
        pi_old_prob = tf.gather_nd(params=self.pi_old, indices=actions_indices)
        # calculate the ratio between the new and old policies
        ratio = pi_prob / tf.maximum(pi_old_prob, 1e-12)
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
            h1 = tf.layers.dense(self.in_state, HIDDEN_UNITS, activation=tf.nn.relu,
                                 kernel_initializer=HE_INIT, name=f"{name}_H1", trainable=trainable)
            h2 = tf.layers.dense(h1, HIDDEN_UNITS, activation=tf.nn.relu,
                                 kernel_initializer=HE_INIT, name=f"{name}_H2", trainable=trainable)
            actions_probs = tf.layers.dense(h2, self.action_space, activation=tf.nn.softmax,
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
            h1 = tf.layers.dense(self.in_state, HIDDEN_UNITS, activation=tf.nn.elu,
                                 kernel_initializer=HE_INIT, name="Critic_H1")
            h2 = tf.layers.dense(h1, HIDDEN_UNITS, activation=tf.nn.elu,
                                 kernel_initializer=HE_INIT, name="Critic_H2")
            v = tf.layers.dense(h2, 1, name="V")
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
        return self.possible_actions[np.random.choice(self.action_space, p=actions_probs.ravel())]

    def calculate_state_value(self, state: np.array) -> float:
        """
        :param state: observed environment space
        :return: predicted state value based on the critic network
        """
        state = state.reshape(1, -1)
        return self.sess.run(self.v, feed_dict={self.in_state: state})[0, 0]

    def rollout(self):
        steps = 0
        buffer_a = list()
        buffer_adv = list()
        all_states = list()
        all_rewards = list()
        ep_count = 0
        while steps < ROLLOUT_STEPS:
            states = list()
            rewards = list()
            s = self.env.reset()
            for _ in range(MAX_EPS_LEN):
                a = self.sample_action(s[None, :])
                #a = self.sample_action(s.reshape(-1, self.state_space))
                try:
                    new_s, rew, done = self.env.step(a)  # actions are the same as their indices, i.e; 0, 1, 2
                except ValueError:
                    new_s, rew, done, _ = self.env.step(a)  # actions are the same as their indices, i.e; 0, 1, 2
                states.append(new_s)
                rewards.append(rew)

                buffer_a.append(a)

                s = new_s
                steps += 1
                if done:
                    break
            ep_count += 1

            all_states.append(states)
            all_rewards.append(rewards)

        # discount and normalize rewards of each episode
        dc_rewards = discount_and_normalize_rewards(all_rewards, DISCOUNT_RATE)
        for episode_states, episode_rewards in zip(all_states, dc_rewards):
            assert len(episode_states) == len(episode_rewards)
            states = np.array(episode_states).reshape(-1, self.state_space)
            dc_rews = np.array(episode_rewards).reshape(-1, 1)
            advs = self.sess.run(self.advantage,
                                 {self.in_state: states,
                                  self.critic_dc_rew: dc_rews})
            buffer_adv += list(advs.reshape(-1))

        return (ep_count,
                np.concatenate(all_states),
                np.array(buffer_a),
                np.concatenate(all_rewards),
                np.array(buffer_adv),
                np.concatenate(all_rewards),
                steps/ep_count
                )

    def update(self):
        episodes_count, states, actions, rewards, advantages, untransformed_rewards, mean_ep_len = self.rollout()
        permutations = np.random.permutation(len(states))
        states = states[permutations]
        actions = actions[permutations]
        rewards = rewards[permutations]
        advantages = advantages[permutations]

        # update old actor network
        # print("Updating old actor network")
        self.sess.run(self.update_old_actor_op)

        self._update_actor_critic_networks(states, actions, rewards, advantages)

        return episodes_count, np.mean(untransformed_rewards), mean_ep_len

    def _update_actor_critic_networks(self, states, actions, rewards, advantages):
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
        generator = batch_generator(states, actions, rewards, advantages)
        for _ in range(UPDATE_EPOCHS * len(states) // BATCH_SIZE):
            s, a, adv, dc_rew = generator.__next__()
            self.sess.run(self.actor_train_op,
                          {self.in_state: s.reshape(-1, self.state_space),
                           self.actions: a.reshape(-1), self.advantage_placeholder: adv.reshape(-1)})
            self.sess.run(self.critic_train_op,
                          {self.in_state: s.reshape(-1, self.state_space),
                           self.critic_dc_rew: dc_rew.reshape(-1, 1)})
