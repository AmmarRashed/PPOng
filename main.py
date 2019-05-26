import os
from tqdm import tqdm
from PPO import *
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.append("Environment")
from Environment import PongEnvironment

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlabel("Epochs")
ax.set_ylabel("Rewards")
fig.show()


try:
    os.makedirs("model")
except FileExistsError:
    pass


def update_plot(x, y):
    plt.cla()
    ax.plot(x, y)
    plt.pause(1e-4)
    fig.tight_layout()


EPOCHS = int(2500)  # maximum number of updates
ENVIRONMENT = "Pong"
if __name__ == "__main__":
    env = PongEnvironment(False)
    num_states = len(env.observe())
    ppo = PPO(env, num_states=num_states, actions=np.arange(3))
    rewards = list()
    steps_count = 0
    eps = 0
    for e in tqdm(range(1, EPOCHS+1)):
        actions_set, avg_rews, steps, ep_count = ppo.update()
        steps_count += steps
        eps += ep_count
        rewards.append(avg_rews)

        if e % 10 == 0:
            x = range(0, len(rewards), 10)
            update_plot(x, [rewards[i] for i in x])
            print(
                f"Update: {e + 1}\tReward: {avg_rews}\tSteps: {steps_count}\n"
                f"Total episodes: {eps}\tactions: {actions_set}")
            print()
        if e % 500 == 0:
            update_plot(range(len(rewards)), rewards)
            print("Saved model checkpoint")
            ppo.saver.save(ppo.sess, f"./model_res/{ENVIRONMENT}_model.ckpt")
            plt.savefig("ppo_res.png")

            pickle.dump(rewards, open('pong_rewards_res.pkl', 'wb'))
