from PPO import *
import matplotlib.pyplot as plt

import sys
sys.path.append("Environment")
from Environment import PongEnvironment

fig, ax = plt.subplots()
ax.set_xlabel("Epochs")
ax.set_ylabel("Rewards")
fig.show()


def update_plot(x):
    plt.cla()
    ax.plot(range(x), x)
    plt.pause(1e-4)
    fig.tight_layout()


EPOCHS = 2  # maximum number of updates
if __name__ == "__main__":
    env = PongEnvironment()
    ppo = PPO(env)
    rewards = list()
    eps = 0
    for e in range(EPOCHS):
        episodes_count, mean_trajectories_reward = ppo.update()
        eps += episodes_count
        rewards.append(mean_trajectories_reward)
        print(f"Epoch: {e}\tEpisodes: {episodes_count}\tMean Reward: {mean_trajectories_reward}\nEpisodes: {eps}")
        print()
        if e % 10 == 0:
            ppo.saver.save(ppo.sess, f"./results/{e}_memory.ckpt")
    plt.savefig("ppo.png")
