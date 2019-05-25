import os
from tqdm import tqdm
from PPO import *
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.append("Environment")
from Environment import PongEnvironment

fig, ax = plt.subplots()
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


EPOCHS = 2_000  # maximum number of updates
ENVIRONMENT = "Pong"
if __name__ == "__main__":
    env = PongEnvironment(False)
    num_states = len(env.observe())
    ppo = PPO(env, num_states=num_states, actions=np.arange(3))
    rewards = list()
    eps = 0
    for e in tqdm(range(EPOCHS)):
        episodes_count, mean_trajectories_reward, mean_ep_len = ppo.update()
        eps += episodes_count
        rewards.append(mean_trajectories_reward)
        print(f"Epoch: {e+1}\tRolled episodes: {episodes_count}\tMean ep len: {mean_ep_len}\tMean Reward: {mean_trajectories_reward}\n"
              f"Total episodes: {eps}")
        print()
        #if e % 10 == 0:
        #    x = range(0, len(rewards), 10)
        #    update_plot(x, [rewards[i] for i in x])
        update_plot(range(len(rewards)), rewards)
        if e+1 % 200 == 0:
            print("Saved model checkpoint")
            ppo.saver.save(ppo.sess, f"./model/{ENVIRONMENT}_model.ckpt")
    plt.savefig("ppo.png")
    pickle.dump(rewards, open('pong_rewards.pkl', 'wb'))
