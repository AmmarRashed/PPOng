import sys
import time
import pickle

sys.path.append("Environment")
from Environment import PongEnvironment


from PPO import *

if __name__ == '__main__':
    env = PongEnvironment()
    ppo = PPO(env, num_states=len(env.observe()), actions=np.arange(3))
    ppo.saver.restore(ppo.sess, "model/Pong_model.ckpt")
    actions = list()
    for i in range(10):
        ep_actions = list()
        done = False
        s = env.reset()
        while not done:
            a = ppo.sample_action(s[None, :])
            ep_actions.append(a)
            env.render()
            time.sleep(1e-3)
            try:
                s, r, done = env.step(a)
            except ValueError:
                s, r, done, _ = env.step(a)
        actions.append(ep_actions)
    print(set(np.concatenate(actions)))
    pickle.dump(actions, open('actions.pkl', 'wb'))
