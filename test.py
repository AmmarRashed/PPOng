import sys
import time
import pickle

sys.path.append("Environment")
from Environment import PongEnvironment


from PPO import *

if __name__ == '__main__':
    env = PongEnvironment(False)
    ppo = PPO(env, num_states=len(env.observe()), actions=np.arange(3))
    ppo.saver.restore(ppo.sess, "model_res/Pong_model.ckpt")
    actions = set()
    all_scores = list()
    for trial in range(10):
        score = 0
        for i in range(100):
            ep_actions = list()
            done = False
            s = env.reset()
            while not done:
                a = ppo.sess.run(ppo.action, {ppo.in_state: s.reshape(-1, ppo.state_space)})[0]
                ep_actions.append(a)
                #env.render()
                #time.sleep(1e-3)
                try:
                    s, r, done = env.step(a)
                except ValueError:
                    s, r, done, _ = env.step(a)
                actions.add(a)
        score += env.right_point
        env.right_point = 0
        env.left_point = 0
        print(f"{trial+1}: {score}")
        all_scores.append(score)
    print(actions)
    pickle.dump(all_scores, open("scores_res.pkl", 'wb'))

