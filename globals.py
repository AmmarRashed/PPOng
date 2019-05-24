from multiprocessing import cpu_count

GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
GLOBAL_RUNNING_R = list()

EP_MAX = 1000

NUM_WORKERS = cpu_count()

EP_LEN = 10000
GAMMA = 0.99                 # reward discount factor
MIN_BATCH_SIZE = 512         # minimum batch size for updating PPO
UPDATE_STEP = 100            # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective
STATE_SPACE = 2
ACTION_SPACE = 3
HIDDEN_UNITS = 128
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-4
