from q_learning_complete import train_agent, env_layout_builder
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from config import cfg

num_episodes = cfg['num_episodes']
num_low_risk = cfg['reward']['num_low_risk']
num_high_risk = cfg['reward']['num_high_risk']
env_shape = cfg['env_shape']
agent_cfg = cfg['agent']
terminate_after_reward = cfg['terminate_after_reward']

def get_nearest_loc_of_interest(a, env_map, strategy='last'):
    '''
    strategy: 'last' when we terminate after a reward, 'most_common' when we don't.
                most_common just takes the most visited node (since a non-terminating episode
                will tend to oscillate around one reward-dispensing square)
    '''

    history = a.q_history[num_episodes - 1]

    path = history[-1]

    if strategy == 'most_common':
        # most visited node
        values, counts = np.unique(path, return_counts=True)
        fav = values[np.argmax(counts)]
        fav_loc = np.array([fav // env_map.shape[0], fav % env_map.shape[1]])
    if strategy == 'last':
        fav = path[-1]
        fav_loc = np.array([fav // env_map.shape[0], fav % env_map.shape[1]])

    min_dist = 1e6
    nearest = ''
    for i in range(env_map.shape[0]):
        for j in range(env_map.shape[1]):
            loc = env_map[i, j]
            if loc in ['l', 'h']:
                dist = np.linalg.norm(fav_loc - np.array([i, j]))
                if min_dist > dist:
                    nearest = loc
                    min_dist = dist

    # print(nearest)
    return nearest

all_locs = np.array(np.meshgrid(
                np.arange(env_shape[0]), np.arange(env_shape[1])
                )).T.reshape(-1, 2)
np.random.seed(10)

hs = []
ls = []

risky_probs = [0, 0.25, 0.5, 0.75, 1]
num_trials = 100

for risky_prob in risky_probs:
    counts = defaultdict(int)
    for _ in range(num_trials):

        reward_locs = np.random.randint(0, len(all_locs), num_low_risk + num_high_risk)

        low_risk_locs = [{'loc': all_locs[loc], 'type': 'l'} for loc in reward_locs[:num_low_risk]]
        high_risk_locs = [{'loc': all_locs[loc], 'type': 'h'} for loc in reward_locs[num_low_risk:]]
        other_locs = [
            {'loc': [0, 0], 'type': 's'},
            {'loc': [-1, -1], 'type': 'g'},   
        ]
        thing_list = low_risk_locs + high_risk_locs + other_locs


        env_layout = env_layout_builder(env_shape, thing_list)

        # reward_probs = (
        #     [1, 0],                         # low risk
        #     [1 - risky_prob, risky_prob]    # high risk
        # )

        reward_cfg = {
            'low_risk_prob': [1, 0],
            'high_risk_prob': [1 - risky_prob, risky_prob],
            'values': [10, 30]
        }

        a = train_agent(env_layout, reward_cfg, plot=False, num_episodes=num_episodes, agent_cfg=cfg['agent'], terminate_after_reward=terminate_after_reward)

        nearest = get_nearest_loc_of_interest(a, env_layout, strategy='last')
        counts[nearest] += 1

    hs.append(counts['h'] / num_trials)
    ls.append(counts['l'] / num_trials)

plt.plot(risky_probs, hs, color='b', label='chose high risk')
# plt.plot(risky_probs, ls, color='g', label='chose low risk')
plt.ylabel('probability of mouse choice')
plt.xlabel('probability of punishment for risky reward')
plt.legend()
plt.show()
