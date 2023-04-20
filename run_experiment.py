from q_learning_complete import train_agent, env_layout_builder
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

num_episodes = 1001

def get_nearest_loc_of_interest(a, env_map):
    history = a.q_history[num_episodes - 1]

    path = history[-1]

    # most visited node
    values, counts = np.unique(path, return_counts=True)
    fav = values[np.argmax(counts)]
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

    print(nearest)
    return nearest

num_trials = 10
num_low_risk = 3
num_high_risk = 3

env_shape = (7, 7)
all_locs = np.array(np.meshgrid(
                np.arange(env_shape[1]), np.arange(env_shape[1])
                )).T.reshape(-1, 2)
np.random.seed(1)

hs = []
ls = []

risky_probs = [0, 0.25, 0.5, 0.75, 1]
# risky_probs = [0, 1]

for risky_prob in risky_probs:
    counts = defaultdict(int)
    for _ in range(num_trials):

        reward_locs = np.random.randint(0, len(all_locs), num_low_risk + num_high_risk)

        low_risk_locs = [{'loc': all_locs[loc], 'type': 'l'} for loc in reward_locs[:num_low_risk]]
        high_risk_locs = [{'loc': all_locs[loc], 'type': 'h'} for loc in reward_locs[num_low_risk:]]
        other_locs = [
            {'loc': [0, 0], 'type': 's'},
            {'loc': [6, 6], 'type': 'g'},   
        ]
        thing_list = low_risk_locs + high_risk_locs + other_locs


        env_layout = env_layout_builder(env_shape, thing_list)

        reward_probs = (
            [1, 0],                         # low risk
            [1 - risky_prob, risky_prob]    # high risk
        )

        reward_cfg = {
            'probs': reward_probs,
            'values': [1, 3]
        }

        a = train_agent(env_layout, reward_cfg, plot=True, num_episodes=num_episodes)

        nearest = get_nearest_loc_of_interest(a, env_layout)
        counts[nearest] += 1

    hs.append(counts['h'] / num_trials)
    ls.append(counts['l'] / num_trials)

plt.plot(risky_probs, hs, color='b', label='chose high risk')
plt.plot(risky_probs, ls, color='g', label='chose low risk')
plt.ylabel('probability of mouse choice')
plt.xlabel('probability of punishment for risky reward')
plt.legend()
plt.show()
