import numpy as np
from config import cfg
from q_learning_complete import train_agent, env_layout_builder

num_episodes = cfg['num_episodes']
num_low_risk = cfg['reward']['num_low_risk']
num_high_risk = cfg['reward']['num_high_risk']
env_shape = cfg['env_shape']
agent_cfg = cfg['agent']
reward_cfg = cfg['reward']
max_steps = cfg['max_steps_per_episode']

# picking random locs. This is definitely not the best way of doing this but idk
all_locs = np.array(np.meshgrid(
    np.arange(env_shape[0]), np.arange(env_shape[1])
    )).T.reshape(-1, 2)

reward_locs = np.random.randint(0, len(all_locs), num_low_risk + num_high_risk)

# some are high-risk, some are low
low_risk_locs = [{'loc': all_locs[loc], 'type': 'l'} for loc in reward_locs[:num_low_risk]]
high_risk_locs = [{'loc': all_locs[loc], 'type': 'h'} for loc in reward_locs[num_low_risk:]]
other_locs = [
    {'loc': [0, 0], 'type': 's'},   #start
    {'loc': [-1, -1], 'type': 'g'}, #finish (dt this is necessary anymore)
]
thing_list = low_risk_locs + high_risk_locs + other_locs
env_layout = env_layout_builder(env_shape, thing_list)

train_agent(
    env_layout, 
    reward_cfg=reward_cfg, 
    agent_cfg=agent_cfg, 
    plot=True, 
    num_episodes=num_episodes, 
    max_steps=max_steps
    )