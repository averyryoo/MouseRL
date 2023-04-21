cfg = {
    'agent': {
        'rl_algo': 'Q',                 # Q or SARSA
        'lr': 0.3,                      # learning rate
        'epsilon': 0.05,                # epsilon
        'df': 0.5                       # discount factor
    },
    'num_episodes': 20,
    'max_steps_per_episode': 50,
    'env_shape': [12, 12],
    'reward': {
        'values': [10, 30],             # [low, high]
        'num_high_risk': 5,
        'num_low_risk': 5,
        'high_risk_prob': [0.9, 0.1],   # [prob of receiving reward, prob of punishment]
        'low_risk_prob': [1, 0]         # [prob of receiving reward, prob of punishment]
    },
    'terminate_after_reward': False
}