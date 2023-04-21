import numpy as np
from copy import deepcopy

class Agent_Q:
    def __init__(
        self,
            env,                # environment
            lr,                 # learning rate
            epsilon,            # epsilon
            df=1                # discount_factor
    ):
        # # of actions and observations
        self.n_actions = env.action_space.n
        self.n_observations = env.observation_space.n

        # Q values for each action in each observation
        self.q_vals = np.zeros((self.n_observations, self.n_actions))
        self.lr = lr # learning rate
        self.epsilon = epsilon
        self.df = df # discount factor
        self.td_history = []
        self.q_history = {}
        self.env = env

    # Get the agent's action epsilon-greedy selection given an observation
    def get_action(self, obs):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_vals[obs]))
    
    # Update values based on the action that gives the most value
    def update(self,obs,action,reward,finished,next_obs):
        if finished:
            next_q_val = 0
        
        else:
            next_q_val = np.amax(self.q_vals[next_obs])
        
        td_loss = reward + self.df * next_q_val - self.q_vals[obs,action]
        self.q_vals[obs,action] += self.lr * td_loss
        self.td_history.append(td_loss)

class Agent_SARSA:
    def __init__(
        self,
            env,                # environment
            lr,                 # learning rate
            epsilon,            # epsilon
            df=1                # discount_factor
    ):
        
        # # of actions and observations
        self.n_actions = env.action_space.n
        self.n_observations = env.observation_space.n

        self.q_vals = np.zeros((self.n_observations, self.n_actions))
        self.lr = lr # learning rate
        self.epsilon = epsilon
        self.df = df # discount factor
        self.td_history = []
        self.q_history = {}
        self.curr_action = self.get_action(env.start_pos)
        self.env = env

    # Get agent's action given an observation
    def get_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0,1,2,3]).astype(int)
        else:
            return int(np.argmax(self.q_vals[obs]))
        
    # Updates q_values using the actual action selected, not the one
    # returning the most value
    def update(self,obs,action,reward,finished,next_obs,next_action):
        if finished:
            next_q_val = 0
        
        else:
            next_q_val =self.q_vals[next_obs][next_action]
        
        td_loss = reward + self.df * next_q_val - self.q_vals[obs,action]
        self.q_vals[obs,action] += self.lr * td_loss
        self.td_history.append(td_loss)


agents = {
    'Q': Agent_Q,
    'SARSA': Agent_SARSA
}