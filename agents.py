import numpy as np

class Agent_Q:
    def __init__(
        self,
            env,                # environment
            lr,                 # learning rate
            epsilon,            # epsilon
            df=1   # discount_factor
    ):
        self.n_actions = env.action_space.n
        self.n_observations = env.observation_space.n

        self.q_vals = np.zeros((self.n_observations, self.n_actions))
        self.lr = lr
        self.epsilon = epsilon
        self.df = df
        self.td_history = []
        self.q_history = {}
        self.env = env

    def get_action(self, obs):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_vals[obs]))
        
    def update(self,obs,action,reward,finished,next_obs):
        if finished:
            next_q_val = 0
        
        else:
            next_q_val = np.amax(self.q_vals[next_obs])
        
        td_loss = reward + self.df * next_q_val - self.q_vals[obs,action]
        self.q_vals[obs,action] += self.lr * td_loss
        self.td_history.append(td_loss)

agents = {
    'Q': Agent_Q
}