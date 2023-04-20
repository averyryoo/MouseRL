# -*- coding: utf-8 -*-
"""q-learning_complete.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mf5LLW8KTBW4WEEhF5fMva6ket_HJ_Qs

# Reinforcement activity

In this activity, you will make an agent that uses reinforcement learning to navigate the Cliff Walking environment. You will write methods in the agent class to implement Q-learning and an $\epsilon$-greedy action selection strategy. 

There are 3 main tasks for you to complete and 2 tasks to plot the results of the RL model. Cells have clearly marked `# TODO` and `#####` comments for you to insert your code between. Variables assigned to `None` should keep the same name but assigned to their proper implementation.

1. Complete the `get_action` method of the agent and implement an $\epsilon$-greedy strategy
2. Complete the `update` method of the agent and implement the temporal difference loss and update equation for Q-learning
3. Complete the training loop for the RL model
4. Plot the training metrics
5. Plot the path and policy at various points of training
"""

# TODO: Run this in Google Colab to install the RL gym package
# !pip install gymnasium

# TODO: Run this cell to import relevant packages

import numpy as np  # Arrays and numerical computing
from tqdm import tqdm  # Progress bar
import gymnasium as gym  # Reinforcement learning environments
# from gymnasium.envs.classic_control import rendering
import random  # Random number generation
from copy import deepcopy

import matplotlib.pyplot as plt  # Plotting and visualization
from viz_util import plot_agent_history
from agents import agents
# class Agent:
#     def __init__()

class NavigationEnv(gym.Env):
    def __init__(self, env_map, reward_cfg):
        # Define action and observation spaces
        self.n_rows = env_map.shape[0]
        self.n_cols = env_map.shape[1]
        self.action_space = gym.spaces.Discrete(4) # Up, Down, Left, Right
        # self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.observation_space = gym.spaces.Discrete(self.n_rows*self.n_cols)

        # Initialize starting and goal positions
        for i in range(env_map.shape[0]):
            for j in range(env_map.shape[1]):
                if env_map[i,j] == "s":
                    self.start_pos = np.array([i,j])
                if env_map[i,j] == "g":
                    self.goal_pos = np.array([i,j])

        # self.start_pos = np.array([0,0])
        # self.goal_pos = np.array([self.n_rows-1,self.n_cols-1])
        self.current_pos = np.where(env_map == 's')
        #(self.current_pos)
        self.env_map = env_map

        # Define reward for reaching the goal state
        self.goal_reward = 100

        # Define reward for taking each step
        self.step_reward = -1

        # Change in position
        self.action_map = {
            0: np.array([1,0]),  # Up
            1: np.array([0,1]),  # Right
            2: np.array([-1,0]), # Down
            3: np.array([0,-1]), # Left
        }


        self.rewards = reward_cfg['values']
        self.safe_reward_probs = reward_cfg['low_risk_prob']
        self.risky_reward_probs = reward_cfg['high_risk_prob']

        # # Define maximum number of steps allowed
        # self.max_steps = 100

        # # Define current step
        # self.current_step = 0

        # # Define initial state
        # self.current_state = np.array([0, 0])

        # # Define goal state
        # self.goal_pos = np.array([50, 50])
        # # Define other squares

        # self.square_pos = np.array(
        #     [[i,i] for i in range(10,80,10)]
        # )

        # # Define reward for each square
        # self.small_reward = 10
        # self.large_reward = 50
        # self.penalty_reward = -20

        # # Define penalty turns for each large reward square
        # self.penalty_turns = 3
        # self.penalty_countdown = [0, 0, 0]

    def step(self, action):
        assert action in set([0,1,2,3])
        
        # new_state = deepcopy(self.current_state)

        new_pos = np.add(self.current_pos,self.action_map[action])

        if new_pos[0] in range(self.n_rows) and new_pos[1] in range(self.n_cols):
            self.current_pos = new_pos
        
        reward_type = self.env_map[self.current_pos[0],self.current_pos[1]]

        #reward = self.reward_map[reward_type]

        if reward_type == "n" or reward_type == 's' or reward_type == 'g':
            reward = -1
            #finished = False
        elif reward_type == "l":
            reward = np.random.choice([self.rewards[0],-10],p=self.safe_reward_probs)
            #finished = True
        elif reward_type == "h":
            reward = np.random.choice([self.rewards[1],-10], p=self.risky_reward_probs)
            #finished = True
        else:
            raise Exception("Unknown reward type - use 'n', 'l', 'h', or 'g'")

        # If caught in a trap, return to start
        if reward == -100:
            self.current_pos = deepcopy(self.start_pos)
        #finished = False

        # If reached the final destination, terminate
        # finished = np.all(self.current_pos == self.goal_pos)

        # Index of new state in the flattened 
        obs = self.n_cols*self.current_pos[0] + self.current_pos[1]

        return obs, reward, finished, None, None

        # done = self.current_step >= self.max_steps or np.array_equal(self.current_pos,self.goal_pos)



        # # Check if current position is on any other squares
        # reward = self.step_reward
        # for i, square in enumerate(self.square_pos):
        #     if np.array_equal(self.current_pos, square):
        #         if i < 3:
        #             reward += self.small_reward
        #         else:
        #             if self.penalty_countdown[i-3] > 0:
        #                 self.penalty_countdown[i-3] -= 1
        #             else:
        #                 if np.random.rand() < 0.5:
        #                     reward += self.large_reward
        #                     self.penalty_countdown[i-3] = self.penalty_turns
        #                 else:
        #                     reward += self.penalty_reward

        # # Calculate total reward
        # if done and np.array_equal(self.current_pos, self.goal_pos):
        #     reward += self.goal_reward

        # # Define observation
        # observation = np.zeros((64, 64, 3), dtype=np.uint8)
        # observation[self.current_pos[0], self.current_pos[1], :] = 255
        # for square in self.square_pos:
        #     observation[square[0], square[1], 0] = 255

        # # Return step information
        # return observation, reward, done, {}

        # # Define movement based on action taken
        # if action == 0: # Up
        #     self.current_pos[1] += 1
        # elif action == 1: # Down
        #     self.current_pos[1] -= 1
        # elif action == 2: # Left
        #     self.current_pos[0] -= 1
        # # elif action == 3: # Right
        # else:
        #     self.current_pos[0] += 1

        # # Update current step
        # self.current_step += 1

        # # Calculate distance to goal state
        # distance_to_goal = np.sqrt(np.sum((self.current_pos - self.goal_pos) ** 2))

        # Check if maximum number of steps has been reached or if goal state has been reached
        # if self.current_step >= self.max_steps or np.array_equal(self.current_pos, self.goal_pos):
        #     done = True
        # else:
        #     done = False

    def reset(self):
        # Reset current position
        self.current_pos = deepcopy(self.start_pos)

        return self.n_cols*self.current_pos[0] + self.current_pos[1]

        # # Reset penalty countdowns
        # self.penalty_countdown = [0, 0, 0]

        # # Reset current step
        # self.current_step = 0

        # # Define observation
        # observation = np.zeros((64, 64, 3), dtype=np.uint8)
        # observation[self.current_pos[0], self.current_pos[1], :] = 255
        # for square in self.square_pos:
        #     observation[square[0], square[1], 0] = 255

        # # Return initial observation
        # return observation

    def render(self, ax=None, mode='human'):

        if ax is None:
            _, ax = plt.subplots(1)
        # pass
        # Define color for goal state
        goal_color = [0, 255, 0]

        # Define color for current position
        current_pos_color = [255, 0, 0]

        # Define color for other squares
        empty_color = [255, 255, 255]
        small_reward_color = [255, 255, 0]
        large_reward_color = [0, 255, 255]
        penalty_reward_color = [255, 0, 255]

        colors = {
            'n': empty_color,
            's': empty_color,
            'l': small_reward_color,
            'h': large_reward_color,
            'g': goal_color
        }

        # Define color for penalty countdowns
        penalty_countdown_color = [128, 128, 128]

        # Define image
        #img = np.zeros((64, 64, 3), dtype=np.uint8)
        img = np.zeros((*self.env_map.shape, 3))

        # Add goal state to image
        #img[self.goal_pos[0], self.goal_pos[1], :] = goal_color

       

        for i in range(self.env_map.shape[0]):
            for j in range(self.env_map.shape[1]):
                img[i, j] = colors[self.env_map[i, j]]

        # Add current position to image
        #img[self.current_pos[0], self.current_pos[1], :] = current_pos_color
        
        # Add other squares to image
        # for i, square in enumerate(self.square_pos):
        #     if i < 3:
        #         color = small_reward_color
        #     else:
        #         color = large_reward_color
        #         if self.penalty_countdown[i-3] > 0:
        #             img[square[0], square[1], :] = penalty_countdown_color
        #     img[square[0], square[1], :] = color

        # Show image
        if mode == 'human':
            
            # if self.viewer is None:
            #     self.viewer = rendering.SimpleImageViewer()
            # self.viewer.imshow(img)
            # return self.viewer.isopen
            ax.imshow(img)
            ax.invert_yaxis()
            #plt.show()
        else:
            return img

    def close(self):
        pass
        # if self.viewer is not None:
        #     self.viewer.close()
        #     self.viewer = None

def env_layout_builder(shape,things_list):
    """
    Builds an environment by passing in a shape and a list of dictionaries of special squares
    Args:
        shape: (nxn) shape of environment
        things_list: list of dicts of form {
                                                "loc": [x,y] coordinates
                                                "type": "s", "l", "h", "g" i.e., start, low risk, high risk, goal
                                            }
    """
    env = np.full(shape,"n")
    for thing in things_list:
        env[thing["loc"][0],thing["loc"][1]] = thing["type"]
    return env

def train_agent(env_layout, reward_cfg, agent_cfg, plot=False, num_episodes=10000, max_steps=50):
    
    agent_type = agent_cfg['rl_algo']
    lr = agent_cfg['lr']
    epsilon = agent_cfg['epsilon']
    df = agent_cfg['df']

    env = NavigationEnv(env_layout, reward_cfg)
    # env.render()
    # plt.show()

    a = agents[agent_type](env,lr,epsilon,df)

    save_episodes = [0, 5, 10, 15, 25, 50, 1000, 5000, 10000, num_episodes - 1]

    if agent_type == "Q":
        for episode in tqdm(range(num_episodes)):
            obs = env.reset()
            path = [obs]
            finished = False
            i = 0
            while not finished and i < max_steps:
                i += 1
                action = a.get_action(obs)

                next_obs, reward, finished, _, _ = env.step(action)

                a.update(obs, action, reward, finished, next_obs)

                obs = next_obs
                path.append(obs)

            if episode in save_episodes:
                a.q_history[episode] = (a.q_vals.copy(), path)

    # Kinda hacky but I do not have the mental fortitude to make a generalized training loop for both agents rn
    if agent_type == "SARSA":
        for episode in tqdm(range(num_episodes)):
            obs = env.reset()
            path = [obs]
            finished = False
            i = 0
            while not finished and i < max_steps:
                i += 1
                # action = a.get_action(obs)

                next_obs, reward, finished, _, _ = env.step(a.curr_action)
                a_actual_curr = deepcopy(a.curr_action)
                a.curr_action = a.get_action(next_obs)
                a.update(obs, a_actual_curr, reward, finished, next_obs,a.curr_action)

                obs = next_obs
                path.append(obs)

            if episode in save_episodes:
                a.q_history[episode] = (a.q_vals.copy(), path)


    if plot:
        plot_agent_history(a)

    return a
    
def main():
    thing_list = [
        {"loc": [0,0], "type": "s"},
        {"loc": [0,3], "type": "l"},
        {"loc": [0,6], "type": "l"},
        {"loc": [3,6], "type": "l"},
        {"loc": [3,0], "type": "h"},
        {"loc": [6,0], "type": "h"},
        {"loc": [6,3], "type": "h"},
        {"loc": [6,6], "type": "g"},
    ]

    env_layout = env_layout_builder([7,7],thing_list)

    reward_cfg = {
        "probs": (
            [0.9, 0.1],
            [0.5, 0.5]
        ),
        "values": [1,3]
    }

    train_agent(env_layout, reward_cfg, plot=True, agent_type="SARSA")

if __name__ == "__main__":
    main()