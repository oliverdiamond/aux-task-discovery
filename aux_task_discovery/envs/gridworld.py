import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
from typing import Tuple

class GridWorldEnv(gym.Env):
    def __init__(self, 
                 size = 10, 
                 start_pos = (0,0), 
                 goal_pos = (9,9), 
                 obstacles = [],
                 action_noise = 0.0, 
                 seed = 42):
        # Size of square grid
        self.size = size
        # Starting location for each episode
        self.start_pos = start_pos
        # Goal location for each episode
        self.goal_pos = goal_pos
        # Current agent pos
        self.curr_pos = (None, None)
        # Obstacles are positions the agent cannot move to
        # Attempting to move to one of these locations results in no move
        self.obstacles = obstacles
        # Total number of states
        self.n_states = self.size*self.size
        # Observation is one-hot encoding of (x,y) grid location
        self.observation_space = Discrete(self.n_states)
        # Actions are directions: up, left, down, right
        self.action_space = Discrete(4)
        # Agent moves in selected direction with prob 1-action_noise
        # and in each other direction with prob action_noise/3
        self.action_noise = action_noise
        # Change in (x,y) pos assosiated with each action
        self.action_deltas = [(-1,0), (0,-1), (1,0), (0,1)]
        # Random generator for action noise
        self.noise_generator = np.random.RandomState(seed)

    def _out_of_bounds(self, pos):
        return not (-1<pos[0]<self.size and -1<pos[1]<self.size)

    def _is_obstacle(self, pos):
        return pos in self.obstacles

    def _get_obs(self, pos):
        '''
        Converts (x,y) pos to one-hot numpy array
        '''
        state_idx = pos[0]*self.size + pos[1]
        obs = np.zeros(self.n_states, dtype=np.int64)
        obs[state_idx] = 1
        return obs
    
    def _get_new_pos(self, pos, action):
        if self.noise_generator.random() < self.action_noise:
            # Choose random unselected action 
            action_set = [0,1,2,3]
            action_set.remove(action)
            action = self.noise_generator.choice(action_set)
        # Get change in coordinates
        delta_x, delta_y = self.action_deltas[action]
        new_pos = (pos[0]+delta_x, pos[1]+delta_y)
        if self._out_of_bounds(new_pos) or self._is_obstacle(new_pos):
            # No change if action moves to obstacle or outside grid
            return pos
        return new_pos

    def reset(self):
        self.curr_pos = self.start_pos
        obs = self._get_obs(pos=self.curr_pos)
        return obs, {}

    def step(self, action):
        new_pos = self._get_new_pos(self.curr_pos, action)
        obs = self._get_obs(new_pos)
        rew = -1
        terminated = new_pos == self.goal_pos
        truncated = False
        return obs, rew, terminated, truncated, {}
    
    def print_grid(self):
        '''
        Prints grid as string
        '''
        pass

