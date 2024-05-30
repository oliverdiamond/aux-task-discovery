import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
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
        assert not self._out_of_bounds(start_pos), 'Start pos must be in bounds'
        self.start_pos = start_pos
        # Goal location for each episode
        assert not self._out_of_bounds(start_pos), 'Goal pos must be in bounds'
        self.goal_pos = goal_pos
        # Current agent pos
        self.agent_pos = (None,None)
        # Obstacles are positions the agent cannot move to
        # Attempting to move to one of these locations results in no move
        self.obstacles = obstacles
        # Total number of states
        self.n_states = self.size*self.size
        # Observation is one-hot encoding of (x,y) grid location
        self.observation_space = Box(low=0, 
                                     high=1, 
                                     shape=(self.n_states,))
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
        assert not self._out_of_bounds(pos), 'Position must be in bounds to convert to one-hot'
        state_idx = pos[0]*self.size + pos[1]
        obs = np.zeros(self.n_states, dtype=np.float32)
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

    def reset(self, *, seed=None, options=None):
        self.agent_pos = self.start_pos
        obs = self._get_obs(pos=self.agent_pos)
        return obs, {}

    def step(self, action):
        self.agent_pos = self._get_new_pos(self.agent_pos, action)
        obs = self._get_obs(self.agent_pos)
        rew = -1
        terminated = self.agent_pos == self.goal_pos
        truncated = False
        return obs, rew, terminated, truncated, {}
    
    def print_grid(self):
        '''
        Prints grid as string
        '''
        grid = ''
        for x in range(-1, self.size+1):
            for y in range(-1, self.size+1):
                # Displays walls around grid
                if x in [-1, self.size] or y in [-1, self.size]:
                    grid+='#'
                elif (x,y) in self.obstacles:
                    grid+='#'
                elif (x,y) == self.start_pos:
                    grid+='S'
                elif (x,y) == self.goal_pos:
                    grid+='G'
                else:
                    grid+=' '
            grid+='\n'
        print(grid)

