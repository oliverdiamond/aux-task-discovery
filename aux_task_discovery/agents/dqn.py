import copy
from typing import Union

import numpy as np
import torch
from torch import optim

from aux_task_discovery.utils import random_argmax
import aux_task_discovery.utils.pytorch_utils as ptu
from aux_task_discovery.agents import BaseAgent, ReplayBuffer
from aux_task_discovery.models import ActionValueNetwork


class DQNAgent(BaseAgent):
    '''
    DQN Agent with FF Q-network and Adam optimizer
    '''
    def __init__(
        self,
        input_size,
        n_actions,
        learning_rate = 1e-4, 
        epsilon = 0.1,
        anneal_epsilon = False,
        n_anneal = 10000,
        gamma = 0.9,
        n_hidden = 1, 
        hidden_size = 500,
        activation = 'tanh',
        buffer_size = 1000,
        batch_size = 16,
        update_freq = 1,
        target_update_freq=100,
        learning_start = 100
    ):
        self.input_size = input_size
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.anneal_epsilon = anneal_epsilon
        self.n_anneal = n_anneal
        self.gamma = gamma
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.activation = activation
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.target_update_freq= target_update_freq
        self.learning_start = learning_start
        self.step_idx = 1
        self._setup_replay_buffer()
        self._setup_model()
        self._setup_optimizer()
        self._update_target_network()

    def _setup_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_size)

    def _setup_model(self):
        self.model = ActionValueNetwork(
                        input_size=self.input_size,
                        n_actions=self.n_actions,
                        n_hidden=self.n_hidden,
                        hidden_size=self.hidden_size,
                        activation=self.activation,
                        )
    
    def _setup_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def _update_target_network(self):
        self.target_model = copy.deepcopy(self.model)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray):
        '''
        Selects e-greedy action using q-values from model
        '''
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        obs = ptu.from_numpy(obs)
        q_vals = ptu.to_numpy(self.model(obs))
        return random_argmax(q_vals)

    def step(
        self, 
        obs: np.ndarray, 
        act: int,
        rew: Union[float, int], 
        next_obs: np.ndarray, 
        terminated: bool,
        truncated: bool,
    ):
        '''
        Adds a single transition to the replay buffer and updates model weights 
        and algorithm parameters as nessesary
        '''
        log_data = {}
        self.replay_buffer.insert(obs, act, rew, next_obs, terminated, truncated)
        if self.anneal_epsilon and self.step_idx <= self.n_anneal:
            # Linearly decreases epsilon from init value to 0.1 over n_anneal steps
            self.epsilon -= (self.epsilon-0.1)/self.n_anneal
            log_data['epsilon'] = self.epsilon
        if self.step_idx >= self.learning_start and self.step_idx % self.update_freq == 0:
            train_data = self.train()
            log_data.update(train_data)
        if self.step_idx % self.target_update_freq == 0:
            self._update_target_network()
        self.step_idx += 1
        return log_data

    def get_loss(self):
        '''
        Samples batch from replay buffer computes DQN loss
        '''
        batch = self.replay_buffer.sample(batch_size=self.batch_size)
        obs = batch["observations"]
        act = batch["actions"]
        rew = batch["rewards"]
        next_obs = batch["next_observations"]
        terminated = batch["terminateds"]
        truncated = batch["truncateds"]

        # Get max state-action values for next states from target net
        next_q = self.target_model(ptu.from_numpy(next_obs)).max(dim=1).detach()
        next_q[terminated & ~truncated] = 0
        targets = rew + self.gamma * next_q

        # Get pred q_vals for current obs
        preds = self.model(ptu.from_numpy(obs))[torch.arange(preds.shape[0]), ptu.from_numpy(act)]
        
        # Calculate MSE
        losses = (targets - preds) ** 2
        loss = losses.mean()
        return loss

    def train(self):
        '''
        Computes loss on batch from replay buffer and updates model weights.
        Returns a dict containing loss metrics.
        '''
        self.model.train()
        loss = self.get_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'train_loss': loss.item()}