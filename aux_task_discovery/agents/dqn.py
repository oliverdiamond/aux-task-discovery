import copy
from typing import Union

import numpy as np
import torch
from torch import optim

from aux_task_discovery.utils import random_argmax
import aux_task_discovery.utils.pytorch_utils as ptu
from aux_task_discovery.agents.base import BaseAgent, ReplayBuffer
from aux_task_discovery.models import ActionValueNetwork


class DQNAgent(BaseAgent):
    '''
    DQN Agent with FF Q-network and Adam optimizer
    '''
    def __init__(
        self,
        input_shape: tuple,
        n_actions: int,
        seed = 42,
        learning_rate = 0.0001, 
        epsilon = 0.1,
        epsilon_final = 0.1,
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
        super().__init__(seed=seed)
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.anneal_epsilon = anneal_epsilon
        self.n_anneal = n_anneal
        self.gamma = gamma
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.activation = activation
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        self.learning_start = learning_start
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, seed=seed)
        self._setup_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self._update_target_network()

    def _update_target_network(self):
        self.target_model = copy.deepcopy(self.model)

    def _setup_model(self):
        self.model = ActionValueNetwork(
                input_shape=self.input_shape,
                n_actions=self.n_actions,
                n_hidden=self.n_hidden,
                hidden_size=self.hidden_size,
                activation=self.activation,
                )

    @torch.no_grad()
    def get_action(self, obs: np.ndarray):
        '''
        Selects e-greedy action using q-values from model
        '''
        if self.rand_gen.rand() < self.epsilon:
            return self.rand_gen.randint(0, self.n_actions)
        obs = ptu.from_numpy(obs).unsqueeze(0)
        q_vals = ptu.to_numpy(self.model(obs))[0]
        act = random_argmax(q_vals)
        return act

    def _step(
        self, 
        obs: np.ndarray, 
        act: int,
        rew: Union[float, int], 
        next_obs: np.ndarray, 
        terminated: bool,
        truncated: bool
    ) -> dict:
        '''
        Adds a single transition to the replay buffer and updates model weights 
        and algorithm parameters as nessesary. Returns dict for logging.
        '''
        log_info = {}
        self.replay_buffer.insert(obs, act, rew, next_obs, terminated, truncated)
        if self.anneal_epsilon and self.step_idx <= self.n_anneal:
            # Linearly decreases epsilon from init value to 0.1 over n_anneal steps
            self.epsilon -= (self.epsilon-self.epsilon_final)/self.n_anneal
            log_info['epsilon'] = self.epsilon
        if self.step_idx >= self.learning_start and self.step_idx % self.update_freq == 0:
            train_info = self.train()
            log_info.update(train_info)
        if self.step_idx % self.target_update_freq == 0:
            self._update_target_network()
        return log_info

    def get_losses(self, batch: dict):
        '''
        Computes squarred TD error for each transition in batch.
        '''
        obs = batch["observations"]
        act = batch["actions"]
        rew = batch["rewards"]
        next_obs = batch["next_observations"]
        terminated = batch["terminateds"]
        truncated = batch["truncateds"]

        # Get max state-action values for next states from target net
        next_qs = self.target_model(ptu.from_numpy(next_obs)).max(dim=-1)[0].detach()
        next_qs[terminated & ~truncated] = 0
        targets = ptu.from_numpy(rew) + (self.gamma * next_qs)

        # Get pred q_vals for current obs
        preds = self.model(ptu.from_numpy(obs))[torch.arange(obs.shape[0]), ptu.from_numpy(act)]
        
        # Calculate squared error for each transition
        losses = (targets - preds) ** 2
        return losses

    def train(self):
        '''
        Computes mean loss on batch from replay buffer and updates model weights.
        Returns a dict containing loss metrics.
        '''
        self.model.train()
        batch = self.replay_buffer.sample(batch_size=self.batch_size)
        losses = self.get_losses(batch)
        loss = losses.mean()
        loss_info = {'DQN_loss': loss.item()}
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_info