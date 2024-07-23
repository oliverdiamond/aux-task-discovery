import warnings 
from typing import Any, Dict, Sequence
import math

import torch
from torch import nn
from torch.nn import init
import numpy as np

import aux_task_discovery.utils.pytorch_utils as ptu

_str_to_activation = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "identity": nn.Identity(),
}

class ActionValueNetwork(nn.Module):
    '''
    Basic feed forward MLP for state-action values
    '''
    def __init__(
        self, 
        input_shape: tuple,
        n_actions: int,
        obs_bound = None,
        n_hidden = 1, 
        hidden_size = 500,
        activation = 'tanh',
    ):
        super().__init__()
        self.obs_bound = obs_bound
        # Build MLP
        activation = _str_to_activation[activation]
        layers = []
        # Normalize input if bounds are provided
        if obs_bound is not None:
            layers.append(MinMaxNormalization(feature_min=obs_bound[0], feature_max=obs_bound[1]))
        # Flatten obs to 1d tensor
        layers.append(nn.Flatten())
        # Add hidden layers
        in_size = np.prod(input_shape)
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(activation)
            in_size = hidden_size
        # Add output layer
        layers.append(nn.Linear(in_size, n_actions))
        self.net = nn.Sequential(*layers).to(ptu.device)
        # Initialize all weights and biases
        self.init_weights()

    def forward(self, obs: torch.FloatTensor):
        '''
        Returns Q-values for all actions
        '''
        return self.net(obs)

    @torch.no_grad()
    def init_weights(self):
        '''
        Initialize all network weights using Xavier uniform initialization and all biases to 0
        '''
        def linear_init(m):
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
        self.net.apply(linear_init)


class MasterUserNetwork(nn.Module):
    '''
    Multi-headed MLP for state-action values across any number of auxillery tasks. 
    Uses a single hidden layer as shared feature representation.
    '''
    def __init__(
        self, 
        input_shape: tuple,
        n_actions: int,
        obs_bound = None,
        n_aux_tasks = 5, 
        hidden_size = 500,
        activation = 'tanh',
    ):
        super().__init__()
        assert hidden_size >= n_aux_tasks+1, 'Hidden layer size must be >= n_aux_tasks+1'
        self.obs_bound = obs_bound
        self.hidden_size = hidden_size
        self.n_aux_tasks = n_aux_tasks
        self.n_heads = n_aux_tasks + 1 # Includes head for main task

        if hidden_size % self.n_heads != 0:
            print('Master-User hidden layer size not divisable by number of output heads. Extra hidden features will be delegated to the main task.')
        activation = _str_to_activation[activation]
        in_size = np.prod(input_shape)
        if self.obs_bound is not None:
            self.processing_layer = nn.Sequential(
                nn.Flatten(),
                MinMaxNormalization(feature_min=self.obs_bound[0], feature_max=self.obs_bound[1]),
                ).to(ptu.device)
        else:
            self.processing_layer = nn.Flatten().to(ptu.device)
        # Build shared representation
        self.shared_layer = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            activation,
            ).to(ptu.device)
        # Build output heads
        aux_heads = []
        self.feature_ranges = {} # Tracks start and stop idxs for shared features induced by each task
        start = 0
        stop = (hidden_size // self.n_heads) + (hidden_size % self.n_heads)
        for i in range(self.n_heads):
            # Sets gradient to 0 during backward pass for all hidden features not shaped by this output head
            def backward_hook(module, grad_input, grad_output, start=start, stop=stop):
                new_grad = torch.zeros_like(grad_input[0])
                new_grad[:,start:stop] = grad_input[0][:,start:stop]
                return (new_grad,)

            head = nn.Linear(hidden_size, n_actions)
            head.register_full_backward_hook(backward_hook)
            if i == 0:
                main_head = head
                self.feature_ranges['main'] = (start, stop)
            else:
                aux_heads.append(head)
                self.feature_ranges[i-1] = (start, stop)

            start = stop
            stop = start + (hidden_size // self.n_heads)
        
        self.main_head = main_head.to(ptu.device)
        self.aux_heads = nn.ModuleList(aux_heads).to(ptu.device)
        # Initialize all weights and biases
        self.init_weights()

    def forward(self, obs: torch.FloatTensor) -> Dict[Any, torch.FloatTensor]:
        '''
        Returns dictionary with head IDs as keys and their respective outputs as values. \n
        ID for the main task is 'main'. IDs for aux tasks are their indicies in MasterUserNetwork.aux_heads. \n
        '''
        obs = self.processing_layer(obs)
        shared_features = self.shared_layer(obs)
        outputs = {idx : head(shared_features) for idx, head in enumerate(self.aux_heads)}
        outputs['main'] = self.main_head(shared_features)
        return outputs
    
    @torch.no_grad()
    def init_weights(self):
        '''
        Initialize all network weights using Xavier uniform initialization and all biases to 0
        '''
        init.xavier_uniform_(self.shared_layer[0].weight)
        init.zeros_(self.shared_layer[0].bias)
        init.xavier_uniform_(self.main_head.weight)
        init.zeros_(self.main_head.bias)
        for i in np.arange(self.n_aux_tasks):
            init.xavier_uniform_(self.aux_heads[i].weight)
            init.zeros_(self.aux_heads[i].bias)
    
    @torch.no_grad()
    def get_shared_features(self, obs: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Get features produced by the hidden layer which are shared across all task heads 
        for a given observation
        '''
        obs = self.processing_layer(obs)
        shared_features = self.shared_layer(obs)
        return shared_features

    @torch.no_grad()
    def reset_task_params(self, tasks: Sequence[int]):
        '''
        For each given task, resets the weights and biases for the shared features induced 
        by the task and resets the weights for those features in each of the output heads.
        '''
        for task_idx in tasks:
            # Get feature indicies for the task
            start, stop = self.feature_ranges[task_idx]
            # Reset input weights for shared features induced by the task
            init.xavier_uniform_(self.shared_layer[0].weight[start:stop,:])
            # Reset output weights on the main task head for features induced by the task 
            init.xavier_uniform_(self.main_head.weight[:,start:stop])
            # Reset output weights on aux task heads for features induced by the task 
            for i in np.arange(self.n_aux_tasks):
                if i == task_idx:
                    init.xavier_uniform_(self.aux_heads[i].weight)
                else:
                    init.xavier_uniform_(self.aux_heads[i].weight[:,start:stop])

class MinMaxNormalization(nn.Module):
    def __init__(self, feature_min=0, feature_max=1, scaled_min=-1.0, scaled_max=1.0):
        super(MinMaxNormalization, self).__init__()
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.scaled_min = scaled_min
        self.scaled_max = scaled_max

    def forward(self, x):
        # Apply min-max normalization
        x_normalized = (x - self.feature_min) / (self.feature_max - self.feature_min)
        # Scale to the desired range [min_val, max_val]
        x_scaled = x_normalized * (self.scaled_max - self.scaled_min) + self.scaled_min
        return x_scaled