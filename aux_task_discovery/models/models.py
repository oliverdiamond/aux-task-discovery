import torch
from torch import nn
import numpy as np

import aux_task_discovery.utils.pytorch_utils as ptu

_str_to_activation = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}

class ActionValueNetwork(nn.Module):
    '''
    Basic feed forward MLP that outputs (state,action) values
    '''
    def __init__(
        self, 
        input_shape,
        n_actions,
        n_hidden = 1, 
        hidden_size = 500,
        activation = 'tanh',
    ):
        super().__init__()
        # Build MLP
        activation = _str_to_activation[activation]
        layers = []
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

    def forward(self, obs: torch.FloatTensor):
        '''
        Returns Q-values for all actions
        '''
        return self.net(obs)

class MasterUserNetwork(nn.Module):
    '''
    Multi-headed MLP for state-action values across any number of auxillery tasks.
    '''
    