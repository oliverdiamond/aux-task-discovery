import warnings 
from collections import OrderedDict

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
    Basic feed forward MLP for state-action values
    '''
    def __init__(
        self, 
        input_shape: tuple,
        n_actions: int,
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
    Uses a single hidden layer as shared feature representation.
    '''
    def __init__(
        self, 
        input_shape: tuple,
        n_actions: int,
        n_heads = 1, 
        hidden_size = 500,
        activation = 'tanh',
    ):
        super().__init__()
        if hidden_size % n_heads == 0:
            warnings.warn("""
                          Master-User hidden layer size not divisable by number of output heads.\n
                          Extra hidden features will be delegated to the main task. 
                          """)
        activation = _str_to_activation[activation]
        in_size = np.prod(input_shape)
        # Build shared representation
        self.shared_layer = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(in_size, hidden_size),
                                activation,
                            ).to(ptu.device)
        # Build output heads
        heads = {}
        start = 0
        stop = (hidden_size // n_heads) + (hidden_size % n_heads)
        for i in range(n_heads):
            # Sets gradient to 0 during backward pass for all hidden features not shaped by this output head
            def backward_hook(module, grad_input, grad_output):
                new_grad = torch.zeros_like(grad_input[0])
                new_grad[:,start:stop] = grad_input[0][:,start:stop]
                return (new_grad,)

            head = nn.Linear(hidden_size, n_actions)
            head.register_full_backward_hook(backward_hook)
            if i == 0:
                # Main task head
                heads['main'] = head
            else:
                # Aux task heads have keys 0 through n_aux_tasks-1
                heads[i-1] = head

            start = stop
            stop = start + (hidden_size // n_heads)
        
        self.output_heads = nn.ModuleDict(heads).to(ptu.device)

    def forward(self, obs: torch.FloatTensor):
        '''
        Returns dictionary with head IDs as keys and their respective outputs as values. \n
        ID for the main task is 'main'. IDs for aux tasks are 0 through n_aux_tasks-1. \n
        Also returns the shared representation layer under the key 'hidden_features'.
        '''
        hidden_features = self.shared_layer(obs)
        outputs = {key : head(hidden_features) for key, head in self.output_heads.items()}
        outputs['hidden_features'] = hidden_features
        return outputs





