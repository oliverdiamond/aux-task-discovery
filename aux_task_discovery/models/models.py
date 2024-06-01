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
        n_aux_tasks = 5, 
        hidden_size = 500,
        activation = 'tanh',
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_aux_tasks = n_aux_tasks
        self.n_heads = n_aux_tasks + 1 # Includes head for main task

        if hidden_size % self.n_heads != 0:
            print('Master-User hidden layer size not divisable by number of output heads. Extra hidden features will be delegated to the main task.')
        activation = _str_to_activation[activation]
        in_size = np.prod(input_shape)
        # Build shared representation
        self.shared_layer = nn.Sequential(
                                nn.Flatten(),
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
            def backward_hook(module, grad_input, grad_output):
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


    def forward(self, obs: torch.FloatTensor) -> Dict[Any, torch.FloatTensor]:
        '''
        Returns dictionary with head IDs as keys and their respective outputs as values. \n
        ID for the main task is 'main'. IDs for aux tasks are their indicies in MasterUserNetwork.aux_heads. \n
        '''
        shared_features = self.shared_layer(obs)
        outputs = {idx : head(shared_features) for idx, head in enumerate(self.aux_heads)}
        outputs['main'] = self.main_head(shared_features)
        return outputs
    
    def get_shared_features(self, obs: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Get features produced by the hidden layer which are shared across all task heads 
        for a given observation
        '''
        with torch.no_grad():
            shared_features = self.shared_layer(obs)
            return shared_features

    def reset_task_params(self, tasks: Sequence[int]):
        '''
        For each given task, resets the weights and biases for the shared features induced 
        by the task and resets the weights for those features in each of the output heads.
        '''
        # NOTE Currently use the default pytorch initialization methods for linear layers
        if isinstance(tasks, int):
            tasks = [tasks]
        with torch.no_grad():
            for task in tasks:
                # Get feature indicies for the task
                start, stop = self.feature_ranges[task]

                # Reset input weights for shared features induced by the task
                new_w = init.kaiming_uniform_(self.shared_layer[1].weight.clone(), a=math.sqrt(5))
                self.shared_layer[1].weight[start:stop,:] = new_w[start:stop,:]
                # Reset input bias for shared features induced by the task
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.shared_layer[1].weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                new_b = init.uniform_(self.shared_layer[1].bias.clone(), -bound, bound)
                self.shared_layer[1].bias[start:stop] = new_b[start:stop]

                # Reset output weights on the main task head for features induced by the task 
                dummy_weights = self.main_head.weight.clone()
                new_w = init.kaiming_uniform_(dummy_weights, a=math.sqrt(5))
                self.main_head.weight[:,start:stop] = new_w[:,start:stop]
                # Reset output weights on aux task heads for features induced by the task 
                for head in self.aux_heads:
                    new_w = init.kaiming_uniform_(dummy_weights, a=math.sqrt(5))
                    head.weight[:,start:stop] = new_w[:,start:stop]


