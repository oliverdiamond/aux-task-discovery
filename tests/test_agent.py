import pytest

import numpy as np
import torch

import aux_task_discovery.utils.pytorch_utils as ptu
from aux_task_discovery.agents import DQNAgent, GenTestAgent

@pytest.fixture
def dqn_agent():
    agent = DQNAgent(
        input_shape=(2,),
        n_actions=2,
        seed=42,
        learning_rate=0.01,
        epsilon=0.1,
        epsilon_final=0.1,
        anneal_epsilon=False,
        n_anneal=10000,
        gamma=0.9,
        n_hidden=1,
        hidden_size=2,
        activation='identity',
        buffer_size=1000,
        batch_size=2,
        update_freq=1,
        target_update_freq=2,
        learning_start=2
    )
    return agent

@pytest.fixture
def data():
    return {
        'obs' : np.array([1.0, 0.0]),
        'act' : 0,
        'rew' : 1.0,
        'next_obs' : np.array([0.0, 1.0]),
        'terminated' : False,
        'truncated' : False,
    }

def test_DQNAgent_(dqn_agent):

