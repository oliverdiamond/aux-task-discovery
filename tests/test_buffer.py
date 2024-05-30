import pytest

import numpy as np

from aux_task_discovery.agents.base import ReplayBuffer

'''
Tests for basic replay buffer
'''
@pytest.fixture
def data():
    return {
        'obs' : np.array([1.0, 2.0, 3.0]),
        'act' : 1,
        'rew' : 1.0,
        'next_obs' : np.array([1.1, 2.1, 3.1]),
        'terminated' : False,
        'truncated' : False,
    }

def test_insert(data):
    buffer = ReplayBuffer()
    buffer.insert(
        data['obs'], 
        data['act'], 
        data['rew'], 
        data['next_obs'], 
        data['terminated'], 
        data['truncated'])
    assert len(buffer) == 1, "Size should be 1 after one insert."
    buffer.insert(
        data['obs'], 
        data['act'], 
        data['rew'], 
        data['next_obs'], 
        data['terminated'], 
        data['truncated'])
    assert len(buffer) == 2, "Size should be 2 after two inserts."

def test_capacity(data):
    capacity = 4
    buffer = ReplayBuffer(capacity=capacity)
    for i in range(capacity + 2):
        act = i
        buffer.insert(
            data['obs'], 
            act, 
            data['rew'], 
            data['next_obs'], 
            data['terminated'], 
            data['truncated'])
    assert len(buffer) == capacity, "Size should be equal to capacity when overfilled."
    assert buffer.actions[0] == i-1, 'First action should be overwritten'
    assert buffer.actions[1] == i, 'Second action should be overwritten'

def test_sample(data):
    buffer = ReplayBuffer(seed=42)
    for i in range(10):
        act = i
        buffer.insert(
            data['obs'], 
            act, 
            data['rew'], 
            data['next_obs'], 
            data['terminated'], 
            data['truncated'])
    batch = buffer.sample(5)
    assert batch["observations"].shape == (5, 3), "Observations batch size mismatch."
    assert batch["actions"].shape == (5,), "Actions batch size mismatch."
    assert batch["rewards"].shape == (5,), "Rewards batch size mismatch."
    assert batch["next_observations"].shape == (5, 3), "Next observations batch size mismatch."
    assert batch["terminateds"].shape == (5,), "Terminateds batch size mismatch."
    assert batch["truncateds"].shape == (5,), "Truncateds batch size mismatch."
    assert np.array_equal(batch["actions"], np.array([6, 3, 7, 4, 6]))
