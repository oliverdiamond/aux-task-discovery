import pytest

import numpy as np

from aux_task_discovery.envs.gridworld import GridWorldEnv

'''
Tests for GridWorldEnv
'''
@pytest.mark.parametrize(
        "pos, expected",
        [
            ((2, 3), False),
            ((4, 4), False),
            ((1, 5), True),
            ((-1, 2), True),
        ]
)
def test_out_of_bounds(pos, expected):
    env = GridWorldEnv(size=5)
    assert env._out_of_bounds(pos) == expected


@pytest.mark.parametrize(
        "pos, expected",
        [
            ((2, 2), True),
            ((3, 3), True),
            ((1, 2), False),
            ((4, 4), False),
        ]
)
def test_is_obstacle(pos, expected):
    env = GridWorldEnv(size=5, 
                       obstacles=[(2, 2), (3, 3)])
    assert env._is_obstacle(pos) == expected

@pytest.mark.parametrize(
        "pos, expected",
        [
            ((1, 2), np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])),
            ((2, 2), np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])),
        ]
)
def test_get_obs(pos, expected):
    # Testing one-hot encoding of (x,y) pos
    env = GridWorldEnv(size=3)
    assert np.array_equal(env._get_obs(pos), expected)

@pytest.mark.parametrize(
        "pos, act, expected",
        [
            ((1, 1), 0, (0, 1)),
            ((1, 1), 1, (1, 1)),
            ((2, 2), 2, (2, 2)),
        ]
)
def test_get_new_pos(pos, act, expected):
    # Tests for env with no noise
    env = GridWorldEnv(size=3, 
                       obstacles=[(1, 0)])
    assert env._get_new_pos(pos, act) == expected

def test_get_new_pos_with_noise():
    # Test for env with noise
    env = GridWorldEnv(size=3, 
                       seed=42, 
                       action_noise=0.5, 
                       obstacles=[(1, 0)])
    assert env._get_new_pos((1, 1), 1) == (0, 1)

def test_reset():
    env = GridWorldEnv(size=3, 
                       start_pos=(1, 1))
    obs, _ = env.reset()
    # Test if agent position is reset to the starting position
    assert np.array_equal(obs, np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]))

def test_step():
    # Test for env with no noise
    env = GridWorldEnv(size=3, 
                       start_pos=(1, 1), 
                       goal_pos=(2, 2))
    env.reset()
    obs, rew, terminated, truncated, _ = env.step(3)
    assert np.array_equal(obs, np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]))
    assert rew == -1
    assert not terminated
    assert not truncated
    
    # Test move to goal state
    obs, rew, terminated, truncated, _ = env.step(2)
    assert np.array_equal(obs, np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert rew == -1
    assert terminated
    assert not truncated

def test_step_with_noise():
    # Tests for env with noise
    env = GridWorldEnv(size=3, 
                        start_pos=(1, 1), 
                        goal_pos=(2, 2), 
                        action_noise=0.5, 
                        seed=42,
                        obstacles=[(0,1)])
    env.reset()
    obs, rew, terminated, truncated, _ = env.step(3)
    assert np.array_equal(obs, np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]))
    assert rew == -1
    assert not terminated
    assert not truncated

    obs, rew, terminated, truncated, _ = env.step(2)
    assert np.array_equal(obs, np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]))
    assert rew == -1
    assert not terminated
    assert not truncated

    obs, rew, terminated, truncated, _ = env.step(3)
    assert np.array_equal(obs, np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]))
    assert rew == -1
    assert not terminated
    assert not truncated

    obs, rew, terminated, truncated, _ = env.step(3)
    assert np.array_equal(obs, np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert rew == -1
    assert terminated
    assert not truncated