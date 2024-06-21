import pytest

import numpy as np
import torch

from aux_task_discovery.agents.gen_test.testers import TraceTester, BatchTraceTester
from aux_task_discovery.models import MasterUserNetwork

@pytest.fixture
def batch():
    return {
        'observations' : np.array([[1.0, 0.0],[2.0, 1.0]]),
        'actions' : np.array([0, 1]),
        'rewards' : np.array([1.0, 1.0]),
        'next_observations' : np.array([[1.0, 1.0],[1.0, 1.0]]),
        'terminateds' : np.array([False, True]),
        'truncateds' : np.array([False, True]),
    }

@pytest.fixture
def model():
    model = MasterUserNetwork(
        input_shape=(2,),
        n_actions=2,
        n_aux_tasks=3,
        hidden_size=4,
        activation='identity'
    )
    with torch.no_grad():
        model.shared_layer[1].weight[0:2].fill_(1.)
        model.shared_layer[1].weight[2:4].fill_(-1.)
        model.shared_layer[1].bias[0:2].fill_(1.)
        model.shared_layer[1].bias[2:4].fill_(-1.)


        model.main_head.weight[0].fill_(-1.)
        model.main_head.weight[1].fill_(1.)
        model.main_head.bias.fill_(1.)

        for head in model.aux_heads:
            head.weight.fill_(1.)
            head.bias.fill_(1.)
    
    return model

def test_BatchTraceTester_eval_tasks(model, batch):
    tester = BatchTraceTester(model, tau=0.05)
    utils = tester.eval_tasks(batch)
    assert np.allclose(utils, np.array([0.3,0.3,0.3]))
    assert np.allclose(tester.trace, np.array([[0.1,0.1,0.1,0.1],[0.2,0.2,0.2,0.2]]))
    utils = tester.eval_tasks(batch)
    assert np.allclose(utils, np.array([0.585,0.585,0.585]))
    assert np.allclose(tester.trace, np.array([[0.195,0.195,0.195,0.195],[0.39,0.39,0.39,0.39]]))

def test_BatchTraceTester_reset_tasks(model, batch):
    tester = BatchTraceTester(model, tau = 0.05)
    tester.eval_tasks(batch)
    tester.reset_tasks([0,2])
    assert np.allclose(tester.trace, np.array([[0.1,0.15,0.1,0.15],[0.2,0.15,0.2,0.15]]))

def test_TraceTester_eval_tasks(model):
    tester = TraceTester(model, tau=0.05)
    utils = tester.eval_tasks(np.array([1.0,0.0]))
    assert np.allclose(utils, np.array([0.2,0.2,0.2]))
    assert np.allclose(tester.trace, np.array([0.1,0.1,0.1,0.1]))
    utils = tester.eval_tasks(np.array([1.0,0.0]))
    assert np.allclose(utils, np.array([0.39,0.39,0.39]))
    assert np.allclose(tester.trace, np.array([0.195,0.195,0.195,0.195]))

def test_TraceTester_reset_tasks(batch):
    model = MasterUserNetwork(
        input_shape=(2,),
        n_actions=2,
        n_aux_tasks=3,
        hidden_size=4,
        activation='identity'
    )
    with torch.no_grad():
        model.shared_layer[1].weight[0:3].fill_(1.)
        model.shared_layer[1].weight[3].fill_(-2.)
        model.shared_layer[1].bias[0:3].fill_(1.)
        model.shared_layer[1].bias[3].fill_(-2.)


        model.main_head.weight[0].fill_(-1.)
        model.main_head.weight[1].fill_(1.)
        model.main_head.bias.fill_(1.)

        for head in model.aux_heads:
            head.weight.fill_(1.)
            head.bias.fill_(1.)
    tester = TraceTester(model, tau = 0.05)
    tester.eval_tasks(observation=batch['observations'][0])
    tester.reset_tasks([0])
    assert np.allclose(tester.trace, np.array([0.1,0.15,0.1,0.2]))








