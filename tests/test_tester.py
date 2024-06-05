import pytest

import numpy as np
import torch

from aux_task_discovery.agents.gen_test.testers import TraceTester
from aux_task_discovery.models import MasterUserNetwork


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

def test_TraceTester_eval_tasks(model):
    tester = TraceTester(model, tau=0.05)
    utils = tester.eval_tasks(np.array([1.0,0.0]))
    assert np.allclose(utils, np.array([0.2,0.2,0.2]))
    utils = tester.eval_tasks(np.array([1.0,0.0]))
    assert np.allclose(utils, np.array([0.39,0.39,0.39]))
    utils = tester.eval_tasks(np.array([1.0,0.0]))
    assert np.allclose(utils, np.array([0.5705,0.5705,0.5705]))











