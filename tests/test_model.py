import pytest
import copy

import numpy as np
import torch

from aux_task_discovery.models import ActionValueNetwork, MasterUserNetwork
import aux_task_discovery.utils.pytorch_utils as ptu

@pytest.fixture
def action_value_model():
    model = ActionValueNetwork(
        input_shape=(2,),
        n_actions=2,
        n_hidden=2,
        hidden_size=3,
        activation='identity',
    )
    with torch.no_grad():
        # Fill weights and biases for two hidden layers and output layer with 1s
        for i in [1,3,5]:
            model.net[i].weight.fill_(1.)
            model.net[i].bias.fill_(1.)

    return model
    
@pytest.mark.parametrize(
        "obs, expected",
        [
            (np.array([1.0,0.0]), np.array([22.0, 22.0])),
            (np.array([1.0,1.0]), np.array([31.0,31.0]))
        ]
)
def test_ActionValueNetwork_forward(action_value_model, obs, expected):
    input = ptu.from_numpy(obs).unsqueeze(0)
    output = ptu.to_numpy(action_value_model(input))[0]
    assert np.allclose(output, expected)

@pytest.fixture
def master_user_model():
    model = MasterUserNetwork(
        input_shape=(2,),
        n_actions=2,
        n_aux_tasks=2,
        hidden_size=4,
        activation='identity'
    )
    # Set all weights and biases to 1
    with torch.no_grad():
        model.shared_layer[1].weight.fill_(1.)
        model.shared_layer[1].bias.fill_(1.)

        model.main_head.weight[0].fill_(-1.)
        model.main_head.weight[1].fill_(1.)
        model.main_head.bias.fill_(1.)

        for head in model.aux_heads:
            head.weight.fill_(1.)
            head.bias.fill_(1.)
    
    return model

@pytest.mark.parametrize(
        "obs, main_expected, aux_expected",
        [
            (np.array([1.0,0.0]), np.array([-7.0, 9.0]), np.array([9.0, 9.0])),
            (np.array([1.0,1.0]), np.array([-11.0, 13.0]), np.array([13.0, 13.0])),
        ]
)
def test_MasterUserNetwork_forward(master_user_model, obs, main_expected, aux_expected):
    input = ptu.from_numpy(obs).unsqueeze(0)
    output = ptu.to_numpy(master_user_model(input))
    assert type(output) == dict
    assert np.allclose(output['main'], main_expected)
    assert np.allclose(output[0], aux_expected)
    assert np.allclose(output[1], aux_expected)

@pytest.mark.parametrize(
        "obs, expected",
        [
            (np.array([1.0,0.0]), np.array([2.0, 2.0, 2.0, 2.0])),
            (np.array([1.0,1.0]), np.array([3.0, 3.0, 3.0, 3.0])),
        ]
)
def test_MasterUserNetwork_get_shared_features(master_user_model, obs, expected):
    input = ptu.from_numpy(obs).unsqueeze(0)
    shared_features = ptu.to_numpy(master_user_model.get_shared_features(input))[0]
    assert np.allclose(shared_features, expected)

@pytest.mark.parametrize("tasks",[[0],[1]])
def test_MasterUserNetwork_reset_task_params(master_user_model, tasks):
    old_model = copy.deepcopy(master_user_model)
    master_user_model.reset_task_params(tasks)
    for task in tasks:
        start, stop = master_user_model.feature_ranges[task]
        
        # For each output head, check that weights for the features induced by the task have been reset.
        assert not torch.allclose(master_user_model.main_head.weight[:,start:stop], old_model.main_head.weight[:,start:stop])
        assert torch.allclose(master_user_model.main_head.weight[:,:start], old_model.main_head.weight[:,:start])
        assert torch.allclose(master_user_model.main_head.weight[:,stop:], old_model.main_head.weight[:,stop:])

        for idx in range(master_user_model.n_aux_tasks):
            assert not torch.allclose(master_user_model.aux_heads[idx].weight[:,start:stop], old_model.aux_heads[idx].weight[:,start:stop])
            assert torch.allclose(master_user_model.aux_heads[idx].weight[:,:start], old_model.aux_heads[idx].weight[:,:start])
            assert torch.allclose(master_user_model.aux_heads[idx].weight[:,stop:], old_model.aux_heads[idx].weight[:,stop:])
        
        # Check that input weights and biases for the features induced by the task have been reset
        assert not torch.allclose(master_user_model.shared_layer[1].weight[start:stop,:], old_model.shared_layer[1].weight[start:stop,:])
        assert torch.allclose(master_user_model.shared_layer[1].weight[:start,:], old_model.shared_layer[1].weight[:start,:])
        assert torch.allclose(master_user_model.shared_layer[1].weight[stop:,:], old_model.shared_layer[1].weight[stop:,:])

        #assert not torch.allclose(master_user_model.shared_layer[1].bias[start:stop], old_model.shared_layer[1].bias[start:stop])
        #assert torch.allclose(master_user_model.shared_layer[1].bias[:start], old_model.shared_layer[1].bias[:start])
        #assert torch.allclose(master_user_model.shared_layer[1].bias[stop:], old_model.shared_layer[1].bias[stop:])

@pytest.mark.parametrize("tasks",[[0,1]])
def test_MasterUserNetwork_reset_task_params_multiple(master_user_model, tasks):
    old_model = copy.deepcopy(master_user_model)
    master_user_model.reset_task_params(tasks)
    start, stop = 2, 4
    # For each output head, check that weights for the features induced by the task have been reset.
    assert not torch.allclose(master_user_model.main_head.weight[:,start:stop], old_model.main_head.weight[:,start:stop])
    assert torch.allclose(master_user_model.main_head.weight[:,:start], old_model.main_head.weight[:,:start])
    assert torch.allclose(master_user_model.main_head.weight[:,stop:], old_model.main_head.weight[:,stop:])

    for idx in range(master_user_model.n_aux_tasks):
        assert not torch.allclose(master_user_model.aux_heads[idx].weight[:,start:stop], old_model.aux_heads[idx].weight[:,start:stop])
        assert torch.allclose(master_user_model.aux_heads[idx].weight[:,:start], old_model.aux_heads[idx].weight[:,:start])
        assert torch.allclose(master_user_model.aux_heads[idx].weight[:,stop:], old_model.aux_heads[idx].weight[:,stop:])
    
    # Check that input weights and biases for the features induced by the task have been reset
    assert not torch.allclose(master_user_model.shared_layer[1].weight[start:stop,:], old_model.shared_layer[1].weight[start:stop,:])
    assert torch.allclose(master_user_model.shared_layer[1].weight[:start,:], old_model.shared_layer[1].weight[:start,:])
    assert torch.allclose(master_user_model.shared_layer[1].weight[stop:,:], old_model.shared_layer[1].weight[stop:,:])

    #TODO Update when you know if biases should be reset for shared layer
    #assert not torch.allclose(master_user_model.shared_layer[1].bias[start:stop], old_model.shared_layer[1].bias[start:stop])
    #assert torch.allclose(master_user_model.shared_layer[1].bias[:start], old_model.shared_layer[1].bias[:start])
    #assert torch.allclose(master_user_model.shared_layer[1].bias[stop:], old_model.shared_layer[1].bias[stop:])

