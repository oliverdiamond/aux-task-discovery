import pytest

import numpy as np
import torch

import aux_task_discovery.utils.pytorch_utils as ptu
from aux_task_discovery.agents import DQNAgent, GenTestAgent


@pytest.fixture
def dqn_batch():
    return {
        'observations' : np.array([[1.0, 0.0],[1.0, 0.0],[1.0, 1.0]]),
        'actions' : np.array([0, 1, 1]),
        'rewards' : np.array([1.0, 1.0, 1.0]),
        'next_observations' : np.array([[1.0, 1.0],[1.0, 1.0],[1.0, 0.0]]),
        'terminateds' : np.array([False, True, True]),
        'truncateds' : np.array([False, True, False]),
    }

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
        target_update_freq=3,
        learning_start=2
    )
    # Setting weights and biases for model and target model to specific values for testing
    with torch.no_grad():
        agent.model.net[1].weight.fill_(1.)
        agent.model.net[1].bias.fill_(0.)
        agent.model.net[3].weight[0].fill_(0.)
        agent.model.net[3].weight[1].fill_(1.)
        agent.model.net[3].bias.fill_(0.)
    agent._update_target_network()
    return agent

@pytest.mark.parametrize(
        "epsilon, expected",
        [
            (0.1, 1),
            (0.5, 0),
        ]
)
def test_DQNAgent_get_action(epsilon, expected):
    agent = DQNAgent(
        input_shape=(2,),
        n_actions=2,
        seed=42,
        epsilon=epsilon,
        n_hidden=1,
        hidden_size=2,
        activation='identity',
    )
    with torch.no_grad():
        agent.model.net[1].weight.fill_(1.)
        agent.model.net[1].bias.fill_(0.)
        agent.model.net[3].weight[0].fill_(0.)
        agent.model.net[3].weight[1].fill_(1.)
        agent.model.net[3].bias.fill_(0.)
    obs = np.array([1.0, 0.0])
    # q_vals[0] will be 0, q_vals[1] will be 2
    # With seed set to 42, self.rand_gen.rand() will return 0.3745401188473625 
    # So if epsilon > 0.37, then randomly selected action should be 0, otherwise greedy action is 1
    action = agent.get_action(obs)
    assert action == expected

def test_DQNAgent_get_losses(dqn_agent, batch):
    losses = dqn_agent.get_losses(batch)
    assert np.allclose(ptu.to_numpy(losses), np.array([21.16, 6.76, 9]))

def test_DQNAgent_train(dqn_agent, batch):
    dqn_agent.replay_buffer.insert(
        batch['observations'][0], 
        batch['actions'][0], 
        batch['rewards'][0], 
        batch['next_observations'][0], 
        batch['terminateds'][0], 
        batch['truncateds'][0]
        )
    dqn_agent.replay_buffer.insert(
        batch['observations'][1], 
        batch['actions'][1], 
        batch['rewards'][1], 
        batch['next_observations'][1], 
        batch['terminateds'][1], 
        batch['truncateds'][1]
        )
    loss_info = dqn_agent.train()
    # With seed set to 42 and batch size set to 2, agent should sample both the above transitions
    assert np.isclose(loss_info['DQN_loss'], 13.96)
    # Check gradients for output layer
    assert np.allclose(
                ptu.to_numpy(dqn_agent.model.net[3].weight.grad), 
                np.array([[-4.6, -4.6],[-2.6, -2.6]])
            )
    assert np.allclose(
                ptu.to_numpy(dqn_agent.model.net[3].bias.grad), 
                np.array([-4.6, -2.6])
            )
    # Check gradients for hidden layer
    assert np.allclose(
                ptu.to_numpy(dqn_agent.model.net[1].weight.grad), 
                np.array([[-2.6,0.0],[-2.6,0.0]])
            )
    assert np.allclose(
                ptu.to_numpy(dqn_agent.model.net[1].bias.grad), 
                np.array([-2.6, -2.6])
            )
    
def test_DQNAgent_step(dqn_agent, batch):
    assert dqn_agent.step_idx == 1
    info = dqn_agent.step(
            batch['observations'][0], 
            batch['actions'][0], 
            batch['rewards'][0], 
            batch['next_observations'][0], 
            batch['terminateds'][0], 
            batch['truncateds'][0]
            )
    assert dqn_agent.step_idx == 2
    info = dqn_agent.step(
            batch['observations'][1], 
            batch['actions'][1], 
            batch['rewards'][1], 
            batch['next_observations'][1], 
            batch['terminateds'][1], 
            batch['truncateds'][1]
            )
    assert dqn_agent.step_idx == 3
    assert np.isclose(info['DQN_loss'], 13.96)
    # Check gradients for output layer
    assert np.allclose(
                ptu.to_numpy(dqn_agent.model.net[3].weight.grad), 
                np.array([[-4.6, -4.6],[-2.6, -2.6]])
            )
    assert np.allclose(
                ptu.to_numpy(dqn_agent.model.net[3].bias.grad), 
                np.array([-4.6, -2.6])
            )
    # Check gradients for hidden layer
    assert np.allclose(
                ptu.to_numpy(dqn_agent.model.net[1].weight.grad), 
                np.array([[-2.6,0.0],[-2.6,0.0]])
            )
    assert np.allclose(
                ptu.to_numpy(dqn_agent.model.net[1].bias.grad), 
                np.array([-2.6, -2.6])
            )
    # Check that target network has not been updated
    assert not torch.allclose(
                dqn_agent.model.net[3].weight, 
                dqn_agent.target_model.net[3].weight, 
            )
    assert not torch.allclose(
                dqn_agent.model.net[3].bias, 
                dqn_agent.target_model.net[3].bias, 
            )
    assert not torch.allclose(
                dqn_agent.model.net[1].weight, 
                dqn_agent.target_model.net[1].weight, 
            )
    assert not torch.allclose(
                dqn_agent.model.net[1].bias, 
                dqn_agent.target_model.net[1].bias, 
            )
    info = dqn_agent.step(
            batch['observations'][1], 
            batch['actions'][1], 
            batch['rewards'][1], 
            batch['next_observations'][1], 
            batch['terminateds'][1], 
            batch['truncateds'][1]
            )
    # Check that target network has been updated
    assert torch.allclose(
                dqn_agent.model.net[3].weight, 
                dqn_agent.target_model.net[3].weight, 
            )
    assert torch.allclose(
                dqn_agent.model.net[3].bias, 
                dqn_agent.target_model.net[3].bias, 
            )
    assert torch.allclose(
                dqn_agent.model.net[1].weight, 
                dqn_agent.target_model.net[1].weight, 
            )
    assert torch.allclose(
                dqn_agent.model.net[1].bias, 
                dqn_agent.target_model.net[1].bias, 
            )

@pytest.fixture
def gentest_agent():
    agent = GenTestAgent(
        input_shape = (2,),
        n_actions = 2,
        generator = 'onehot',
        tester = 'trace',
        n_aux_tasks = 2,
        age_threshold = 0,
        replace_cycle = 3,
        replace_ratio = 0.5,
        tester_tau = 0.05,
        seed = 42,
        learning_rate = 0.01, 
        epsilon = 0.1,
        epsilon_final = 0.1,
        anneal_epsilon = False,
        n_anneal = 10000,
        gamma = 0.9,
        hidden_size = 3,
        activation = 'identity',
        buffer_size = 1000,
        batch_size = 2,
        update_freq = 1,
        target_update_freq=2,
        learning_start = 2,
    )
    with torch.no_grad():
        agent.model.shared_layer[1].weight.fill_(1.)
        agent.model.shared_layer[1].bias.fill_(0.)
        agent.model.main_head.weight[0].fill_(0.)
        agent.model.main_head.weight[1].fill_(1.)
        agent.model.main_head.bias.fill_(0.)
        for i in range(agent.model.n_aux_tasks):
            agent.model.aux_heads[i].weight[0].fill_(0.)
            agent.model.aux_heads[i].weight[1].fill_(1.)
            agent.model.aux_heads[i].bias.fill_(0.)
    agent._update_target_network()
    return agent

@pytest.fixture
def gentest_batch():
    return {
        'observations' : np.array([[1.0, 0.0],[1.0, 0.0],[0.0, 1.0]]),
        'actions' : np.array([0, 1, 1]),
        'rewards' : np.array([1.0, 2.0, 1.0]),
        'next_observations' : np.array([[0.0, 1.0],[0.0, 1.0],[1.0, 0.0]]),
        'terminateds' : np.array([False, True, True]),
        'truncateds' : np.array([False, False, True]),
    }

@pytest.mark.parametrize(
        "epsilon, expected",
        [
            (0.1, 1),
            (0.5, 0),
        ]
)
def test_GenTestAgent_get_action(epsilon, expected):
    agent = GenTestAgent(
        input_shape = (2,),
        n_actions = 2,
        seed = 42,
        generator = 'onehot',
        tester = 'trace',
        epsilon = epsilon,
        hidden_size = 2,
        activation = 'identity',
    )
    with torch.no_grad():
        agent.model.shared_layer[1].weight.fill_(1.)
        agent.model.shared_layer[1].bias.fill_(0.)
        agent.model.main_head.weight[0].fill_(0.)
        agent.model.main_head.weight[1].fill_(1.)
        agent.model.main_head.bias.fill_(0.)
    obs = np.array([1.0, 0.0])
    # q_vals[0] will be 0, q_vals[1] will be 2
    # With seed set to 42, self.rand_gen.rand() will return 0.3745401188473625 
    # So if epsilon > 0.37, then randomly selected action should be 0, otherwise greedy action is 1
    action = agent.get_action(obs)
    assert action == expected

def test_GenTestAgent_update_tasks():
    gentest_agent = GenTestAgent(
        input_shape = (3,),
        n_actions = 2,
        generator = 'onehot',
        tester = 'trace',
        n_aux_tasks = 4,
        age_threshold = 1,
        replace_cycle = 3,
        replace_ratio = 0.5,
        tester_tau = 0.05,
        seed = 42,
        learning_rate = 0.01, 
        epsilon = 0.1,
        epsilon_final = 0.1,
        anneal_epsilon = False,
        n_anneal = 10000,
        gamma = 0.9,
        hidden_size = 2,
        activation = 'identity',
        buffer_size = 1000,
        batch_size = 2,
        update_freq = 1,
        target_update_freq=2,
        learning_start = 2,
    )
    assert gentest_agent.n_replace == 2
    assert gentest_agent.tasks[0].gamma(np.array([0.0, 0.0, 1.0])) == 0.0
    assert gentest_agent.tasks[1].gamma(np.array([1.0, 0.0, 0.0])) == 0.0
    assert gentest_agent.tasks[2].gamma(np.array([0.0, 0.0, 1.0])) == 0.0
    assert gentest_agent.tasks[2].gamma(np.array([1.0, 0.0, 0.0])) == 1.0
    assert gentest_agent.tasks[3].gamma(np.array([0.0, 0.0, 1.0])) == 0.0
    assert gentest_agent.tasks[3].gamma(np.array([1.0, 0.0, 0.0])) == 1.0
    gentest_agent.task_ages = np.array([1,2,2,2])
    gentest_agent.task_utils = np.array([1, 10, 3, 4])
    gentest_agent.update_tasks()
    # First task should not be updated due to age
    assert gentest_agent.tasks[0].gamma(np.array([0.0, 0.0, 1.0])) == 0.0
    # Second task should not be updated due to higher utility than third and fourth tasks
    assert gentest_agent.tasks[1].gamma(np.array([1.0, 0.0, 0.0])) == 0.0
    # Third and fourth tasks should be updated to new tasks
    assert gentest_agent.tasks[2].gamma(np.array([0.0, 0.0, 1.0])) == 1.0
    assert gentest_agent.tasks[2].gamma(np.array([1.0, 0.0, 0.0])) == 0.0
    assert gentest_agent.tasks[3].gamma(np.array([0.0, 0.0, 1.0])) == 1.0
    assert gentest_agent.tasks[3].gamma(np.array([1.0, 0.0, 0.0])) == 0.0
    assert np.allclose(gentest_agent.task_ages, np.array([1,2,0,0]))

def test_GenTestAgent_get_losses(gentest_agent, batch):
    pass

def test_GenTestAgent_train(gentest_agent):
    pass
