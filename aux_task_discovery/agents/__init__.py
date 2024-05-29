from .base import BaseAgent, ReplayBuffer
from .dqn import DQNAgent
from .gen_test.gen_test import GenTestAgent

AGENT_REG = {
    'dqn': DQNAgent,
    'gentest': GenTestAgent,
}

def get_agent(agent: str):
    assert agent in AGENT_REG, 'Given agent is not registered'
    return AGENT_REG[agent]
