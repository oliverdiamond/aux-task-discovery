from .base import BaseAgent, ReplayBuffer
from .dqn import DQNAgent

AGENT_REG = {
    'dqn': DQNAgent,
}

def get_agent(agent: str):
    assert agent in AGENT_REG, 'Given agent is not registered'
    return AGENT_REG[agent]
