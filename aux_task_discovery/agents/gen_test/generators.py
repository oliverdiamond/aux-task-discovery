from typing import Sequence

import numpy as np

from aux_task_discovery.agents.gen_test.gvf import GVF, SubgoalGVF
from aux_task_discovery.envs import FourRoomsEnv

class Generator():
    def __init__(self, seed, **kwargs):
        self.rand_gen = np.random.RandomState(seed)

    def generate_tasks(self, n_tasks) -> Sequence[GVF]:
        raise NotImplementedError()

class OneHotGenerator(Generator):
    '''
    Generates subgoal-reaching GVFs for one-hot encoded states.
    Gamma is 0 for subgoal and 1 for all other states.
    Cumulant is -1 for all states. 
    '''
    def __init__(self, env, seed=42, **kwargs):
        super().__init__(seed=seed)
        self.input_size = np.prod(env.observation_space.shape)

    def generate_task(self):
        idx = self.rand_gen.randint(self.input_size)
        subgoal = np.zeros(self.input_size, dtype=np.float32)
        subgoal[idx] = 1
        return SubgoalGVF(subgoal)
    
    def generate_tasks(self, n_tasks) -> Sequence[GVF]:
        return [self.generate_task() for _ in range(n_tasks)]

class FourroomsCornerGenerator(Generator):
    '''
    Generates the two corner subgoal tasks for the fourrooms 
    environment from Gen + Test Paper
    '''
    def __init__(self, input_shape=(2,), **kwargs):
        self.dummy_env = FourRoomsEnv()
        assert input_shape == self.dummy_env.observation_space.shape, 'Provided input shape does not match fourrroms env observation space'
        self.input_shape = input_shape
        self.tasks = None

    def generate_tasks(self, n_tasks=2) -> Sequence[GVF]:
        assert n_tasks == 2, 'Corner generator for fourrooms environment only generates two subgoals'
        if self.tasks is not None:
            raise Exception('Corner tasks should not be reset after initialization')
        corner1 = np.zeros(self.input_shape, dtype=np.float32)
        corner1[0] = 1
        corner2 = np.zeros(self.input_shape, dtype=np.float32)
        corner2[2] = 1
        corner1_task = SubgoalGVF(corner1)
        corner2_task = SubgoalGVF(corner2)
        self.tasks = [corner1_task, corner2_task]
        return self.tasks

class FourroomsHallwayGenerator(Generator):
    '''
    Generates the two corner subgoal tasks for the fourrooms 
    environment from Gen + Test Paper
    '''
    def __init__(self, input_shape=2, **kwargs):
        self.dummy_env = FourRoomsEnv()
        assert input_shape == self.dummy_env.observation_space.shape, 'Provided input shape does not match fourrroms env observation space'
        self.input_shape = input_shape
        self.tasks = None

    def generate_tasks(self, n_tasks=2) -> Sequence[GVF]:
        assert n_tasks == 2, 'Hallway generator for fourrooms environment only generates two subgoals'
        if self.tasks is not None:
            raise Exception('Hallway tasks should not be reset after initialization')
        hallway1 = np.zeros(self.input_shape, dtype=np.float32)
        hallway1[33] = 1
        hallway2 = np.zeros(self.input_shape, dtype=np.float32)
        hallway2[38] = 1
        hallway1_task = SubgoalGVF(hallway1)
        hallway2_task = SubgoalGVF(hallway2)
        self.tasks = [hallway1_task, hallway2_task]
        return self.tasks

#TODO Implement FeatureAttainGenerator
class FeatureAttainGenerator(Generator):
    pass

GENERATOR_REG = {
    'onehot': OneHotGenerator,
    'feature': FeatureAttainGenerator,
    'fourrooms_corner': FourroomsCornerGenerator,
    'fourrooms_hallway': FourroomsHallwayGenerator,
}

def get_generator(generator: str):
    assert generator in GENERATOR_REG, 'Given generator is not registered'
    return GENERATOR_REG[generator]