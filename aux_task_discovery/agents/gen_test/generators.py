from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Sequence

import numpy as np

from aux_task_discovery.agents.gen_test.gvf import GVF
from aux_task_discovery.envs import FourRoomsEnv

class Generator(ABC):
    def __init__(self, seed, **kwargs):
        self.rand_gen = np.random.RandomState(seed)

    def generate_tasks(self, n_tasks) -> Sequence[GVF]:
        return [self.generate_task() for _ in range(n_tasks)]

    @abstractmethod
    def generate_task(self) -> GVF:
        raise NotImplementedError()


class OneHotGenerator(Generator):
    '''
    Generates subgoal-reaching GVFs for one-hot encoded states.
    Gamma is 0 for subgoal and 1 for all other states.
    Cumulant is -1 for all states. 
    '''
    def __init__(self, input_shape: tuple, seed=42, **kwargs):
        super().__init__(seed=seed)
        self.input_size = np.prod(input_shape)

    def generate_task(self) -> GVF:
        idx = self.rand_gen.randint(self.input_size)
        subgoal = np.zeros(self.input_size)
        subgoal[idx] = 1
        cumulant = lambda obs : -1
        gamma = lambda obs : int(not np.allclose(obs,subgoal))
        return GVF(cumulant, gamma)

class FeatureAttainGenerator(Generator):
    pass

class FourroomsCornerGenerator(Generator):
    pass

class FourroomsHallwayGenerator(Generator):
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



#------------------------TESTS------------------------#
def test_OneHotGenerator():
    generator = OneHotGenerator(4, seed=42, random_arg=4)
    gvfs = generator.generate_tasks(2)
    assert len(gvfs) == 2
    assert gvfs[0].cumulant(np.array([0,0,0,0])) == -1
    assert gvfs[0].cumulant(np.array([0,0,1,0])) == -1
    assert gvfs[0].gamma(np.array([0,0,0,0])) == 1
    assert gvfs[0].gamma(np.array([0,0,1,0])) == 0
    assert gvfs[1].gamma(np.array([0,0,1,0])) == 1
    assert gvfs[1].gamma(np.array([0,0,0,1])) == 0

def run_tests():
    test_OneHotGenerator()

if __name__ == "__main__":
    run_tests()




