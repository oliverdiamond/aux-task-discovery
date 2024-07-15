from typing import Sequence


import numpy as np

from aux_task_discovery.agents.gen_test.gvf import GVF, SubgoalGVF
from aux_task_discovery.envs import FourRoomsEnv, GridWorldEnv

class Generator():
    def __init__(self, seed, **kwargs):
        self.rand_gen = np.random.RandomState(seed)

    def generate_tasks(self, n_tasks) -> Sequence[GVF]:
        raise NotImplementedError()

class GridSubgoalGenerator(Generator):
    '''
    Generates subgoal-reaching GVFs for a gridworld env.
    Gamma is 0 for subgoal and 1 for all other states.
    Cumulant is -1 for all states. 
    '''
    def __init__(self, env: GridWorldEnv, seed=42, **kwargs):
        super().__init__(seed=seed)
        self.env = env
        self.obstacle_idxs = [np.argmax(env.unwrapped.pos_to_onehot(ob)) for ob in env.unwrapped.obstacles]

    def generate_tasks(self, n_tasks, curr_tasks = None):
        '''
        Generates n_tasks subgoal-reaching GVFs for the gridworld environment.
        Subgoals are randomly selected without replacement from all valid states 
        (exludes env obstacles and subgoals for an optionally provided list of current tasks).
        '''
        # Create list of idxs for one-hot state encoding to exclude from subgoal selection
        exempt_subgoals = self.obstacle_idxs.copy()
        if curr_tasks is not None:
            exempt_subgoals += [np.argmax(task.subgoal) for task in curr_tasks]
        # Randomly select n_tasks valid subgoals
        valid_subgoals = np.setdiff1d(np.arange(self.env.unwrapped.n_states), exempt_subgoals)
        new_subgoal_idxs = np.random.choice(valid_subgoals, n_tasks, replace=False)
        # Create one-hot vectors for new subgoals
        new_subgoals = np.zeros((n_tasks, self.env.unwrapped.observation_space.shape[0]), dtype=np.float32)
        new_subgoals[np.arange(n_tasks), new_subgoal_idxs] = 1
        return [SubgoalGVF(subgoal) for subgoal in new_subgoals]

class FourroomsCornerGenerator(Generator):
    '''
    Generates the two corner subgoal tasks for the fourrooms 
    environment from Gen + Test Paper
    '''
    def __init__(self, **kwargs):
        self.input_shape = FourRoomsEnv().observation_space.shape
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
    def __init__(self, **kwargs):
        self.input_shape = FourRoomsEnv().observation_space.shape
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
    'grid_subgoal': GridSubgoalGenerator,
    'feature': FeatureAttainGenerator,
    'fourrooms_corner': FourroomsCornerGenerator,
    'fourrooms_hallway': FourroomsHallwayGenerator,
}

def get_generator(generator: str):
    assert generator in GENERATOR_REG, 'Given generator is not registered'
    return GENERATOR_REG[generator]