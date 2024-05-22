from abc import ABC, abstractmethod

import numpy as np

class Generator(ABC):
    def __init__(self,
                 seed=42,
                 observation):
        self.rand_gen = np.random.RandomState(seed)


    def generate_task(self):
        pass

    def generate_tasks(self, n):
        pass
