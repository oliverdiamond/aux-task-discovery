from abc import ABC, abstractmethod

import numpy as np

from aux_task_discovery.models import MasterUserNetwork

class Tester(ABC):
    def __init__(self):
        pass
    def test_features(features: np.ndarray,
                      network: MasterUserNetwork):
        pass

