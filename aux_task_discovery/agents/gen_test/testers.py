from typing import Dict

import numpy as np

from aux_task_discovery.models import MasterUserNetwork
import aux_task_discovery.utils.pytorch_utils as ptu

class Tester():
    def eval_tasks(self) -> np.ndarray:
        '''
        Compute task utilities for all auxillery tasks
        '''
        raise NotImplementedError()

class TraceTester(Tester):
    '''
    Proposed tester from Gen+Test Paper. Uses traces of induced features and magnitude 
    of induced feature weights on the main task to evaluate auxillery task utility.
    '''
    def __init__(self, model: MasterUserNetwork, tau = 0.05, **kwargs):
        self.model = model
        self.tau = tau
        self.trace = np.zeros(model.hidden_size)

    def eval_tasks(self, batch: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        # Get magnitude of shared features for current obs
        obs = ptu.from_numpy(batch['observations'])#.unsqueeze(0)
        feature_magnitudes = np.absolute(ptu.to_numpy(self.model.get_shared_features(obs)))#[0])
        # Get trace for all features
        self.trace = (1-self.tau) * self.trace + self.tau * feature_magnitudes
        # Get sum of weight magnitudes across all actions on main task for each feature
        w_main = np.sum(np.absolute(ptu.to_numpy(self.model.main_head.weight)), axis=0)
        # Get feature utilities
        feature_utils = np.mean(w_main*self.trace, axis=0)
        # Get aux_task utilities
        task_utils = []
        for i in range(self.model.n_aux_tasks):
            # Get indicies of features induced by task
            start, stop = self.model.feature_ranges[i]
            # Sum feature utilities
            task_utils.append(feature_utils[start:stop].sum())
        
        return np.array(task_utils)


TESTER_REG = {
    'trace': TraceTester,
}

def get_tester(tester: str):
    assert tester in TESTER_REG, 'Given tester is not registered'
    return TESTER_REG[tester]
