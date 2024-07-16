from typing import Dict, Sequence

import numpy as np

from aux_task_discovery.models import MasterUserNetwork
import aux_task_discovery.utils.pytorch_utils as ptu

class Tester():
    def eval_tasks(self) -> np.ndarray:
        '''
        Compute task utilities for all auxillery tasks
        '''
        raise NotImplementedError()
    
    def reset_tasks(self, idxs: Sequence[int]):
        '''
        Reset tester attributes as needed for replaced tasks
        '''
        raise NotImplementedError()

class TraceTester(Tester):
    '''
    TraceTester uses traces of induced features (calculated using the observation 
    from the current timestep) and magnitude of induced feature weights on the main 
    task to evaluate auxillery task utility.
    '''
    def __init__(self, model: MasterUserNetwork, tau = 0.05, **kwargs):
        self.model = model
        self.tau = tau
        self.trace = np.zeros(model.hidden_size)

    def eval_tasks(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        # Get magnitude of shared features for current obs
        obs = ptu.from_numpy(observation).unsqueeze(0)
        feature_magnitudes = np.absolute(ptu.to_numpy(self.model.get_shared_features(obs))[0])
        # Get trace for all features
        self.trace = (1-self.tau) * self.trace + self.tau * feature_magnitudes
        # Get sum of weight magnitudes across all actions on main task for each feature
        w_main = np.sum(np.absolute(ptu.to_numpy(self.model.main_head.weight)), axis=0)
        # Get feature utilities
        feature_utils = w_main*self.trace
        # Get aux_task utilities
        task_utils = []
        for i in range(self.model.n_aux_tasks):
            # Get indicies of features induced by task
            start, stop = self.model.feature_ranges[i]
            # Sum feature utilities
            task_utils.append(feature_utils[start:stop].sum())
        
        return np.array(task_utils)

    def reset_tasks(self, idxs: Sequence[int]):
        self.reset_traces(idxs)
    
    def reset_traces(self, idxs: Sequence[int]):
        '''
        Reset traces for features induced by the replaced tasks
        '''
        # Calculate median trace for all features induced by auxilliary tasks that were not just replaced
        preserved_feature_idxs = np.concatenate(
            [
                np.arange(self.model.feature_ranges[i][0], self.model.feature_ranges[i][1]) 
                for i in np.arange(self.model.n_aux_tasks)
                if i not in idxs
            ]
        )
        median_trace = np.median(self.trace[preserved_feature_idxs])
        # Reset the trace for features induced by the replaced tasks to the median trace
        for i in idxs:
            start, stop = self.model.feature_ranges[i]
            self.trace[start:stop] = median_trace

class BatchTraceTester(Tester):
    '''
    Tester used in the Gen+Test Paper. Uses traces of induced features (calculated 
    by maintaining a separate trace for each observation in the batch sampled from the 
    replay buffer and then averaging them) and magnitude of induced feature weights on 
    the main task to evaluate auxillery task utility.
    '''
    def __init__(self, model: MasterUserNetwork, tau = 0.05, **kwargs):
        self.model = model
        self.tau = tau
        self.trace = np.zeros(model.hidden_size)

    def eval_tasks(self, batch: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        # Get magnitude of shared features for current obs
        obs = ptu.from_numpy(batch['observations'])
        feature_magnitudes = np.absolute(ptu.to_numpy(self.model.get_shared_features(obs)))
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
    
    def reset_tasks(self, idxs: Sequence[int]):
        self.reset_traces(idxs)
    
    def reset_traces(self, idxs: Sequence[int]):
        '''
        Reset traces for features induced by the replaced tasks
        '''
        # Calculate median trace for all features induced by auxilliary tasks that were not just replaced
        preserved_feature_idxs = np.concatenate(
            [
                np.arange(self.model.feature_ranges[i][0], self.model.feature_ranges[i][1]) 
                for i in np.arange(self.model.n_aux_tasks)
                if i not in idxs
            ]
        )
        median_trace = np.median(self.trace[:, preserved_feature_idxs])
        # Reset the trace for features induced by the replaced tasks to the median trace
        for i in idxs:
            start, stop = self.model.feature_ranges[i]
            self.trace[:, start:stop] = median_trace

TESTER_REG = {
    'trace': TraceTester,
    'batch_trace': BatchTraceTester,
}

def get_tester(tester: str):
    assert tester in TESTER_REG, 'Given tester is not registered'
    return TESTER_REG[tester]
