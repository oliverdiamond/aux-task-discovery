from typing import Callable, Union

import numpy as np

class GVF():
    '''
    Generic GVF (General Value Function) class which takes a cumulant function 
    and gamma (the continuation function) as input.
    '''
    def __init__(self,
                 cumulant: Callable[[np.ndarray], Union[float, int]],
                 gamma: Callable[[np.ndarray], Union[float, int]]):
        self._cumulant = cumulant
        self._gamma = gamma
    
    def gamma(self, obs: np.ndarray) -> np.ndarray:
        '''
        GVF continuation function. Returns probability of continuation for a given state.
        Can be applied to a sinlge obs or batch with dimension 1 greater than obs.
        '''
        if len(obs.shape) == 1:
            prob = np.array([self._gamma(obs)])
        else:
            prob = np.apply_along_axis(self._gamma, 1, obs)
        assert np.all((prob >= 0) & (prob <= 1)), 'GVF gamma did not return value in range [0,1]'
        
        return prob
    
    def cumulant(self, obs: np.ndarray) -> np.ndarray:
        '''
        Returns GVF cumulant for a given state.
        Can be applied to a sinlge obs or batch with dimension 1 greater than obs.
        '''
        if len(obs.shape) == 1:
            return np.array([self._cumulant(obs)])
        
        return np.apply_along_axis(self._cumulant, 1, obs)

class SubgoalGVF(GVF):
    '''
    Subgoal-reaching GVF. The cumulant is -1 at all states, and gamma is 0 at the subgoal state and 1 elsewhere.
    '''
    def __init__(self, subgoal: np.ndarray):
        super().__init__(cumulant=lambda obs : -1, 
                         gamma=lambda obs : int(not np.allclose(obs,subgoal)))
        self.subgoal = subgoal


