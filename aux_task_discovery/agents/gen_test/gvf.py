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
    
    def gamma(self, obs: np.ndarray) -> Union[float, int]:
        '''
        GVF continuation function. Returns probability of continuation for a given state.
        '''
        prob = self._gamma(obs)
        assert 0 <= prob <= 1, 'GVF continuation function did not return value in range [0,1]'
        return prob
    
    def cumulant(self, obs: np.ndarray) -> Union[float, int]:
        '''
        Returns GVF cumulant for a given state
        '''
        return self._cumulant(obs)
