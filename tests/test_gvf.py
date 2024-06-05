import pytest

import numpy as np

from aux_task_discovery.agents.gen_test.gvf import GVF

@pytest.fixture
def gvf():
     return GVF(cumulant = lambda obs : -1, 
                gamma = lambda obs: int(not np.allclose(obs,np.array([0,0,0])))
                )

@pytest.mark.parametrize(
        "obs, expected",
        [
            (np.array([0,0,0]), np.array([0])),
            (np.array([0,0,1]), np.array([1])),
            (np.array([[0,0,0],[0,0,1]]), np.array([0,1]))
        ]
)
def test_gvf_gamma(gvf, obs, expected):
    assert np.array_equal(gvf.gamma(obs), expected)

@pytest.mark.parametrize(
        "obs, expected",
        [
            (np.array([0,0,0]), np.array([-1])),
            (np.array([0,0,1]), np.array([-1])),
            (np.array([[0,0,0],[0,0,1]]), np.array([-1,-1]))
        ]
)
def test_gvf_cumulant(gvf, obs, expected):
    assert np.array_equal(gvf.cumulant(obs), expected)