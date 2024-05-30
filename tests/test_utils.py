import pytest

import numpy as np

from aux_task_discovery.utils import utils

@pytest.mark.parametrize(
        "arr, expected",
        [
            (np.array([1.1111101,1.1111101,1.1111101]), 2),
            (np.array([1,2,3,4]), 3),
            (np.array([-1, 3.44447, 3.44446]), 1)
        ]
)
def test_random_argmax(arr, expected):
    np.random.seed(42)
    assert utils.random_argmax(arr) == expected