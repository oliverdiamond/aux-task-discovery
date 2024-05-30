import pytest

import numpy as np

from aux_task_discovery.agents.gen_test.generators import OneHotGenerator, FourroomsCornerGenerator, FourroomsHallwayGenerator

@pytest.mark.parametrize("generator_cls", [OneHotGenerator, FourroomsCornerGenerator, FourroomsHallwayGenerator])
def test_len_generate_tasks(generator_cls):
    generator = generator_cls(input_shape=(49,), seed=42)
    tasks = generator.generate_tasks(n_tasks=2)
    assert len(tasks) == 2

@pytest.mark.parametrize(
        "task_idx, obs",
        [
            (0, np.array([0,0,0,0])),
            (0, np.array([0,0,1,0])),
            (1, np.array([0,0,0,0])),
            (1, np.array([0,0,1,0])),
        ]
)
def test_OneHotGenerator_cumulant(task_idx, obs):
    generator = OneHotGenerator(input_shape=(4,), seed=42)
    gvfs = generator.generate_tasks(2)
    assert gvfs[task_idx].cumulant(obs) == -1


@pytest.mark.parametrize(
        "task_idx, obs, expected",
        [
            (0, np.array([0,0,0,0]), 1),
            (0, np.array([0,0,1,0]), 0),
            (1, np.array([0,0,1,0]), 1),
            (1, np.array([0,0,0,1]), 0),
        ]
)
def test_OneHotGenerator_gamma(task_idx, obs, expected):
    generator = OneHotGenerator(input_shape=(4,), seed=42)
    gvfs = generator.generate_tasks(2)
    assert gvfs[task_idx].gamma(obs) == expected

@pytest.mark.parametrize("generator_cls", [FourroomsCornerGenerator, FourroomsHallwayGenerator])
def test_fourrooms_generator_input_shape(generator_cls):
    with pytest.raises(Exception):
        generator = generator_cls((5,))

@pytest.mark.parametrize("generator_cls", [FourroomsCornerGenerator, FourroomsHallwayGenerator])
def test_fourrooms_generator_n_tasks(generator_cls):
    with pytest.raises(Exception):
        generator = generator_cls((49,))
        generator.generate_tasks(4)

@pytest.mark.parametrize("generator_cls", [FourroomsCornerGenerator, FourroomsHallwayGenerator])
def test_fourrooms_generator_new_tasks(generator_cls):
    with pytest.raises(Exception):
        generator = generator_cls((49,))
        generator.generate_tasks(2)
        generator.generate_tasks(2)

@pytest.fixture
def corner_obs():
    corner1 = np.zeros(49)
    corner1[0] = 1
    corner2 = np.zeros(49)
    corner2[2] = 1
    return (corner1, corner2)


@pytest.mark.parametrize(
        "task_idx, corner_idx",
        [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
)
def test_FourroomsCornerGenerator_cumulant(corner_obs, task_idx, corner_idx):
    gvfs = FourroomsCornerGenerator((49,)).generate_tasks()
    assert gvfs[task_idx].cumulant(corner_obs[corner_idx]) == -1
    

@pytest.mark.parametrize(
        "task_idx, corner_idx, expected",
        [
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
        ]
)
def test_FourroomsCornerGenerator_cumulant(corner_obs, task_idx, corner_idx, expected):
    gvfs = FourroomsCornerGenerator((49,)).generate_tasks()
    assert gvfs[task_idx].gamma(corner_obs[corner_idx]) == expected

@pytest.fixture
def hallway_obs():
    hallway1 = np.zeros(49)
    hallway1[33] = 1
    hallway2 = np.zeros(49)
    hallway2[38] = 1
    return (hallway1, hallway2)

@pytest.mark.parametrize(
        "task_idx, hallway_idx",
        [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
)
def test_FourroomsHallwayGenerator_cumulant(hallway_obs, task_idx, hallway_idx):
    gvfs = FourroomsHallwayGenerator((49,)).generate_tasks()
    assert gvfs[task_idx].cumulant(hallway_obs[hallway_idx]) == -1
    

@pytest.mark.parametrize(
        "task_idx, hallway_idx, expected",
        [
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
        ]
)
def test_FourroomsCornerGenerator_gamma(hallway_obs, task_idx, hallway_idx, expected):
    gvfs = FourroomsHallwayGenerator((49,)).generate_tasks()
    assert gvfs[task_idx].gamma(hallway_obs[hallway_idx]) == expected