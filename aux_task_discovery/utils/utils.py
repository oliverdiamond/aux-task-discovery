import yaml 

import numpy as np

def random_argmax(arr: np.ndarray):
    '''
    Returns idx of largest element in the array with ties broken randomly
    '''
    return np.random.choice(np.flatnonzero(arr == arr.max()))

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config