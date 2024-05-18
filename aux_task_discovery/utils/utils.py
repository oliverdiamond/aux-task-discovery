import numpy as np

def random_argmax(arr: np.ndarray):
    '''
    Returns idx of largest element in the array with ties broken randomly
    '''
    return np.random.choice(np.flatnonzero(arr == arr.max()))


#------------------------TESTS------------------------#

def test_random_argmax():
    np.random.seed(42)
    assert random_argmax(np.array([1.1111101,1.1111101,1.1111101])) == 2
    assert random_argmax(np.array([1,2,3,4])) == 3
    assert random_argmax(np.array([-1, 3.44447, 3.44446])) == 1

def run_tests():
    test_random_argmax()

if __name__ == "__main__":
    run_tests()