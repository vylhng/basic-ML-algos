import unittest
from machinelearn import utils

import numpy as np

class TestUtils(unittest.TestCase):
    def test_feature_normalize(self):
        inputs = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])
        normalized, mean, std = utils.feature_normalize(inputs)
        self.assertTrue(np.all(normalized[1, :] == 0))
        self.assertTrue(np.all(mean == np.array([5, 6, 7, 8])))

    def test_is_cost_decreasing(self):
        cost_history = np.array([3, 2, 1])
        self.assertTrue(utils.is_cost_decreasing(cost_history))

        cost_history = np.array([2, 3, 1])
        self.assertFalse(utils.is_cost_decreasing(cost_history))

        cost_history = np.array([2])
        self.assertTrue(utils.is_cost_decreasing(cost_history))

        cost_history = np.array([2, 1.9, 1.8, 1])
        self.assertTrue(utils.is_cost_decreasing(cost_history))

def test_extended_inputs():
    inputs = np.array([[3]])
    assert np.all(utils.extended_inputs(inputs) == np.array([[1, 3]])) 

    inputs = np.array([[3], [4]])
    assert np.all(utils.extended_inputs(inputs) 
                  == np.array([[1, 3], [1, 4]])) 

    inputs = np.array([[3, 4, 5], [6, 7, 8]])
    assert np.all(utils.extended_inputs(inputs) 
                  == np.array([[1, 3, 4, 5], [1, 6, 7, 8]])) 

def test_map_feature():
    inputs = np.array([[1, 2]])
    degree = 1
    features = utils.map_feature(inputs, degree)
    assert np.all(features == np.array([[1, 2]]))

    inputs = np.array([[1, 2]])
    degree = 2
    features = utils.map_feature(inputs, degree)
    assert np.all(features == np.array([[1, 2, 1, 2, 4]]))

    inputs = np.array([[1, 2]])
    degree = 3
    features = utils.map_feature(inputs, degree)
    assert np.all(features == np.array([[1, 2, 1, 2, 4, 1, 2, 4, 8]]))

    inputs = np.array([[1, 2], [3, 4], [5, 6]])
    degree = 2
    features = utils.map_feature(inputs, degree)
    assert np.all(features == np.array([[1, 2, 1, 2, 4],
                                        [3, 4, 9, 12, 16],
                                        [5, 6, 25, 30, 36]]))

def test_parameter_unnormalize():
    theta = np.array([1, 2])
    mean = 3
    std = 0.5
    unnormalized_theta = utils.parameter_unnormalize(theta, mean, std)
    assert np.all(unnormalized_theta == np.array([-11, 4]))

    theta = np.array([1, 2, 3])
    mean = np.array([3, -1])
    std = np.array([1, 1.5])
    unnormalized_theta = utils.parameter_unnormalize(theta, mean, std)
    assert np.all(unnormalized_theta == np.array([-3, 2, 2]))

def test_make_limits():
    inputs = np.array([[1, 2]])
    minx, maxx, miny, maxy = utils.make_limits(inputs)
    assert np.all(np.array([minx, maxx, miny, maxy]) 
                  == np.array([1, 1, 2, 2]))

    inputs = np.array([[1, 2], [3, 5]])
    minx, maxx, miny, maxy = utils.make_limits(inputs)
    assert np.all(np.array([minx, maxx, miny, maxy]) 
                  == np.array([0.8, 3.2, 1.7, 5.3]))

if __name__ == '__main__':
    unittest.main()
