import unittest
from machinelearn import models

import numpy as np

class TestModels(unittest.TestCase):
    def test_linear_model(self):
        theta = np.array([1, 1])
        inputs = np.array([[2]])
        self.assertEqual(models.linear_model(theta, inputs), np.array([3]))

        theta = np.array([1, 1])
        inputs = np.array([[2], [3], [4]])
        self.assertTrue(np.all(models.linear_model(theta, inputs) == 
                               np.array([3, 4, 5])))

        theta = np.array([3, 2, 1])
        inputs = np.array([[1, 2], [3, 4]])
        self.assertTrue(np.all(models.linear_model(theta, inputs) ==
                               np.array([7, 13])))

def test_sigmoid():
    assert models.sigmoid(0) == 0.5
    assert np.around(models.sigmoid(1), decimals=4) == 0.7311

    x = np.array([0, 1])
    results = models.sigmoid(x)
    assert np.all(np.around(results, decimals=4) 
                  == np.array([0.5, 0.7311]))

    x = np.array([[0, 1], [1, -10]])
    results = models.sigmoid(x)
    assert np.all(np.around(results, decimals=4)
                  == np.array([[0.5, 0.7311], [0.7311, 0.0]]))

def test_logistic_model():
    theta = np.array([0, 0])
    inputs = np.array([[1]])
    outputs = models.logistic_model(theta, inputs)
    assert outputs == 0.5

    theta = np.array([0, 1])
    inputs = np.array([[1]])
    outputs = models.logistic_model(theta, inputs)
    assert np.around(outputs, decimals=4) == 0.7311

    theta = np.array([0, 1])
    inputs = np.array([[0], [1]])
    outputs = models.logistic_model(theta, inputs)
    assert np.all(np.around(outputs, decimals=4) == np.array([0.5, 0.7311]))

    theta = np.array([1, 1, 2])
    inputs = np.array([[1, -1], [0, 0]])
    outputs = models.logistic_model(theta, inputs)
    assert np.all(np.around(outputs, decimals=4) == np.array([0.5, 0.7311]))

def test_predict_logistic():
    theta = np.array([0, 0])
    inputs = np.array([[1]])
    threshold = 0.5
    assert models.predict_logistic(theta, inputs, threshold) == 1

    theta = np.array([0, 1])
    inputs = np.array([[1]])
    threshold = 0.8
    assert models.predict_logistic(theta, inputs, threshold) == 0

    theta = np.array([0, 1])
    inputs = np.array([[0], [1]])
    threshold = 0.5
    assert np.all(models.predict_logistic(theta, inputs, threshold) 
                  == np.array([1, 1]))

    theta = np.array([1, 1, 2])
    inputs = np.array([[1, -1], [0, 0]])
    threshold = 0.6
    assert np.all(models.predict_logistic(theta, inputs, threshold) 
                  == np.array([0, 1]))


def test_cost_regularize():
    theta = np.array([0, 0])
    n_samples = 1
    reg = 1
    cost = models.cost_regularize(reg, n_samples, theta)
    assert cost == 0

    theta = np.array([1, -1])
    n_samples = 2
    reg = 0.5
    cost = models.cost_regularize(reg, n_samples, theta)
    assert cost == 0.25

    theta = np.array([1, 2, 3])
    n_samples = 2
    reg = 1
    cost = models.cost_regularize(reg, n_samples, theta)
    assert cost == 3.5

def test_cost_rms_regularize():
    theta = np.array([1, 1])
    inputs = np.array([[2]])
    outputs = np.array([3])
    cost = models.cost_rms(theta, inputs, outputs, models.linear_model)
    assert cost == 0

    theta = np.array([1, 1])
    inputs = np.array([[2], [2]])
    outputs = np.array([3, 5])
    cost = models.cost_rms(theta, inputs, outputs, models.linear_model)
    assert cost == 1

    theta = np.array([1, 2, 3])
    inputs = np.array([[2, 3], [4, 5]])
    outputs = np.array([3, 5])
    cost = models.cost_rms(theta, inputs, outputs, models.linear_model)
    assert cost == 120.5

    theta = np.array([1, 1])
    inputs = np.array([[2]])
    outputs = np.array([3])
    reg = 1
    cost = models.cost_rms(theta, inputs, outputs, models.linear_model, reg)
    assert cost == 1

    theta = np.array([1, 1])
    inputs = np.array([[2], [2]])
    outputs = np.array([3, 5])
    reg = 1
    cost = models.cost_rms(theta, inputs, outputs, models.linear_model, reg)
    assert cost == 1.5

    theta = np.array([1, 2, 3])
    inputs = np.array([[2, 3], [4, 5]])
    outputs = np.array([3, 5])
    reg = 1
    cost = models.cost_rms(theta, inputs, outputs, models.linear_model, reg)
    assert cost == 124

def test_cost_logistic():
    theta = np.array([0, 0])
    inputs = np.array([[1]])
    outputs = np.array([1])
    cost = models.cost_logistic(theta, inputs, outputs, models.logistic_model)
    assert np.around(cost, decimals=4) == 0.6931

    theta = np.array([0, 0])
    inputs = np.array([[1]])
    outputs = np.array([0])
    cost = models.cost_logistic(theta, inputs, outputs, models.logistic_model)
    assert np.around(cost, decimals=4) == 0.6931

    theta = np.array([0, 0])
    inputs = np.array([[1], [1]])
    outputs = np.array([1, 0])
    cost = models.cost_logistic(theta, inputs, outputs, models.logistic_model)
    assert np.around(cost, decimals=4) == 0.6931

    theta = np.array([1, -1])
    inputs = np.array([[0], [1]])
    outputs = np.array([1, 0])
    cost = models.cost_logistic(theta, inputs, outputs, models.logistic_model)
    assert np.around(cost, decimals=4) == 0.5032 

    theta = np.array([0, 0])
    inputs = np.array([[1]])
    outputs = np.array([1])
    reg = 1
    cost = models.cost_logistic(theta, inputs, outputs, models.logistic_model, reg)
    assert np.around(cost, decimals=4) == 0.6931

    theta = np.array([1, -1])
    inputs = np.array([[0], [1]])
    outputs = np.array([1, 0])
    reg = 0.5
    cost = models.cost_logistic(theta, inputs, outputs, models.logistic_model, reg)
    assert np.around(cost, decimals=4) == 0.5032 + 0.25

def test_jacobien_regularize():
    theta = np.array([0, 0])
    n_samples = 1
    reg = 1
    jacobien = models.jacobien_regularize(reg, n_samples, theta)
    assert np.all(jacobien == np.array([0, 0]))

    theta = np.array([1, 1])
    n_samples = 1
    reg = 0.5
    jacobien = models.jacobien_regularize(reg, n_samples, theta)
    assert np.all(jacobien == np.array([0, 0.5]))

    theta = np.array([3, 2, 1])
    n_samples = 2
    reg = 1
    jacobien = models.jacobien_regularize(reg, n_samples, theta)
    assert np.all(jacobien == np.array([0, 1, 0.5]))

def test_jacobien_rms_linear():
    theta = np.array([1, 1])
    inputs = np.array([[2]])
    outputs = np.array([3])
    jacobien = models.jacobien_rms_linear(theta, inputs, outputs)
    assert np.all(jacobien == np.array([0, 0]))
    
    theta = np.array([1, 1])
    inputs = np.array([[2]])
    outputs = np.array([5])
    jacobien = models.jacobien_rms_linear(theta, inputs, outputs)
    assert np.all(jacobien == np.array([-2, -4]))

    theta = np.array([1, 1])
    inputs = np.array([[2], [2]])
    outputs = np.array([5, 3])
    jacobien = models.jacobien_rms_linear(theta, inputs, outputs)
    assert np.all(jacobien == np.array([-1, -2]))

    theta = np.array([1, 1])
    inputs = np.array([[2], [3]])
    outputs = np.array([5, 3])
    jacobien = models.jacobien_rms_linear(theta, inputs, outputs)
    assert np.all(jacobien == np.array([-0.5, -0.5]))

    theta = np.array([3, 2, 1])
    inputs = np.array([[1, 2], [3, 4]])
    outputs = np.array([5, 3])
    jacobien = models.jacobien_rms_linear(theta, inputs, outputs)
    assert np.all(jacobien == np.array([6, 16, 22]))

    theta = np.array([1, 1])
    inputs = np.array([[2]])
    outputs = np.array([5])
    reg = 0.5
    jacobien = models.jacobien_rms_linear(theta, inputs, outputs, reg)
    assert np.all(jacobien == np.array([-2, -3.5]))

    theta = np.array([3, 2, 1])
    inputs = np.array([[1, 2], [3, 4]])
    outputs = np.array([5, 3])
    reg = 1
    jacobien = models.jacobien_rms_linear(theta, inputs, outputs, reg)
    assert np.all(jacobien == np.array([6, 17, 22.5]))

def test_jacobien_logistic():
    theta = np.array([1, 1])
    inputs = np.array([[2]])
    outputs = np.array([1])
    jacobien = models.jacobien_logistic(theta, inputs, outputs)
    assert np.all(np.around(jacobien, decimals=4) == np.array([-0.0474, -0.0949]))
    
    theta = np.array([3, 2, 1])
    inputs = np.array([[1, 2], [3, 4]])
    outputs = np.array([1, 0])
    jacobien = models.jacobien_logistic(theta, inputs, outputs)
    assert np.all(np.around(jacobien, decimals=4) 
                  == np.array([0.4995, 1.4995, 1.9991]))


if __name__ == '__main__':
    unittest.main()
