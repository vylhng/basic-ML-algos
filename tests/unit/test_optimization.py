import unittest

from machinelearn import optimization

from machinelearn import models
import numpy as np

class TestGradientDescent(unittest.TestCase):
    def test_gradient_descent(self):
        theta0 = np.array([1, 1])
        inputs = np.array([[2]])
        outputs = np.array([3])
        theta_history = optimization.gradient_descent(theta0, inputs, outputs,
                                                models.jacobien_rms_linear,
                                                0.1, 1)
        self.assertTrue(np.all(theta_history == np.array([[1, 1],
                                                          [1, 1]])))

        theta0 = np.array([1, 1])
        inputs = np.array([[2]])
        outputs = np.array([5])
        theta_history = optimization.gradient_descent(theta0, inputs, outputs,
                                                models.jacobien_rms_linear,
                                                0.1, 1)
        self.assertTrue(np.all(theta_history == np.array([[1, 1],
                                                          [1.2, 1.4]])))

        theta0 = np.array([1, 1])
        inputs = np.array([[2], [2]])
        outputs = np.array([5, 3])
        theta_history = optimization.gradient_descent(theta0, inputs, outputs,
                                                models.jacobien_rms_linear,
                                                0.1, 1)
        self.assertTrue(np.all(theta_history == np.array([[1, 1],
                                                          [1.1, 1.2]])))

        theta0 = np.array([1.1, 1.2])
        inputs = np.array([[2], [2]])
        outputs = np.array([5, 3])
        theta_history = optimization.gradient_descent(theta0, inputs, outputs,
                                                models.jacobien_rms_linear,
                                                0.1, 1)
        self.assertTrue(np.all(theta_history[0, :] == np.array([1.1, 1.2])))
        self.assertTrue(theta_history[1, 1] == 1.3)

    def test_normal_equation(self):
        self.assertTrue(True)
        
if __name__ == '__main__':
    unittest.main()
