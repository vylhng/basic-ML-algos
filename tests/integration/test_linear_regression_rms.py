import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import machinelearn.models as mlmodels
import machinelearn.optimization as mloptim
import machinelearn.utils as mlutils

def load_data(filename):
    data = pd.read_csv('tests/integration/fixtures/ex1data1.txt', 
                       names=['inputs', 'outputs'])
    inputs = data.inputs.to_numpy()
    outputs = data.outputs.to_numpy()
    return inputs, outputs

def check_results_linear_regression_one_variable(theta_history, theta_opt, cost_history):
    assert mlutils.is_cost_decreasing(cost_history) == True
    assert np.all(np.around(theta_history[-1], decimals=4) 
                  == np.around(theta_opt, decimals=4))

def compute_linear_regression_one_variable(inputs, outputs, theta0, alpha, n_iter):
    theta_history = mloptim.gradient_descent(theta0, inputs, outputs, 
                                             mlmodels.jacobien_rms_linear,
                                             alpha, n_iter)
    cost_history = mlutils.costs(theta_history, inputs, outputs,
                                 mlmodels.cost_rms, mlmodels.linear_model)
    theta_opt = mloptim.normal_equation(inputs, outputs)
    return theta_history, theta_opt, cost_history

def test_linear_regression_one_variable():
    X = np.array([[1], [2]])
    y = np.array([1, 2])
    theta0 = np.array([0, 0])
    alpha = 0.3 
    n_iter = 1000
    theta_history, theta_opt, cost_history = compute_linear_regression_one_variable(
                                           X, y, theta0, alpha, n_iter)
    check_results_linear_regression_one_variable(theta_history, theta_opt, 
                                                 cost_history)

    X = np.array([[1], [2], [5]])
    y = np.array([1, 2, 4])
    theta0 = np.array([0, 0])
    alpha = 0.1
    n_iter = 500
    theta_history, theta_opt, cost_history = compute_linear_regression_one_variable(
                                           X, y, theta0, alpha, n_iter)
    check_results_linear_regression_one_variable(theta_history, theta_opt, 
                                                 cost_history)

    X, y = load_data('ex1data1.txt')
    theta0 = np.array([0, 0])
    alpha = 0.01
    n_iter = 7900
    theta_history, theta_opt, cost_history = compute_linear_regression_one_variable(
                                           X, y, theta0, alpha, n_iter)
    check_results_linear_regression_one_variable(theta_history, theta_opt, 
                                                 cost_history)
