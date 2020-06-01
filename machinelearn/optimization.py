import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from machinelearn import utils

def gradient_descent(theta0, inputs, outputs, jacobien, alpha, n_iter, reg=0):
    theta_history = np.zeros((n_iter + 1, len(theta0)))
    theta_history[0] = theta0
    for i in range(n_iter):
        jab = jacobien(theta_history[i], inputs, outputs, reg)
        theta_history[i+1] = theta_history[i] - alpha*jab
    return theta_history

def normal_equation(inputs, outputs):
    X = utils.extended_inputs(inputs)
    theta = np.linalg.inv(X.T @ X) @ X.T @ outputs
    return theta
