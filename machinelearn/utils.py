import numpy as np
import matplotlib.pyplot as plt

def extended_inputs(inputs):
    return np.column_stack((np.ones(inputs.shape[0]), inputs))

def map_feature(inputs, degree):
    features = inputs.copy()
    for total in range(2, degree+1):
        for i in range(total+1):
            new = inputs[:, 0]**(total-i) * inputs[:, 1]**i
            features = np.column_stack((features, new))
    return features

def feature_normalize(inputs):
    mean = np.mean(inputs, axis=0)
    std = np.std(inputs, axis=0)
    normalized = (inputs - mean) / std
    return normalized, mean, std

def parameter_unnormalize(theta, mean, std):
    unnormalized = np.zeros(theta.shape)
    unnormalized[0] = theta[0] - np.sum(mean/std * theta[1:])
    unnormalized[1:] = 1/std * theta[1:]
    return unnormalized

def costs(theta_history, inputs, outputs, cost, model, reg=0):
    cost_history = np.zeros(theta_history.shape[0])
    for i in range(theta_history.shape[0]):
        cost_history[i] = cost(theta_history[i], inputs, outputs, model, reg=0)
    return cost_history

def is_cost_decreasing(cost_history):
    return np.all(cost_history[1:] < cost_history[:-1])

def plot_costs_vs_number_of_iterations(cost_history):
    fig, ax = plt.subplots()
    ax.plot(cost_history)
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Cost function')

def plot_prediction_one_variable(inputs, outputs, model, theta):
    fig, ax = plt.subplots()
    ax.scatter(inputs, outputs, c='r', label='data')
    ax.plot(inputs, model(theta, inputs), label='prediction')
    ax.set_xlabel('Inputs')
    ax.set_ylabel('Outputs')
    ax.legend()

def plot_decision_boundary_2d_linear(theta, inputs, ax=None):
    minx, maxx, miny, maxy = make_limits(inputs)
    xs = np.array([minx, maxx]) 
    if theta[2] != 0:
        ys = - (theta[1]*xs + theta[0]) / theta[2]
    else:
        xs = - theta[0]/theta[1]
        ys = np.array([miny, maxy])

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(xs, ys)
    return ax

def plot_decision_boundary_2d_nonlinear(theta, inputs, map_inputs=None, ax=None):
    minx, maxx, miny, maxy = make_limits(inputs)
    xs = np.linspace(minx, maxx, 50)
    ys = np.linspace(miny, maxy, 50)
    zs = np.zeros((xs.shape[0], ys.shape[0]))
    for i in range(xs.shape[0]):
        for j in range(ys.shape[0]):
            point = np.array([[xs[i], ys[j]]])
            if map_inputs is not None:
                point = map_inputs(point)
            zs[i, j] = extended_inputs(point) @ theta

    if ax is None:
        fig, ax = plt.subplots()
    cs = ax.contour(xs, ys, zs.T, [0])
    ax.clabel(cs, inline=1, fontsize=10)
    return ax

def make_limits(inputs):
    minx = np.min(inputs[:, 0])
    maxx = np.max(inputs[:, 0])
    dx = 0.1 * (maxx - minx)
    miny = np.min(inputs[:, 1])
    maxy = np.max(inputs[:, 1])
    dy = 0.1 * (maxy - miny)
    return minx-dx, maxx+dx, miny-dy, maxy+dy
