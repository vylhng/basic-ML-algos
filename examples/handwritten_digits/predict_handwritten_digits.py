import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from sklearn.linear_model import LogisticRegression

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, RMSprop

import machinelearn.models as mlmodels
import machinelearn.optimization as mloptim
import machinelearn.utils as mlutils

from scipy.optimize import minimize

def load_data(filename):
    mat = scipy.io.loadmat(filename)
    inputs = mat['X']
    outputs = mat['y']
    return inputs, outputs

def load_params(filename):
    mat = scipy.io.loadmat(filename)
    theta1 = mat['Theta1']
    theta2 = mat['Theta2']
    return theta1, theta2

def display_data(inputs, n_images_per_row):
    plt.figure()
    plt.subplots(n_images_per_row, n_images_per_row)
    n_plots = 0
    chosen_images = np.random.randint(1, len(X), n_images_per_row**2)
    for i in chosen_images:
        n_plots += 1
        plt.subplot(n_images_per_row, n_images_per_row, n_plots)
        plt.imshow(X[i-1].reshape(20, 20).transpose())
    return chosen_images

def compute_cost_history(theta_history,
                         X, y_vec,
                         reg):
    cost_history = np.zeros((theta_history.shape[0], 1))
    for i in range(theta_history.shape[0]):
        cost_history[i] = cost_neural_network(theta_history[i, :], X, y_vec, reg)
    return cost_history    


def unroll_params(theta1, theta2):
    return np.hstack((theta1.ravel(), theta2.ravel()))

def roll_params(params, input_layer_size, hidden_layer_size, num_labels):
    theta1 = params[ : hidden_layer_size * (input_layer_size + 1)].reshape(
            hidden_layer_size, (input_layer_size + 1))
    theta2 = params[hidden_layer_size * (input_layer_size + 1) : ].reshape(
            num_labels, (hidden_layer_size + 1))
    return theta1, theta2

def convert_theta_to_Wb(theta):
    W = theta[:, 1:]
    b = theta[:, 0].reshape(-1, 1)
    return W, b

def convert_labels_to_vectors(labels, num_labels):
    num_samples, num_col = labels.shape
    y_vec = np.zeros((num_samples, num_labels))
    for i in range(num_samples):
        y_vec[i, labels[i]-1] = 1
    return y_vec

def convert_vectors_to_labels(vectors):
    num_samples, num_labels = vectors.shape
    y = np.zeros((num_samples, 1))
    for i in range(num_samples):
        y[i, 0] = vectors[i, :].argmax() + 1
    return y

def sigmoid_gradient(z):
    return mlmodels.sigmoid(z) * (1 - mlmodels.sigmoid(z))


def rand_initialize_weights(num_in, num_out):
    epsilon_init = 0.12
    theta = (np.random.random((num_out, num_in + 1)) * 2 * epsilon_init 
             - epsilon_init)
    return theta


def forward_propagation(input_layer_size, hidden_layer_size, num_labels, 
                        W1, b1, W2, b2, 
                        inputs):  
    z1 = inputs @ W1.T + b1.T
    a1 = mlmodels.sigmoid(z1)
    z2 = a1 @ W2.T + b2.T
    a2 = mlmodels.sigmoid(z2)
    return z1, a1, z2, a2

def backward_propagation(z1, a2, y, W2):
    delta2 = a2 - y
    delta1 = (delta2 @ W2) * sigmoid_gradient(z1)
    return delta1, delta2

def log_loss(outputs, predicted_outputs):
    m = outputs.shape[0]
    term_1 = outputs * np.log(predicted_outputs)
    term_2 = (1 - outputs) * np.log(1 - predicted_outputs)
    cost_log = -1/m * np.sum(term_1 + term_2)
    return cost_log

def l2_regularization(reg, theta1, theta2, num_samples):
    W1, b1 = convert_theta_to_Wb(theta1)
    W2, b2 = convert_theta_to_Wb(theta2)
    cost_reg = reg/(2 * num_samples) * (np.sum(W1**2) + np.sum(W2**2))
    return cost_reg

def cost_neural_network(params, X, y_vec, reg):
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    theta1, theta2 = roll_params(params, 
                                 input_layer_size, 
                                 hidden_layer_size, 
                                 num_labels)
    W1, b1 = convert_theta_to_Wb(theta1)
    W2, b2 = convert_theta_to_Wb(theta2)
    z1, a1, z2, y_hat = forward_propagation(input_layer_size, 
                                            hidden_layer_size, 
                                            num_labels,
                                            W1, b1, W2, b2,
                                            X)
    loss = log_loss(y_vec, y_hat)
    loss_regularization = l2_regularization(reg, theta1, theta2, num_samples)
    return loss + loss_regularization

def log_loss_gradient(inputs, a1, delta1, delta2):
    num_samples = inputs.shape[0]
    Delta1 = np.zeros((delta1.shape[1], inputs.shape[1] + 1))
    Delta2 = np.zeros((delta2.shape[1], delta1.shape[1] + 1))
    for i in range(num_samples):
        Delta1 += (delta1[i, :].reshape(-1, 1) 
                   @ np.insert(inputs[i, :], 1, 0).reshape(1, -1))
        Delta2 += (delta2[i, :].reshape(-1, 1) 
                   @ np.insert(a1[i, :], 1, 0).reshape(1, -1))
    grad_theta1 = 1/num_samples * Delta1
    grad_theta2 = 1/num_samples * Delta2
    return grad_theta1, grad_theta2

def l2_regularization_gradient(reg, W1, W2, num_samples):
    grad_W1 = reg/num_samples * np.hstack((np.zeros((W1.shape[0], 1)), W1))
    grad_W2 = reg/num_samples * np.hstack((np.zeros((W2.shape[0], 1)), W2))
    return grad_W1, grad_W2

def jacobien_neural_network(params, inputs, outputs, reg):
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    
    theta1, theta2 = roll_params(params, 
                                 input_layer_size, 
                                 hidden_layer_size,
                                 num_labels)
    W1, b1 = convert_theta_to_Wb(theta1)
    W2, b2 = convert_theta_to_Wb(theta2)
    
    z1, a1, z2, a2 = forward_propagation(input_layer_size, 
                                         hidden_layer_size, 
                                         num_labels, 
                                         W1, b1, W2, b2,
                                         inputs)
    delta1, delta2 = backward_propagation(z1, a2, outputs, W2)
    grad_theta1, grad_theta2 = log_loss_gradient(inputs, a1, delta1, delta2)
    grad_reg1, grad_reg2 = l2_regularization_gradient(reg, W1, W2, inputs.shape[0])
    return unroll_params(grad_theta1 + grad_reg1, grad_theta2 + grad_reg2) 

def predict(input_layer_size, hidden_layer_size, num_labels,
            params,
            inputs):
    theta1, theta2 = roll_params(params,
                                 input_layer_size,
                                 hidden_layer_size,
                                 num_labels)
    W1, b1 = convert_theta_to_Wb(theta1)
    W2, b2 = convert_theta_to_Wb(theta2)
    z1, a1, z2, a2 = forward_propagation(input_layer_size, 
                                         hidden_layer_size, 
                                         num_labels, 
                                         W1, b1, W2, b2, 
                                         inputs)
    y_hat = convert_vectors_to_labels(a2)
    return y_hat 


def test_unroll_roll_params():
    theta1, theta2 = load_params('ex4weights.mat')
    
    params = unroll_params(theta1, theta2)
    
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    new_theta1, new_theta2 = roll_params(params,
                                         input_layer_size,
                                         hidden_layer_size,
                                         num_labels)
    
    assert sum((new_theta1==theta1).ravel()) == theta1.size
    assert sum((new_theta2==theta2).ravel()) == theta2.size

def test_convert_theta_to_Wb():
    theta = np.array([[1, 2, 3], 
                      [4, 5, 6]])
    W, b = convert_theta_to_Wb(theta)
    assert np.array_equal(W, np.array([[2, 3],
                                      [5, 6]]))
    assert np.array_equal(b, np.array([[1], 
                                       [4]]))
    
def test_convert_labels_to_vectors():
    y = np.array([[1], 
                  [3], 
                  [2]])
    y_vec = np.array([[1, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0]])
    assert np.array_equal(convert_labels_to_vectors(y, 3), y_vec)

def test_sigmoid_gradient():
    a = np.array([[0, 2, 20], 
                      [-40, -7, 2]])
    assert np.array_equal(sigmoid_gradient(0), 0.25)
    assert np.array_equal(sigmoid_gradient(a).shape, a.shape)
    
    
if __name__ == '__main__':
    X, y = load_data('ex3data1.mat')
    display_data(X, 2)
    
    num_samples = y.shape[0]
    
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    
    y_vec = convert_labels_to_vectors(y, num_labels)
    
    print('Test cost function with given params...')
    theta1, theta2 = load_params('ex4weights.mat')
    params = unroll_params(theta1, theta2)
    reg = 0
    cost = cost_neural_network(params,
                               X, y_vec,
                               reg)
    print('Without regularization, the cost should be about 0.287629:', cost)
    reg = 1
    cost = cost_neural_network(params,
                               X, y_vec,
                               reg)
    print('Without regularization , the cost should be about 0.383770:', cost)
    
    test_unroll_roll_params()
    test_convert_theta_to_Wb()
    test_convert_labels_to_vectors()
    test_sigmoid_gradient()
    
    initial_theta1 = rand_initialize_weights(input_layer_size, 
                                             hidden_layer_size)
    initial_theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
    initial_params = unroll_params(initial_theta1, initial_theta2)
    
    print('Train the network using (my) gradient descent...')
    params_history = mloptim.gradient_descent(initial_params, X, y_vec, 
                                              jacobien_neural_network, 
                                              alpha=0.3, n_iter=5, reg=0.5)
    cost_history = compute_cost_history(params_history,
                                        X, y_vec,
                                        reg)
    mlutils.plot_costs_vs_number_of_iterations(cost_history)
    y_hat = predict(input_layer_size, hidden_layer_size, num_labels,
                    params_history[-1, :],
                    X)
    accuracy = sum(y_hat==y)/num_samples * 100
    print('Train accuracy: ', accuracy)
    print('Gradient descent with alpha = 0.3, lambda = 0.5')
    print('num_iter    accuracy')
    print('20          35.76')
    print('40          57.58')
    print('80          61.86')
    print('200         77.74')
    print('400         87.12')
    print('500         88.78')
    print('900         91.66')
    print('1300        92.68')
    print('1700        93.66')
    print('2100        93.66')
    print('2500        94.58')
    
    print('Train the network using BFGS (scipy)...')
    reg = 0.5
    result = minimize(cost_neural_network, initial_params, 
                      args=(X, y_vec, reg), jac=jacobien_neural_network,
                      options={'disp': True, 'maxiter': 5})
    y_hat_BFGS = predict(input_layer_size, hidden_layer_size, num_labels,
                    result.x, X)
    accuracy_BFGS = sum(y_hat_BFGS==y)/num_samples * 100
    print('Train accuracy: ', accuracy_BFGS)    
