import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from machinelearn import utils

def linear_model(theta, inputs):
    inputs_extended = utils.extended_inputs(inputs)
    return inputs_extended @ theta 

def logistic_model(theta, inputs):
    inputs_extended = utils.extended_inputs(inputs)
    return sigmoid(inputs_extended @ theta)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def predict_logistic(theta, inputs, threshold):
    prob = logistic_model(theta, inputs)
    prob[prob >= threshold] = 1
    prob[prob < threshold] = 0
    return prob


def cost_rms(theta, inputs, outputs, model, reg=0):
    m = outputs.shape[0]
    cost = 1/(2*m) * np.sum((model(theta, inputs) - outputs)**2)
    cost_reg = cost_regularize(reg, m, theta)
    return cost + cost_reg

def cost_logistic(theta, inputs, outputs, model, reg=0):
    m = outputs.shape[0]
    term_1 = outputs * np.log(model(theta, inputs))
    term_2 = (1 - outputs) * np.log(1 - model(theta, inputs))
    cost_log = -1/m * np.sum(term_1 + term_2)
    cost_reg = cost_regularize(reg, m, theta)
    return cost_log + cost_reg

def cost_regularize(reg, n_samples, theta):
    return reg/(2*n_samples) * np.sum(theta**2)


def jacobien_rms_linear(theta, inputs, outputs, reg=0):
    return jacobien_rms_regularize(theta, inputs, outputs, linear_model, reg)

def jacobien_logistic(theta, inputs, outputs, reg=0):
    return jacobien_rms_regularize(theta, inputs, outputs, logistic_model, reg)

def jacobien_rms_regularize(theta, inputs, outputs, model, reg):
    n_samples = outputs.shape[0]
    jacobien = jacobien_rms(theta, inputs, outputs, model)
    jacobien_reg = jacobien_regularize(reg, n_samples, theta)
    return jacobien + jacobien_reg

def jacobien_rms(theta, inputs, outputs, model):
    n_samples = outputs.shape[0]
    return 1/n_samples * ((model(theta, inputs) - outputs) 
                          @ utils.extended_inputs(inputs))

def jacobien_regularize(reg, n_samples, theta):
    out = reg/n_samples * theta[1:]
    return np.insert(out, 0, 0)

