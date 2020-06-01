import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

import machinelearn.models as mlmodels
import machinelearn.optimization as mloptim
import machinelearn.utils as mlutils

from sklearn.linear_model import LinearRegression

def load_data(filename):
    data = pd.read_csv(filename, names=['surface', 'bedrooms', 'price'])
    inputs = data.loc[:, ['surface', 'bedrooms']].to_numpy()
    outputs = data.loc[:, 'price'].to_numpy()
    return inputs, outputs

def predict_price(theta, surface, bedrooms, mean, std):
    normalized_input = (np.array([[surface, bedrooms]]) - mean) / std
    return mlmodels.linear_model(theta, normalized_input)

if __name__ == '__main__':
    plt.close('all')

    inputs, outputs = load_data('ex1data2.txt')
    normalized_inputs, mean, std = mlutils.feature_normalize(inputs)
    theta_history = mloptim.gradient_descent(np.array([0, 0, 0]),
                                            normalized_inputs, outputs,
                                            mlmodels.jacobien_rms_linear,
                                            alpha=0.1, n_iter=50)
    cost_history = mlutils.costs(theta_history, normalized_inputs, outputs,
                                mlmodels.cost_rms, mlmodels.linear_model)
    mlutils.plot_costs_vs_number_of_iterations(cost_history)
    predicted_price = predict_price(theta_history[-1], 1600, 3, mean, std)
    print('Optimal theta (my algo):', theta_history[-1])

    theta_opt = mloptim.normal_equation(inputs, outputs)
    predicted_price_theta_opt = mlmodels.linear_model(theta_opt, np.array([[1600, 3]]))
    print('Optimal theta (normal):', theta_opt)

    print('Use sklearn...')
    model = LinearRegression()
    model.fit(inputs, outputs)
    opt_theta_sklearn = np.insert(model.coef_, 0, model.intercept_)
    print('Optimal theta (sklearn):', opt_theta_sklearn)

    print('Predicted price of a 1600 sq-ft, 3 br house '
            '(using my algo):', predicted_price)
    print('Predicted price of a 1600 sq-ft, 3 br house '
            '(using normal equations):', predicted_price_theta_opt)
