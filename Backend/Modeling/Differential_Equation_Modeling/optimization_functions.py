import numpy as np
from Backend.Modeling.Differential_Equation_Modeling.model_params import LEAST_SQUARES_WEIGHT
import matplotlib.pyplot as plt
from sklearn import preprocessing


def weigh_residuals(residuals: np.array) -> np.array:

    param_lambda = LEAST_SQUARES_WEIGHT['lambda']

    # formula: Exponential decay function from Data Analytics 1 Lecture WS2020/21
    n = len(residuals)
    weights = [2 ** (-param_lambda*(n-t)) for t in range(0, n)]

    # standardise:
    weights = list(np.divide(weights, weights[0]))

    # Plot for debugging purposes:
    # plt.plot(weights)
    # plt.show()

    resid = residuals.copy()

    return np.multiply(resid, weights)


def weigh_residuals_reversed(residuals: np.array) -> np.array:

    param_lambda = LEAST_SQUARES_WEIGHT['lambda_inverse']

    # formula: Exponential decay function from Data Analytics 1 Lecture WS2020/21
    n = len(residuals)
    weights = [2 ** (-param_lambda*(n-t)) for t in range(0, n)]

    # standardise:
    weights = list(np.divide(weights, weights[0]))

    # Plot for debugging purposes:
    # plt.plot(weights)
    # plt.show()

    resid = residuals.copy()

    return np.multiply(resid, weights)