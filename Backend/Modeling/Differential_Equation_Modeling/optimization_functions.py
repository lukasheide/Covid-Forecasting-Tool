import numpy as np
from Backend.Modeling.Differential_Equation_Modeling.model_params import LEAST_SQUARES_WEIGHT
import matplotlib.pyplot as plt


def weigh_residuals(residuals: np.array) -> np.array:

    param_lambda = LEAST_SQUARES_WEIGHT['lambda']

    # formula: Exponential decay function from Data Analytics 1 Lecture WS2020/21
    n = len(residuals)
    weights = [2 ** (-param_lambda*(n-t)) for t in range(0, n)]

    # Plot for debugging purposes:
    # plt.plot(weights)
    # plt.show()
    # pass

    resid = residuals.copy()

    return np.multiply(resid, weights)