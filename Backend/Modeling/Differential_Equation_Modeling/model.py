import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from Backend.Modeling.Differential_Equation_Modeling.model_params import params_SEIRV_fixed
import matplotlib.pyplot as plt


def run_SEIRV_model():
    num_days = 20

    y0 = get_starting_values()

    # Create a grid of time points (in days)
    t = np.linspace(0, num_days, num_days + 1)

    # Integrate the differential equations over the time grid, t.
    ret = odeint(deriv_SEIRV_model, y0, t)
    S, E, I, R, V, I_cum = ret.T

    return S, E, I, R, V, I_cum


def fit_SEIRV_model(num_days_train, train_data, starting_values):
    """
    Takes as input a training data set, a number of fixed parameters and fit the parameters left open to the training data.
    :param num_days_train: Time duration of training data set
    :param train_data: Numpy Array containing training data
    :param starting_values: Starting values for t=0  for each compartment
    :return: fitted model parameters and predictions
    """

    y0 = starting_values

    # Get overall number of individuals in model:
    N = sum(starting_values[0:4])

    # Create a grid of time points (in days)
    t = np.linspace(0, num_days_train, num_days_train + 1)

    fixed_params = get_params(0)
    fit_params = {'beta': 0.5}

    ret = odeint(deriv_SEIRV_model_v2, y0, t, args=(
        N,
        fit_params['beta'],
        fixed_params['gamma'],
        fixed_params['delta'],
        fixed_params['theta']
    ))

    S, E, I, R, V, I_cum = ret.T

    return S, E, I, R, V, I_cum


def deriv_SEIRV_model_v2(y, t, N, beta, gamma, delta, theta):
    # Unpack values contained in y
    S, E, I, R, V, I_cum = y

    ## Differential equations:
    # Susceptibles:
    dSdt = -beta/ N * S * I

    # Exposed:
    dEdt = beta / N * S * I + theta * beta / N * V * I - delta * E

    # Infectious:
    dIdt = delta * E - gamma * I

    # Recovered:
    dRdt = gamma * I

    # Vaccinated:
    dVdt = - theta * beta/ N * V * I

    ## Cumulated Infections:
    dI_cumdt = delta * E

    return dSdt, dEdt, dIdt, dRdt, dI_cumdt, dVdt


def deriv_SEIRV_model(y, t):
    # Unpack values contained in y
    S, E, I, R, V, I_cum = y

    N = S + E + I + R + V

    # Get model parameters depending on t
    params = get_params(t)

    ## Differential equations:
    # Susceptibles:
    dSdt = -params['beta'] / N * S * I

    # Exposed:
    dEdt = params['beta'] / N * S * I + params['theta'] * params['beta'] / N * V * I - params['delta'] * E

    # Infectious:
    dIdt = params['delta'] * E - params['gamma'] * I

    # Recovered:
    dRdt = params['gamma'] * I

    # Vaccinated:
    dVdt = - params['theta'] * params['beta'] / N * V * I

    ## Cumulated Infections:
    dI_cumdt = params['delta'] * E

    return dSdt, dEdt, dIdt, dRdt, dI_cumdt, dVdt


def get_params(t):
    """
    computes params at time step t
    :return: dictionary containing all params
    """
    params = {
        'theta': 0.1,
        'delta': 1 / 3,
        'gamma': 1 / 11,
        'beta': 0.5
    }

    return params


def get_starting_values():
    S0 = 120000
    E0 = 250
    I0 = 1000
    R0 = 300000 - S0 - E0 - I0
    I_cum = I0 + R0
    V0 = 150000

    return S0, E0, I0, R0, I_cum, V0
