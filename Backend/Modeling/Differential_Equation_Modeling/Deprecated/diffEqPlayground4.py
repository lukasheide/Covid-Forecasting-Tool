import lmfit
from lmfit import minimize, Parameters
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit, leastsq
import matplotlib.pyplot as plt


def runner(y_true):

    num_days = 28
    t_grid = np.linspace(0, num_days, num_days + 1)

    y0 = 280_000, 1_000, 19_000

    beta = 0.4

    fit_params_start_guess = [beta]

    opt_params, success = leastsq(
        func=fit_model,
        x0=fit_params_start_guess,
        args=(t_grid, y0, y_true)
    )

    # Compute results of fitted model:
    # 1) Get fixed params:
    fixed_params = get_fixed_params()

    beta = opt_params[0]
    gamma = fixed_params[0]

    result = odeint(func=sir_model, y0=y0, t=t_grid, args=(beta, gamma))

    pass


def sir_model(y, t, beta, gamma):
    S, I, R = y
    N = S + I + R

    # Susceptibles:
    dSdt = -beta / N * S * I

    # Infectious:
    dIdt = beta / N * S * I - gamma * I

    # Recovered:
    dRdt = gamma * I

    return dSdt, dIdt, dRdt


def calc_ode(y0, t_grid, fit_params):
    fixed_params = get_fixed_params()

    beta = fit_params[0]
    gamma = fixed_params[0]

    # lambda function as shown here: https://www.kaggle.com/baiyanren/modified-seir-model-for-covid-19-prediction-in-us
    seir_ode = lambda y, t: sir_model(y, t, beta, gamma)

    ode_result = odeint(func=seir_ode, y0=y0, t=t_grid)

    return ode_result[:, 1] # return only Infection counts


def fit_model(params_to_fit, t_grid, start_vals, y_true):

    fit_result = calc_ode(start_vals, t_grid, params_to_fit)
    residual = y_true - fit_result

    return residual


def get_fixed_params():
    gamma = 1/14
    return [gamma]
