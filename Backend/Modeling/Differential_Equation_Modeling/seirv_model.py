import numpy as np
from scipy.integrate import odeint
from Backend.Modeling.Differential_Equation_Modeling.model_params import params_SEIRV_fixed, params_SEIRV_fit
from scipy.optimize import curve_fit, leastsq
import matplotlib.pyplot as plt
import lmfit


def seirv_model_pipeline(y_true:np.array, start_vals:tuple):

    # Get number of days to forecast
    num_days = len(y_true)-1

    # Create a grid of time points (in days)
    t_grid = np.linspace(0, num_days, num_days + 1)

    # Get start guess for parameters that are fitted:
    fit_params_start_guess = (params_SEIRV_fit['beta'],)

    # Fit parameters:
    opt_params, success = leastsq(
        func=fit_model,
        x0=fit_params_start_guess,
        args=(t_grid, start_vals, y_true)
    )

    # Apply optimal parameters to get the results for all compartments:
    fixed_params = [param for param in params_SEIRV_fixed.values()]
    fitted_and_fixed_params = tuple(opt_params) + tuple(fixed_params)

    # Retrieve values for each compartment over time:
    S, E, I, R, V = solve_ode(y0=start_vals, t=t_grid, params=fitted_and_fixed_params).T

    # Return predicted infections
    return I


def fit_model(params_to_fit, t_grid, start_vals, y_true):

    fit_result = solve_ode_and_return_estimates_only(start_vals, t_grid, params_to_fit)
    residual = y_true - fit_result

    return residual


def solve_ode_and_return_estimates_only(y0, t_grid, fit_params):

    beta = fit_params[0]
    gamma = params_SEIRV_fixed['gamma']
    delta = params_SEIRV_fixed['delta']
    theta = params_SEIRV_fixed['theta']

    # lambda function as shown here: https://www.kaggle.com/baiyanren/modified-seir-model-for-covid-19-prediction-in-us
    seir_ode = lambda y, t: seirv_ode(y, t, beta, delta, gamma, theta)

    ode_result = odeint(func=seir_ode, y0=y0, t=t_grid)

    return ode_result[:, 2] # return only Infection counts


def solve_ode(y0, t, params):
    ode_result = odeint(func=seirv_ode, y0=y0, t=t, args=params)

    return ode_result



def seirv_ode(y, t, beta, delta, gamma, theta):
    S, E, I, R, V = y
    N = S + E + I + R + V

    ## Differential equations:
    # Susceptible:
    dSdt = -beta/ N * S * I

    # Exposed:
    dEdt = beta / N * S * I + theta * beta / N * V * I - delta * E

    # Infectious:
    dIdt = delta * E - gamma * I

    # Recovered:
    dRdt = gamma * I

    # Vaccinated:
    dVdt = - theta * beta/ N * V * I

    return dSdt, dEdt, dIdt, dRdt, dVdt