import numpy as np
from scipy.integrate import odeint
from Backend.Modeling.Differential_Equation_Modeling.model_params import params_SEIRV_fixed, params_SEIRV_fit
from Backend.Modeling.Differential_Equation_Modeling.optimization_functions import weigh_residuals
from scipy.optimize import curve_fit, leastsq
import matplotlib.pyplot as plt
from Backend.Visualization.modeling_results import plot_train_and_val_infections



def seirv_model_pipeline(y_train: np.array, start_vals_fitting: tuple, len_pred_period = 14):
    """
    Pipeline functions first calls a fitting function to obtain the optimal set of parameters that are fitted to the data.
    Having obtained the optimal parameters the differential equation system is solved again to obtain the number of
    individuals over time for each compartment.

    Procedure:
    1) Model fitting
    - Input:
    -- start_vals (S0, E0, I0, R0)
    - Output:
    -- end_vals (St, Et, It, Rt)
    -- beta_t

    :param y_train: Value to be predicted. In this case the daily infection numbers.
    :param start_vals_fitting: Starting values for each of the five compartments for the: (S0,E0,I0,R0,V0)
    :return: Predicted daily infections
    """

    # Get length of model fitting period:
    num_days_train = len(y_train) - 1

    # Create a grid of time points (in days)
    t_grid_train = np.linspace(0, num_days_train, num_days_train + 1)

    # Add 0 to starting values for tracking the cumulated number of infections in the model run:
    # Start_vals always refers to the number of individuals per compartment at time t.
    # y_t also includes cumulated infections.
    y0_train = start_vals_fitting + (0,)

    # Get start guess for parameters that are fitted as a tuple:
    fit_params_start_guess = (params_SEIRV_fit['beta'],)

    # Fit parameters:
    opt_params, success = leastsq(
        func=fit_model,
        x0=fit_params_start_guess,
        args=(t_grid_train, y0_train, y_train)
    )

    # Apply optimal parameters to get the end values for all compartments:
    fixed_params = [param for param in params_SEIRV_fixed.values()]
    fitted_and_fixed_params = tuple(opt_params) + tuple(fixed_params)

    # Retrieve values for each compartment over time:
    S, E, I, R, V, cum_infections_fitted, daily_infections_fitted = solve_ode(y0=y0_train,
                                                                              t=t_grid_train,
                                                                              params=fitted_and_fixed_params)


    ## FORECASTING: Applying fitted model

    # Retrieve start_values for each compartment from results of previous model run:
    start_vals_predict = (S[-1], E[-1], I[-1], R[-1], V[-1])
    y0_predict = start_vals_predict + (0,)

    # Set up parameters for prediction run:
    fixed_params_predict = [param for param in params_SEIRV_fixed.values()]
    fitted_and_fixed_params_predict = tuple(opt_params) + tuple(fixed_params)


    ### For debugging:
    plot_train_and_val_infections(y_train=y_train, y_val=np.append(y_train[0], daily_infections_fitted))


    # Set up t_grid:
    t_grid_train = np.linspace(0, len_pred_period, len_pred_period + 1)

    S_pred, E_pred, I_pred, R_pred, V_pred, cum_infections_fitted, daily_infections_fitted = solve_ode(y0=y0_predict,
                                                                                                   t=t_grid_train,
                                                                                                   params=fitted_and_fixed_params_predict)

    return daily_infections_fitted


def fit_model(params_to_fit, t_grid, start_vals, y_true):
    fit_result = solve_ode_and_return_estimates_only(start_vals, t_grid, params_to_fit)

    # add value at t=0 to fit_result:
    y_fit = np.append(y_true[0], fit_result)

    # compute abs difference between predicted and actual infection counts:
    residual = y_true - y_fit

    # weight residuals:
    weighted_residuals = weigh_residuals(residual)

    return weighted_residuals


def solve_ode_and_return_estimates_only(y0, t_grid, fit_params):
    beta = fit_params[0]
    gamma = params_SEIRV_fixed['gamma']
    delta = params_SEIRV_fixed['delta']
    theta = params_SEIRV_fixed['theta']

    # lambda function as shown here: https://www.kaggle.com/baiyanren/modified-seir-model-for-covid-19-prediction-in-us
    # this circumvents issues with the scipy odeint function, which can only handle a predefined number of params
    seir_ode = lambda y0, t: seirv_ode(y0, t, beta, delta, gamma, theta)

    ode_result = odeint(func=seir_ode, y0=y0, t=t_grid).T

    predicted_daily_infections = compute_daily_infections(ode_result[5,:])

    return predicted_daily_infections  # return only Infection counts


def solve_ode(y0, t, params):
    ode_result = odeint(func=seirv_ode, y0=y0, t=t, args=params).T

    # cumulated infections:
    cum_infections = ode_result[5, :]

    # compute daily new infections from cumulated infections:
    daily_infections = compute_daily_infections(cum_infections)

    # combine the numbers of individuals for each compartment over time + cumulated infections + daily infections:
    # to ensure that the sizes of both arrays fit the starting values for each compartment are dropped (t=0):
    cropped_ode_result = ode_result[:, 1:]

    result = np.vstack([cropped_ode_result, daily_infections])

    return result


def seirv_ode(y, t, beta, delta, gamma, theta):
    S, E, I, R, V, V_cum = y
    N = S + E + I + R + V

    ## Differential equations:
    # Susceptible:
    dSdt = -beta / N * S * I

    # Exposed:
    dEdt = beta / N * S * I + theta * beta / N * V * I - delta * E

    # Infectious:
    dIdt = delta * E - gamma * I

    # Recovered:
    dRdt = gamma * I

    # Vaccinated:
    dVdt = - theta * beta / N * V * I

    ## Cumulated Infection Counts:
    dICumdt = delta * E

    return dSdt, dEdt, dIdt, dRdt, dVdt, dICumdt


def compute_daily_infections(cumulated_infections:np.array) -> np.array:
    return np.diff(cumulated_infections)