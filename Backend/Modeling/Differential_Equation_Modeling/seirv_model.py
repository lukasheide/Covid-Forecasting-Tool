import numpy as np
from scipy.integrate import odeint
from Backend.Modeling.Differential_Equation_Modeling.model_params import params_SEIRV_fixed, params_SEIRV_fit
from Backend.Modeling.Differential_Equation_Modeling.optimization_functions import weigh_residuals
from scipy.optimize import curve_fit, leastsq
import matplotlib.pyplot as plt
from Backend.Visualization.modeling_results import plot_train_and_val_infections


def seirv_pipeline(y_train: np.array, start_vals_fixed: tuple, forecast_horizon=14):

    """
    Takes as input a numpy array containing daily infection counts for the training period and a tuple containing
    the population size N and fixed starting values for each compartment: I0, R0 and V0. E0 and S0 are fitted.

    The starting value for the exposed and susceptible compartments are then fitted as well as the force of infection
    parameter beta.

    The obtained fitted parameters are then applied to the desired forecasting period.

    The result dicitonary contains the predicted daily infections, the model parameters and starting values for the
    forecasting step.
    """

    ## 1) Model fitting
    # Run model fitting:
    fitting_result = fit_seirv_model(y_train, start_vals_fixed)

    ## 2) Model application
    # Set up starting values and model parameters used for applying the model in the next step:
    start_vals = merge_fitted_and_fixed_start_vals(fitted_start_vals=fitting_result['fitted_params'], fixed_start_vals=fitting_result['end_vals'])
    model_params = setup_model_params(fitted_model_params=fitting_result['fitted_params'])

    # Run forecasting:
    pred_daily_infections = forecast_seirv(all_model_params=model_params, y0=start_vals)

    ## 3) Bundling up results:
    results_dict = {
        'pred_daily_infections': pred_daily_infections,
        'model_params': model_params,
        'model_start_vals': start_vals
    }

    return results_dict


def fit_seirv_model(y_train: np.array, start_vals_fixed: tuple) -> dict:

    """
    Takes as input a numpy array with the daily new infections and a tuple containing the population size N and fixed
    starting values for each compartment: I0, R0 and V0. E0 and S0 are fitted.
    Running the model requires the following parameters: Beta, gamma, delta and theta. Beta is fit, while the others
    are fixed and imported from a settings function turing model training/fitting.
    The model is trained/fitted using a curve-fitting approach with weighted root mean squared error as the metric.
    The weights ensure that the residuals near the end of the fitting period receive higher penalties.

    This function returns a dictionary containing:
    1) A dictionary containing the fitted parameters
    2) The end-values for each compartment (how many individuals are in which compartment at the end of training)
    3) A time series for the number of individuals in each compartment over time
    4) The daily infection counts produced by the fitted model
    """

    ## 1) Create variables needed for applying the model fitting part:
    # Compute length of train_data:
    num_days_train = len(y_train) - 1

    # Create a grid of time points (in days)
    t_grid_train = np.linspace(0, num_days_train, num_days_train + 1)


    ## 2) Get start guess for parameters that are fitted as a tuple:
    fit_params_start_guess = (params_SEIRV_fit['beta'], 1234)


    ## 3) Call fitting function:
    opt_params, success = leastsq(
        func=get_weighted_residuals,
        x0=fit_params_start_guess,
        args=(t_grid_train, start_vals_fixed, y_train)
    )


    ## 4) Prepare model parameters and start values to run the model again:
    # Model params:
    fixed_model_params = [param for param in params_SEIRV_fixed.values()]
    fitted_and_fixed_model_params = tuple(opt_params[:1]) + tuple(fixed_model_params)

    # Compute starting values for each compartment:
    N = start_vals_fixed[0]
    E0 = opt_params[1]
    I0 = start_vals_fixed[1]
    R0 = start_vals_fixed[2]
    V0 = start_vals_fixed[3]
    S0 = N - E0 - I0 - R0 - V0

    # pack all together and add start value 0 for cumulated infections:
    y0_train = (S0, E0, I0, R0, V0, 0)

    ## 5) Apply fitted parameters to get the end values for all compartments:
    # Retrieve values for each compartment over time:
    S, E, I, R, V, cum_infections_fitted, daily_infections_fitted = solve_ode(y0=y0_train,
                                                                              t=t_grid_train,
                                                                              params=fitted_and_fixed_model_params)

    ## 6) Prepare results for returning them
    # Pack retrieved values for each compartment over time into one tuple:
    compartment_time_series = (S, E, I, R, V)

    # Compute end values:
    end_vals = (S[-1], E[-1], I[-1], R[-1], V[-1])

    # Fit params:
    fitted_params = {
        'beta': opt_params[0],
        'E0': opt_params[1]
    }


    # Bundle them all up in one dictionary:
    fitting_results = {
        'fitted_params': fitted_params,
        'compartment_time_series': compartment_time_series,
        'daily_infections':daily_infections_fitted,
        'end_vals': end_vals
    }

    return fitting_results


def forecast_seirv(all_model_params:tuple, y0:np.array, forecast_horizon=14) -> np.array:
    ## 1) Set up variables needed for applying the model:
    # Create a grid of time points (in days)
    t_grid = np.linspace(0, forecast_horizon, forecast_horizon + 1)

    ## 2) Apply values to produce forecast:
    S_pred, E_pred, I_pred, R_pred, V_pred, cum_infections, daily_infections = \
        solve_ode(y0=y0, t=t_grid, params=all_model_params)


    return daily_infections



def merge_fitted_and_fixed_start_vals(fitted_start_vals, fixed_start_vals) -> tuple:
    """
    Combines fitted and fixed starting_values so that they can be forwarded as one tuple containing all starting values.
    """

    N = fixed_start_vals[0]

    E0 = fitted_start_vals['E0']
    I0 = fixed_start_vals[1]
    R0 = fixed_start_vals[2]
    V0 = fixed_start_vals[3]

    S0 = N - E0 - I0 - R0 - V0

    # Add 0 for cumulated infection counts:
    return S0, E0, I0, R0, V0, 0


def setup_model_params(fitted_model_params):
    """
    Combines fitted and fixed model parameter into one tuple.
    """

    beta = fitted_model_params['beta']
    gamma = params_SEIRV_fixed['gamma']
    delta = params_SEIRV_fixed['delta']
    theta = params_SEIRV_fixed['theta']

    return beta, gamma, delta, theta


def get_weighted_residuals(params_to_fit, t_grid, start_vals, y_true):
    fit_result = solve_ode_for_fitting_partly_fitted_y0(start_vals, t_grid, params_to_fit)

    # add value at t=0 to fit_result:
    y_fit = np.append(y_true[0], fit_result)

    # compute abs difference between predicted and actual infection counts:
    residual = y_true - y_fit

    # weight residuals:
    weighted_residuals = weigh_residuals(residual)

    return weighted_residuals


def solve_ode_for_fitting_partly_fitted_y0(fixed_start_vals, t_grid, fit_params):
    ## 1) Pull apart fitting params:
    beta = fit_params[0]
    E0 = fit_params[1]

    ## 2) Setup
    #  2a) Get y0:
    #  Starting values for I, R and V are given. E0 is fitted and S0 is then computed as the last missing value.
    N = fixed_start_vals[0]
    I0 = fixed_start_vals[1]
    R0 = fixed_start_vals[2]
    V0 = fixed_start_vals[3]
    S0 = N - E0 - I0 - R0 - V0

    # pack all together and add cumulated infections:
    y0 = (S0, E0, I0, R0, V0, 0)

    # 2b) Get fixed model params:
    gamma = params_SEIRV_fixed['gamma']
    delta = params_SEIRV_fixed['delta']
    theta = params_SEIRV_fixed['theta']

    # lambda function as shown here: https://www.kaggle.com/baiyanren/modified-seir-model-for-covid-19-prediction-in-us
    # this circumvents issues with the scipy odeint function, which can only handle a predefined number of params
    seir_ode = lambda y0, t: ode_seirv(y0, t, beta, gamma, delta, theta)

    ode_result = odeint(func=seir_ode, y0=y0, t=t_grid).T

    predicted_daily_infections = compute_daily_infections(ode_result[5, :])

    return predicted_daily_infections  # return only Infection counts



def solve_ode_for_fitting_fixed_y0(y0, t_grid, fit_params):
    beta = fit_params[0]
    gamma = params_SEIRV_fixed['gamma']
    delta = params_SEIRV_fixed['delta']
    theta = params_SEIRV_fixed['theta']

    # lambda function as shown here: https://www.kaggle.com/baiyanren/modified-seir-model-for-covid-19-prediction-in-us
    # this circumvents issues with the scipy odeint function, which can only handle a predefined number of params
    seir_ode = lambda y0, t: ode_seirv(y0, t, beta, gamma, delta, theta)

    ode_result = odeint(func=seir_ode, y0=y0, t=t_grid).T

    predicted_daily_infections = compute_daily_infections(ode_result[5, :])

    return predicted_daily_infections  # return only Infection counts


def solve_ode(y0, t, params):
    ode_result = odeint(func=ode_seirv, y0=y0, t=t, args=params).T

    # cumulated infections:
    cum_infections = ode_result[5, :]

    # compute daily new infections from cumulated infections:
    daily_infections = compute_daily_infections(cum_infections)

    # combine the numbers of individuals for each compartment over time + cumulated infections + daily infections:
    # to ensure that the sizes of both arrays fit the starting values for each compartment are dropped (t=0):
    cropped_ode_result = ode_result[:, 1:]

    result = np.vstack([cropped_ode_result, daily_infections])

    return result


def ode_seirv(y0, t, beta, gamma, delta, theta):
    S, E, I, R, V, V_cum = y0
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


def seirv_model_pipeline_DEPRECATED(y_train: np.array, start_vals_fitting: tuple, len_pred_period = 14):
    """
    Pipeline functions first calls a fitting function to obtain the optimal set of parameters that are fitted to the data.
    Having obtained the optimal parameters the differential equation system is solved again to obtain the number of
    individuals over time for each compartment.

    Procedure:
    1) Model fitting
    - Input:
    -- start_vals (S0, E0, I0, R0, V0)
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
        func=get_weighted_residuals,
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