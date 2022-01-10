import numpy as np
from scipy.integrate import odeint
from Backend.Modeling.Differential_Equation_Modeling.model_params import params_SEIRV_fixed, params_SEIRV_fit, \
    draw_value_from_param_distribution, draw_random_beta
from Backend.Modeling.Differential_Equation_Modeling.optimization_functions import weigh_residuals
from scipy.optimize import curve_fit, leastsq, least_squares
import matplotlib.pyplot as plt
from Backend.Visualization.modeling_results import plot_train_and_val_infections, plot_train_and_fitted_infections_line_plot, plot_train_fitted_and_validation
from Backend.Modeling.Util.pipeline_util import models_params_to_dictionary, models_compartment_values_to_dictionary


def seirv_pipeline(y_train: np.array, start_vals_fixed: tuple, forecast_horizon=14,
                   allow_randomness_fixed_params=False, allow_randomness_fixed_beta=False, random_runs=100, pred_quantile=0.9):
    """
    Takes as input a numpy array containing daily infection counts for the training period and a tuple containing
    the population size N and fixed starting values for each compartment: I0, R0 and V0. E0 and S0 are fitted.

    The starting value for the exposed and susceptible compartments are then fitted as well as the force of infection
    parameter beta.

    The obtained fitted parameters are then applied to the desired forecasting period.

    The result dictionary contains the predicted daily infections, the model parameters and starting values for the
    forecasting step.
    """

    ## 1) Model fitting
    # Run model fitting:
    fitting_result = fit_seirv_model(y_train, start_vals_fixed)

    ## 2) Model application / forecasting:

    ## 2.1 Model Run with fixed parameters derived from fitting:
    # 2.1.1 Set up starting values and model parameters used for applying the model in the next step:
    model_params = setup_model_params_for_forecasting_after_fitting(fitted_model_params=fitting_result['fitted_params'],
                                                                    random_draw_fixed_params=allow_randomness_fixed_params)
    # 2.1.2 Run forecasting - Starting point: beginning of training period
    pred_daily_infections_from_start = forecast_seirv(all_model_params=model_params,
                                                      y0=fitting_result['start_vals'] + (0,),
                                                      forecast_horizon=forecast_horizon + len(y_train) - 1)

    # 2.2.1) No Randomness:
    if not allow_randomness_fixed_params and not allow_randomness_fixed_beta:

        # Run forecasting - Starting point: after training period
        pred_daily_infections = forecast_seirv(all_model_params=model_params,
                                               y0=fitting_result['end_vals'] + (0,),
                                               forecast_horizon=forecast_horizon)

    # 2.2.2) Random Runs:
    else:
        results_list = []
        for run in range(random_runs):
            # 2.2.2.1) Set up starting values and model parameters used for applying the model in the next step:
            model_params_randomness = setup_model_params_for_forecasting_after_fitting(
                fitted_model_params=fitting_result['fitted_params'], random_draw_fixed_params=allow_randomness_fixed_params, random_draw_beta=allow_randomness_fixed_beta)

            # 2.2.2.2)
            # Run forecasting - Starting point: after training period
            pred_daily_infections = forecast_seirv(all_model_params=model_params_randomness,
                                                   y0=fitting_result['end_vals'] + (0,),
                                                   forecast_horizon=forecast_horizon)
            # 2.2.2.3)
            # Push results to list:
            results_list.append({
                'model_params':model_params_randomness,
                'pred_daily_infections':pred_daily_infections
            })

        ## Computations based on multiple runs:
        # Setup numpy array:
        all_pred_daily_infections = np.array([model_run['pred_daily_infections'] for model_run in results_list])

        # Compute average prediction:
        pred_daily_infections_mean = np.mean(all_pred_daily_infections, axis=0)

        # 90% Quantile:
        pred_daily_infections_upper_quantile = np.quantile(all_pred_daily_infections, q=pred_quantile, axis=0)
        # 10% Quantile:
        pred_daily_infections_lower_quantile = np.quantile(all_pred_daily_infections, q=1-pred_quantile, axis=0)

        # FOR DEBUGGING:
        # plt.plot(pred_daily_infections_mean)
        # plt.plot(pred_daily_infections_upper_quantile)
        # plt.plot(pred_daily_infections_lower_quantile)
        # plt.show()


    ## Prepare everything for return:
    # Handle randomness:
    if not allow_randomness_fixed_params and not allow_randomness_fixed_beta:
        pred_daily_infections_upper_quantile = None
        pred_daily_infections_lower_quantile = None

    # Bundle up model params and start vals to dictionaries to return them:
    model_params_as_dictionary = models_params_to_dictionary(model_params)
    model_start_vals_as_dictionary = models_compartment_values_to_dictionary(fitting_result['end_vals'])

    ## 3) Bundling up results:
    results_dict = {
        'y_pred_including_train_period': pred_daily_infections_from_start,
        'y_pred_without_train_period': pred_daily_infections,
        'y_pred_without_train_period_upper_bound': pred_daily_infections_upper_quantile,       # None if randomness was excluded!
        'y_pred_without_train_period_lower_bound': pred_daily_infections_lower_quantile,       # None if randomness was excluded!
        'model_params_forecast_period': model_params_as_dictionary,
        'model_start_vals_forecast_period': model_start_vals_as_dictionary
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
    fit_params_start_guess = (params_SEIRV_fit['beta']['mean'], 678, 789)

    ## 3) Get values for fixed model parameters:
    fixed_params = {
        't_grid': t_grid_train,
        'fixed_model_params': {
            'gamma_I': params_SEIRV_fixed['gamma_I']['mean'],
            'gamma_U': params_SEIRV_fixed['gamma_U']['mean'],
            'delta': params_SEIRV_fixed['delta']['mean'],
            'theta': params_SEIRV_fixed['theta']['mean'],
            'rho': params_SEIRV_fixed['rho']['mean']
        }
    }

    ## 4) Call fitting function:
    ret = least_squares(
        fun=compute_weighted_residuals,
        x0=fit_params_start_guess,
        args=(fixed_params, start_vals_fixed, y_train),
        method='trf',
        ftol=1e-10,
        xtol=1e-10,
        bounds=(0, np.inf)
    )

    # get optimal parameters from least squares result:
    opt_params = tuple(ret.x)

    ## 5) Prepare model parameters and start values to run the model again:
    # Model params:
    fixed_params = [param['mean'] for param in params_SEIRV_fixed.values()]
    fitted_and_fixed_model_params = tuple(opt_params[:1]) + tuple(fixed_params)

    # Compute starting values for each compartment:
    N = start_vals_fixed[0]
    E0 = opt_params[1]
    I0 = opt_params[2]
    U0 = I0 * params_SEIRV_fixed['rho']['mean'] / (1 - params_SEIRV_fixed['rho']['mean'])
    R0 = start_vals_fixed[1]
    V0 = start_vals_fixed[2]
    S0 = N - E0 - I0 - R0 - V0

    # pack all together and add start value 0 for cumulated infections:
    y0_train = (S0, E0, I0, U0, R0, V0, 0)

    ## 6) Apply fitted parameters to get the end values for all compartments:
    # Retrieve values for each compartment over time:
    S, E, I, U, R, V, cum_infections_fitted, daily_infections_fitted = solve_ode(y0=y0_train,
                                                                                 t=t_grid_train,
                                                                                 params=fitted_and_fixed_model_params)


    #### Debugging ####
    # plot_train_and_fitted_infections_line_plot(y_train, daily_infections_fitted)


    ## 7) Prepare results for returning them
    # Pack retrieved values for each compartment over time into one tuple:
    compartment_time_series = (S, E, I, U, R, V)

    # Compute end values:
    end_vals = (S[-1], E[-1], I[-1], U[-1], R[-1], V[-1])
    start_vals = (S0, E0, I0, U0, R0, V0)

    # Fit params:
    fitted_params = {
        'beta': opt_params[0],
        'E0': opt_params[1],
        'I0': opt_params[2]
    }

    # Bundle them all up in one dictionary:
    fitting_results = {
        'fitted_params': fitted_params,
        'compartment_time_series': compartment_time_series,
        'daily_infections': daily_infections_fitted,
        'end_vals': end_vals,
        'start_vals': start_vals
    }

    return fitting_results


def forecast_seirv(all_model_params: tuple, y0: np.array, forecast_horizon=14) -> np.array:
    ## 1) Set up variables needed for applying the model:
    # Create a grid of time points (in days)
    t_grid = np.linspace(0, forecast_horizon, forecast_horizon + 1)

    ## 2) Apply values to produce forecast:
    S_pred, E_pred, I_pred, U_pred, R_pred, V_pred, cum_infections, daily_infections = \
        solve_ode(y0=y0, t=t_grid, params=all_model_params)

    return daily_infections


def merge_fitted_and_fixed_start_vals(fitted_start_vals, tot_pop_size, fixed_start_vals) -> tuple:
    """
    Combines fitted and fixed starting_values so that they can be forwarded as one tuple containing all starting values.
    """

    N = tot_pop_size

    E0 = fitted_start_vals['E0']
    I0 = fitted_start_vals['I0']
    R0 = fixed_start_vals[3]
    V0 = fixed_start_vals[4]

    S0 = N - E0 - I0 - R0 - V0

    # Add 0 for cumulated infection counts:
    return S0, E0, I0, R0, V0, 0


def setup_model_params_for_forecasting_after_fitting(fitted_model_params, random_draw_fixed_params=False, random_draw_beta=False):
    """
    Combines fitted and fixed model parameter into one tuple.
    """
    ## Beta:
    if not random_draw_beta:
        beta = fitted_model_params['beta']

    else:
        beta = draw_random_beta(
            beta_estimate=fitted_model_params['beta'],
            sd_passed=False
        )

    ## Fixed Params:
    if not random_draw_fixed_params:
        gamma_I = params_SEIRV_fixed['gamma_I']['mean']
        gamma_U = params_SEIRV_fixed['gamma_U']['mean']
        delta = params_SEIRV_fixed['delta']['mean']
        theta = params_SEIRV_fixed['theta']['mean']
        rho = params_SEIRV_fixed['rho']['mean']

    else:
        gamma_I = draw_value_from_param_distribution('gamma_I')
        gamma_U = draw_value_from_param_distribution('gamma_U')
        delta = draw_value_from_param_distribution('delta')
        theta = draw_value_from_param_distribution('theta')
        rho = draw_value_from_param_distribution('rho')


    return beta, gamma_I, gamma_U, delta, theta, rho


def compute_weighted_residuals(params_to_fit, t_grid, start_vals, y_true):
    fit_result = solve_ode_for_fitting_partly_fitted_y0(start_vals, t_grid, params_to_fit)

    # add value at t=0 to fit_result:
    y_fit = np.append(y_true[0], fit_result)

    # compute abs difference between predicted and actual infection counts:
    residual = y_true - y_fit

    # weight residuals:
    weighted_residuals = weigh_residuals(residual)

    #### Debugging ####
    # plot_train_and_fitted_infections_line_plot(y_true, y_fit)

    return weighted_residuals


def solve_ode_for_fitting_partly_fitted_y0(fixed_start_vals, fixed_params, fit_params):
    ## 1) Pull apart fitting params:
    beta = fit_params[0]
    E0 = fit_params[1]
    I0 = fit_params[2]

    ## 2) Setup other parameters:
    # 2a) Get time grid:
    t_grid = fixed_params['t_grid']

    # 2b) Get fixed model params:
    gamma_I = fixed_params['fixed_model_params']['gamma_I']
    gamma_U = fixed_params['fixed_model_params']['gamma_U']
    delta = fixed_params['fixed_model_params']['delta']
    theta = fixed_params['fixed_model_params']['theta']
    rho = fixed_params['fixed_model_params']['rho']

    #  2c) Get y0:
    #  Starting values for I, R and V are given. E0 is fitted and S0 is then computed as the last missing value.
    N = fixed_start_vals[0]
    R0 = fixed_start_vals[1]
    V0 = fixed_start_vals[2]

    # Compute U0 and S0:
    # Expected number of individuals in undetected compartment: Depends on "Dunkelziffer" factor
    U0 = I0 * rho / (1 - rho)
    S0 = N - E0 - I0 - U0 - R0 - V0

    # pack all together and add cumulated infections:
    y0 = (S0, E0, I0, U0, R0, V0, 0)

    # lambda function as shown here: https://www.kaggle.com/baiyanren/modified-seir-model-for-covid-19-prediction-in-us
    # this circumvents issues with the scipy odeint function, which can only handle a predefined number of params
    seir_ode = lambda y0, t: ode_seirv(y0, t, beta, gamma_I, gamma_U, delta, theta, rho)

    ode_result = odeint(func=seir_ode, y0=y0, t=t_grid).T

    # compute predicted daily infections from estimated cumulated number of infections:
    predicted_daily_infections = compute_daily_infections(ode_result[6, :])

    return predicted_daily_infections  # return only Infection counts


def solve_ode_for_fitting_fixed_y0(y0, t_grid, fit_params):
    beta = fit_params[0]
    gamma = params_SEIRV_fixed['gamma']['mean']
    delta = params_SEIRV_fixed['delta']['mean']
    theta = params_SEIRV_fixed['theta']['mean']

    # lambda function as shown here: https://www.kaggle.com/baiyanren/modified-seir-model-for-covid-19-prediction-in-us
    # this circumvents issues with the scipy odeint function, which can only handle a predefined number of params
    seir_ode = lambda y0, t: ode_seirv(y0, t, beta, gamma, delta, theta)

    ode_result = odeint(func=seir_ode, y0=y0, t=t_grid).T

    predicted_daily_infections = compute_daily_infections(ode_result[5, :])

    return predicted_daily_infections  # return only Infection counts


def solve_ode(y0, t, params):
    ode_result = odeint(func=ode_seirv, y0=y0, t=t, args=params).T

    # cumulated infections:
    cum_infections = ode_result[6, :]

    # compute daily new infections from cumulated infections:
    daily_infections = compute_daily_infections(cum_infections)

    # combine the numbers of individuals for each compartment over time + cumulated infections + daily infections:
    # to ensure that the sizes of both arrays fit the starting values for each compartment are dropped (t=0):
    cropped_ode_result = ode_result[:, 1:]

    result = np.vstack([cropped_ode_result, daily_infections])

    return result


def ode_seirv(y0, t, beta, gamma_I, gamma_U, delta, theta, rho):
    S, E, I, U, R, V, V_cum = y0
    N = S + E + I + U + R + V

    ## Differential equations:
    # Susceptible:
    dSdt = -beta / N * S * (I + U)

    # Exposed:
    dEdt = beta / N * S * (I + U) + \
           theta * beta / N * V * (I + U) - \
           delta * E

    # Infectious - Detected:
    dIdt = (1 - rho) * delta * E - gamma_I * I

    # Infectious - Undetected:
    dUdt = rho * delta * E - gamma_U * U

    # Recovered:
    dRdt = gamma_I * I + gamma_U * U

    # Vaccinated:
    dVdt = - theta * beta / N * V * (I + U)

    ## Cumulated Detected Infection Counts:
    dICumdt = (1 - rho) * delta * E

    return dSdt, dEdt, dIdt, dUdt, dRdt, dVdt, dICumdt


def compute_daily_infections(cumulated_infections: np.array) -> np.array:
    return np.diff(cumulated_infections)


def seirv_model_pipeline_DEPRECATED(y_train: np.array, start_vals_fitting: tuple, len_pred_period=14):
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
    fit_params_start_guess = (params_SEIRV_fit['beta']['mean'],)

    # Fit parameters:
    opt_params, success = leastsq(
        func=compute_weighted_residuals,
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
