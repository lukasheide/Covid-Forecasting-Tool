import numpy as np

#######################################################################################################################
# Below the values of each parameter of the differential equation model can be adjusted.
# Here only the mean-value and thus the point estimate of each model parameter is used in the end.
# Other values related to upper and lower limits, variances, as well as distributions could be used to introduce
# stochasticity. We decided against that here, as we instead opted for a so-called bootstrapping approach to provide
# predictions intervals instead of using a simulation in which for each model run all parameters are randomly drawn
# and then the same model is rerun n times and the average values are used as point estimates.
#######################################################################################################################

params_SEIURV_fixed = {
    'gamma_I': { # mean infectious period detected cases
       'mean': 1 / 3.6,
        'sd': 1 / 1.85,
        'upper_lim': 1 / 2.1,
        'lower_lim': 1 / 10,
        'distribution': "lognormal"
    },

    'gamma_U': {  # mean infectious period undetected cases
        'mean': 1 / 2.1,
        'sd': 1 / 1.1,
        'upper_lim': 1 / 1.4,
        'lower_lim': 1 / 5.88,
        'distribution': "normal"
    },

    'delta': {  # mean incubation time
        'mean': 1 / 2,
        'sd': 1 / 2 / 10, # placeholder
        'upper_lim': 1 / 5,
        'lower_lim': 1 / 1.5,
        'distribution': "normal"
    },

    'theta': {  # vaccination protection #### OVERWRITTEN BY COMPUTED VACCINATION EFFICIENCY (-> no longer used)
        'mean': 0.1,
        'sd': 0.1 / 10, # placeholder
        'upper_lim': 1,
        'lower_lim': 0,
        'distribution': "normal"
    },

    'rho': {  # probability related to ratio of undetected cases (Undetected Ratio)
        'mean': (1.35 - 1) / 1.35,                  # Transformation from Ratio to probability
        'sd': (1.35 - 1) / 1.35 / 10, # placeholder
        'upper_lim': 1,
        'lower_lim': 0,
        'distribution': "normal"
    }
}

params_SEIRV_fit_DEPRECATED = {
    'beta': {
        'mean': 0.4,
        'sd': 0.1,
        'relative_sd': 0.2,      # relative = sd related to mean value
        'upper_lim': 2,
        'lower_lim': 0,
        'distribution': "normal"
    }
}


# Parameter is used for fitting the differential equation model to historical data. The model fitting is done by
# adjusting the fitted parameters such that the difference between the resulting predicted infections and historical
# infections is minimized. This is done by computing the Mean-Squared-Errors. We additionally weighted these errors
# and assigned more weight to residuals towards the end of the fitting period. This ensures that large deviations
# are penalized more strongly shortly before the forecasting period starts compared to the point that is furthest
# away from the forecasting period.
LEAST_SQUARES_WEIGHT = {
    'lambda': 0.05,          # exponential decay
    'lambda_inverse': -0.02
}


def draw_value_from_param_distribution(paramname):
    """
    This function did not end up being used in our final modeling infrastructure. More details why and can be found
    above. The aim of this function was to draw random parameter values for introducing stochasticity.
    """

    # get information for parameter:
    param = params_SEIURV_fixed[paramname]

    ## Different distributions:
    # Normal:
    if param['distribution'] == 'normal':
        random_val = np.random.normal(
            loc=param['mean'],
            scale=param['sd']
        )

    # Lognormal
    elif param['distribution'] == 'lognormal':
        random_val = np.random.lognormal(
            loc=param['mean'],
            scale=param['sd']
        )

    # Uniform
    elif param['distribution'] == 'uniform':
        random_val = np.random.uniform(
            loc=param['mean'],
            scale=param['sd']
        )

    # If Distribution does not exist throw an error:
    else:
        raise Exception('Distribution {} does not exist.'.format(param['distribution']))

    # Ensure boundaries:
    if random_val > param['upper_lim']:
        random_val = param['upper_lim']
    elif random_val < param['lower_lim']:
        random_val = param['lower_lim']

    return random_val


def draw_random_beta(beta_estimate, sd=-1, sd_passed=False):
    mean = beta_estimate
    # Compute standard deviation relative to how large the beta estimate is:
    if not sd_passed:
        sd = mean * params_SEIRV_fit_DEPRECATED['beta']['relative_sd']

    # Normal:
    if params_SEIRV_fit_DEPRECATED['beta']['distribution'] == 'normal':
        random_val = np.random.normal(
            loc=mean,
            scale=sd
        )

    # Lognormal
    elif params_SEIRV_fit_DEPRECATED['beta']['distribution'] == 'uniform':
        random_val = np.random.lognormal(
            loc=mean,
            scale=sd
        )

    # Uniform
    elif params_SEIRV_fit_DEPRECATED['beta']['distribution'] == 'uniform':
        random_val = np.random.uniform(
            loc=mean,
            scale=sd
        )

    # If Distribution does not exist throw an error:
    else:
        raise Exception('Distribution {} does not exist.'.format(params_SEIRV_fit_DEPRECATED['beta']['distribution']))

    # Ensure boundaries:
    if random_val > params_SEIRV_fit_DEPRECATED['beta']['upper_lim']:
        random_val = params_SEIRV_fit_DEPRECATED['beta']['upper_lim']
    elif random_val < params_SEIRV_fit_DEPRECATED['beta']['lower_lim']:
        random_val = params_SEIRV_fit_DEPRECATED['beta']['lower_lim']

    return random_val