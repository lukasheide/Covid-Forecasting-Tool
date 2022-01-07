import numpy as np

params_SEIRV_fixed = {
    'gamma_I': { # mean infectious period detected cases
       'mean': 1 / 7,
        'sd': 1 / 7 / 10,
        'upper_lim': 2,
        'lower_lim': 0,
        'distribution': "normal"
    },

    'gamma_U': {  # mean infectious period undetected cases
        'mean': 1 / 6,
        'sd': 1 / 6 / 10,
        'upper_lim': 2,
        'lower_lim': 0,
        'distribution': "normal"
    },

    'delta': {  # mean incubation time
        'mean': 1 / 6.8,
        'sd': 1 / 6.8 / 10,
        'upper_lim': 2,
        'lower_lim': 0,
        'distribution': "normal"
    },

    'theta': {  # vaccination protection
        'mean': 0.1,
        'sd': 0.1 / 10,
        'upper_lim': 1,
        'lower_lim': 0,
        'distribution': "normal"
    },

    'rho': {  # probability related to ratio of undetected cases (Dunkelziffer)
        'mean': 0.5,
        'sd': 0.5 / 10,
        'upper_lim': 1,
        'lower_lim': 0,
        'distribution': "normal"
    }
}

params_SEIRV_fit = {
    'beta': {
        'mean': 0.4,
        'sd': 0.1,
        'relative_sd': 0.2,      # relative = sd related to mean value
        'upper_lim': 2,
        'lower_lim': 0,
        'distribution': "normal"
    }
}


LEAST_SQUARES_WEIGHT = {
    'lambda': 0.03                # exponential decay
}


def draw_value_from_param_distribution(paramname):
    # get information for parameter:
    param = params_SEIRV_fixed[paramname]

    ## Different distributions:
    # Normal:
    if param['distribution'] == 'normal':
        random_val = np.random.normal(
            loc=param['mean'],
            scale=param['sd']
        )

    # Lognormal
    elif param['distribution'] == 'uniform':
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
        sd = mean * params_SEIRV_fit['beta']['relative_sd']

    # Normal:
    if params_SEIRV_fit['beta']['distribution'] == 'normal':
        random_val = np.random.normal(
            loc=mean,
            scale=sd
        )

    # Lognormal
    elif params_SEIRV_fit['beta']['distribution'] == 'uniform':
        random_val = np.random.lognormal(
            loc=mean,
            scale=sd
        )

    # Uniform
    elif params_SEIRV_fit['beta']['distribution'] == 'uniform':
        random_val = np.random.uniform(
            loc=mean,
            scale=sd
        )

    # If Distribution does not exist throw an error:
    else:
        raise Exception('Distribution {} does not exist.'.format(params_SEIRV_fit['beta']['distribution']))

    # Ensure boundaries:
    if random_val > params_SEIRV_fit['beta']['upper_lim']:
        random_val = params_SEIRV_fit['beta']['upper_lim']
    elif random_val < params_SEIRV_fit['beta']['lower_lim']:
        random_val = params_SEIRV_fit['beta']['lower_lim']

    return random_val