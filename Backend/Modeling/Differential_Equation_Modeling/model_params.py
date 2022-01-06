params_SEIRV_fixed = {
    'gamma_I': 1 / 7,           # mean infectious period detected cases
    'gamma_U': 1 / 6,           # mean infectious period undetected cases
    'delta': 1 / 6.8,           # mean incubation time
    'theta': 0.1,
    'rho': 0.5                 # probability related to ratio of undetected cases
}

params_SEIRV_fit = {
    'beta': 0.4
}


LEAST_SQUARES_WEIGHT = {
    'lambda': 0.02                # exponential decay
}