import numpy as np
import matplotlib.pyplot as plt

from Backend.Modeling.Differential_Equation_Modeling.model_params import params_SEIRV_fixed


def produce_simulated_infection_counts():
    # fake infection cases for 28 days:
    inf_cases = [1000]
    inc_factor = 1.03
    noise_sd = 0.01
    num_days = 28

    decrease_factor = 0.001

    # set seed for drawing random numbers:
    np.random.seed(1)

    # create list with infection cases for 28 days
    for d in range(num_days-1):
        noise = np.random.normal(0, noise_sd)

        # use previous day as base for next day and multiply it with a factor to add some noise and subtract factor by
        # which curve is flattening.
        new_val = inf_cases[-1] * (inc_factor+noise-decrease_factor*d)

        inf_cases.append(new_val)

    # transform list to numpy array
    inf_cases = np.array(inf_cases)

    # for debugging purposes: plotting
    # plt.plot(inf_cases)
    # plt.show()


    return inf_cases


def set_starting_values(y_train):
    gamma = params_SEIRV_fixed['gamma']
    delta = params_SEIRV_fixed['delta']

    I0 = y_train[0] / gamma
    V0 = 150_000
    R0 = 17_500

    # estimate E0:
    E0 = I0 * gamma / delta

    S0 = 300_000 - I0 - V0 - R0

    return S0, E0, I0, R0, V0


def set_starting_values_e0_fitted(y_train):
    gamma = params_SEIRV_fixed['gamma']
    delta = params_SEIRV_fixed['delta']

    I0 = y_train[0] / gamma
    V0 = 150_000
    R0 = 17_500

    N = 300_000

    return N, I0, R0, V0
