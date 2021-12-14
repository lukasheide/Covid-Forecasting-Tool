import lmfit
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from Backend.Modeling.Differential_Equation_Modeling.model_params import params_SEIRV_fixed
from lmfit import minimize, Parameters
import matplotlib.pyplot as plt


def outer_function(y_true, start_vals):
    pass


class SEIRV_model:
    def __init__(self, params=None):
        self.params = params

    def get_fit_params(self):
        params = lmfit.Parameters()
        params.add('beta', value=0.5, vary=True)
        params.add('gamma', value=1 / 11, vary=False)
        params.add('delta', value=1 / 3, vary=False)
        params.add('theta', value=0.1, vary=False)

        params.add('N', value= 300_000, vary=False)
        params.add('t', value=14, vary=False)
        return params

    def get_initial_conditions(self, data):
        population = self.params['N']
        t = self.params['t']

        t = np.arange(t)

        (S, E, I, R, V) = self.predict(t, )











def outer_function_blab(y_true, start_vals):

    params = Parameters
    # model params
    params.add('beta', value=0.5, vary=True)
    params.add('gamma', value=1/11, vary=False)
    params.add('delta', value=1/3, vary=False)
    params.add('theta', value=0.1, vary=False)

    # additional params:
    params.add('theta', value=0.1, vary=False)
    params.add('theta', value=0.1, vary=False)

    num_days = 14
    # Get overall number of individuals in model:
    N = sum(start_vals[0:4])




def integrate(num_days, start_vals, N, beta, gamma, delta, theta):

    y0 = start_vals

    # Create a grid of time points (in days)
    t = np.linspace(0, num_days, num_days + 1)

    ret = odeint(deriv, y0, t, args=(
        N,
        beta,
        gamma,
        delta,
        theta
    ))


def deriv(y, t, N, beta, gamma, delta, theta):
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