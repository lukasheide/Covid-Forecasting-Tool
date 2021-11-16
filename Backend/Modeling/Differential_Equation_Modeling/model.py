import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def run_SEIRV_model():

    num_days = 20

    y0 = get_starting_values()

    # Create a grid of time points (in days)
    t = np.linspace(0, num_days, num_days+1)

    # Integrate the differential equations over the time grid, t.
    ret = odeint(deriv_SEIRV_model, y0, t)
    S, E, I, R, V, I_cum = ret.T

    return S, E, I, R, V, I_cum



def deriv_SEIRV_model(y, t):
    # Unpack values contained in y
    S, E, I, R, V, I_cum = y

    N = S+E+I+R+V

    # Get model parameters depending on t
    params = get_params(t)


    ## Differential equations:
    # Susceptibles:
    dSdt = -params['beta']/N * S * I

    # Exposed:
    dEdt = params['beta']/N * S * I + params['theta'] * params['beta']/N * V * I - params['delta'] * E

    # Infectious:
    dIdt = params['delta'] * E - params['gamma'] * I

    # Recovered:
    dRdt = params['gamma'] * I

    # Vaccinated:
    dVdt = - params['theta'] * params['beta']/N * V * I

    ## Cumulated Infections:
    dI_cumdt = params['delta'] * E

    return dSdt, dEdt, dIdt, dRdt, dI_cumdt, dVdt


def get_params(t):
    """
    computes params at time step t
    :return: dictionary containing all params
    """
    params = {
        'theta': 0.1,
        'delta': 1/3,
        'gamma': 1/11,
        'beta': 0.5
    }

    return params


def get_starting_values():
    S0 = 120000
    E0 = 250
    I0 = 1000
    R0 = 300000 - S0 - E0 - I0
    I_cum = I0 + R0
    V0 = 150000

    return S0, E0, I0, R0, I_cum, V0


