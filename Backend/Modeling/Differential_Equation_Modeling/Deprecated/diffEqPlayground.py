import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from Backend.Modeling.Differential_Equation_Modeling.model_params import params_SEIRV_fixed
import matplotlib.pyplot as plt


def model_fit(y_true):

    integrate_model(beta=0.1)

    curve_fit(integrate_model, xdata=123, ydata=y_true, p0=[0.5])

    print('pass')


def integrate_model(beta, gamma):
    S0 = 120000
    E0 = 250
    I0 = 1000

    y0 = S0, E0, I0

    num_days = 28

    # Create a grid of time points (in days)
    t = np.linspace(0, num_days, num_days + 1)

    # Integrate the differential equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(beta, gamma))
    S, I_pred, R = ret.T

    return I_pred


def deriv(y, t, beta, gamma):
    S, I, R = y

    # Susceptibles:
    dSdt = -beta * S * I

    # Infectious:
    dIdt = beta * S * I - gamma * I

    # Recovered:
    dRdt = gamma * I

    return dSdt, dIdt, dRdt



def get_params(t):
    params = {
        'gamma': 1/14
    }

    return params