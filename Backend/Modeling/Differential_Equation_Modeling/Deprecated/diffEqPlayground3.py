import lmfit
from lmfit import minimize, Parameters
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def runner(y_data):

    y_true = y_data
    num_days = 28
    t_grid = np.linspace(0, num_days, num_days + 1)

    y0 = 280_000, 1_000, 19_000

    beta = 0.5
    gamma = 1/14

    def sir_model(y, t, beta, gamma):
        S, I, R = y
        N = S + I + R

        # Susceptibles:
        dSdt = -beta / N * S * I

        # Infectious:
        dIdt = beta / N * S * I - gamma * I

        # Recovered:
        dRdt = gamma * I

        return dSdt, dIdt, dRdt

    def fit_ode(t_grid, beta, gamma):
        return odeint(sir_model, y0=y0, t=t_grid, args=(beta, gamma))[:,1] # return only Infection counts

    # standard application
    res = fit_ode(t_grid, beta, gamma)

    # fitting

    # Attempt with scipy.optimize:
    popt, pcov = curve_fit(fit_ode, t_grid, y_data)

    # Use fitted values to retrieve estimates: (asterix unpacks elements of popt)
    fitted = fit_ode(t_grid, *popt)

    # Attempt with lmfit:
    model_wrapper = lmfit.Model(fit_ode)
    params = Parameters()
    params.add('beta', value=0.5, vary=True)
    params.add('gamma', value=1/14, vary=False)
    params.add('t_grid', vary=False)

    model_wrapper.fit(data=y_true, params=params)

