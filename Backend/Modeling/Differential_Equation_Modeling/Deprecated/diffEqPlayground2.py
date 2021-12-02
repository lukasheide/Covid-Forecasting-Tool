import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def outer_function(y_data):

    num_days = 28
    t_grid = np.linspace(0, num_days, num_days + 1)

    N = 30000
    I0 = 1000
    R0 = 0
    S0 = N - I0 - R0


    def fit_odeint(x, beta, gamma):
        return odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))[:, 1]

    def sir_model(y, x, beta, gamma):
        S, I, R = y

        # Susceptibles:
        dSdt = -beta * S * I

        # Infectious:
        dIdt = beta * S * I - gamma * I

        # Recovered:
        dRdt = gamma * I

        return dSdt, dIdt, dRdt


    popt, pcov = curve_fit(fit_odeint, t_grid, y_data)
    fitted = fit_odeint(t_grid, *popt)

    pass