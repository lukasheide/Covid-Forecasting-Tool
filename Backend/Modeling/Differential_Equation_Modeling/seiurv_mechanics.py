from scipy.integrate import odeint
import numpy as np


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