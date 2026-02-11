# Enrico Savorgnan
# University of Trieste, Italy, 2025
# A new model in “Behavioral Epidemiology of Infectious Diseases”: SIRS model with social distancing

"""
The aim of this file is to provide several behavioural SIRS models.
Since each of them has its own parameters, there is no single class for all of them.
At the end of the file, two different methods to compute the β(M) parameter are provided.
"""

import numpy as np

def sirs(t, X, beta_param, gamma, mu, theta):
    """
    Standard SIRS model without memory and human behaviour.
    """
    I, R = X

    dI = I*(beta_param - (mu + gamma)) - beta_param * R * I - beta_param * I**2
    dR = gamma * I - (mu + theta) * R

    return [dI, dR]


def sirs_zero_layer(t, X, beta_1, gamma, mu, theta, a=0, k=1, delta=0):
    """
    SIRS model with only the current information taken into account.
    Parameters 'a' and 'k' are not used, but kept for compatibility with other models.
    """
    I, R, M = X

    if delta != 0:
        # If seasonal forcing is present, pass time 't' to beta_1
        beta_1_func = beta_1(M, t)
    else:
        beta_1_func = beta_1(M)

    dI = I * (beta_1_func * (1.-I-R) - (mu + gamma))
    dR = gamma * I - (mu + theta) * R
    dM = dI

    return [dI, dR, dM]


def sirs_one_layer(t, X, beta_1, gamma, mu, theta, a, k=1, delta=0.0001):
    """
    SIRS model with one layer of memory.
    Parameter 'k' is nopt used but kept for compatibility with other models.
    """
    I, R, M = X

    if delta != 0:
        # If seasonal forcing is present, pass time 't' to beta_1
        beta_1_func = beta_1(M, t)
    else:
        print(delta)
        beta_1_func = beta_1(M)

    dI = I * ( beta_1_func * (1.-I-R) - (mu + gamma) )
    dR = gamma * I - (mu + theta) * R
    dM = a * (I - M)

    return [dI, dR, dM]


def sirs_two_layer(t, X, beta_2, gamma, mu, theta, a1, a2, k=1, delta=0):
    """
    SIRS model with two layers of memory, one more recent and one older.
    Parameter 'k' is nopt used but kept for compatibility with other models.
    """
    I, R, M1, M2 = X

    if delta != 0:
        # If seasonal forcing is present, pass time 't' to beta_2
        beta_2_func = beta_2(M1, M2, t)
    else:
        beta_2_func = beta_2(M1, M2)

    dI = I * ( beta_2_func * (1.-I-R) - (mu + gamma) )
    dR = gamma * I - (mu + theta) * R
    dM1 = a1 * (I - M1)
    dM2 = a2 * (M1 - M2)

    return [dI, dR, dM1, dM2]


def sirs_one_layer_incidence(t, X, beta_1, gamma, mu, theta, a1, a2=0, k=1, delta=0.001):
    """
    SIRS model with one, incidence-based, layer of memory.
    """
    I, R, M = X

    if delta != 0:
        # If seasonal forcing is present, pass time 't' to beta_1
        beta_1_func = beta_1(M, t)
    else:
        beta_1_func = beta_1(M)

    dI = I*( beta_1_func*(1.-R-I) - (mu + gamma) )
    dR = gamma * I - (mu + theta) * R
    dM = a1 * (beta_1_func*(1.-R-I)*I - M)

    return [dI, dR, dM]


def sirs_two_layer_incidence(t, X, beta_2, gamma, mu, theta, a1, a2, k, delta=0):
    """
    SIRS model with two layers of memory, both incidence-based.
    """
    I, R, M1, M2 = X

    if delta != 0:
        # If seasonal forcing is present, pass time 't' to beta_2
        beta_2_func = beta_2(M1, M2, t)
    else:
        beta_2_func = beta_2(M1, M2)

    dI = I*( beta_2_func*(1.-R-I) - (mu + gamma) )
    dR = gamma * I - (mu + theta) * R
    dM1 = a1 * (beta_2_func*(1.-R-I)*I - M1)
    dM2 = a2 * (M1 - M2)

    return [dI, dR, dM1, dM2]


# Memory Functions
def beta_one_layer(beta_0, k):
    """ Returns a function that computes the beta parameter for a one-layer memory model."""
    def beta_inside(x):
        return beta_0 / (1 + k * x)
    return beta_inside

def beta_two_layer(beta_0, k1, k2):
    """ Returns a function that computes the beta parameter for a two-layers memory model."""
    def beta_inside(x, y):
        return beta_0 / (1 + k1 * x) / (1 + k2 * y)
    return beta_inside

def beta_one_layer_seasonal(beta_0, k, delta, omega):
    """ Returns a function that computes the beta parameter for a one-layer memory model, adding a sinusoidal modulation."""
    def beta_inside(x, t):
        seasonal_factor = 1 + delta * np.cos(omega * t)
        return beta_0 / (1 + k * x) * seasonal_factor
    return beta_inside

def beta_two_layer_seasonal(beta_0, k1, k2, delta, omega):
    """ Returns a function that computes the beta parameter for a two-layers memory model, adding a sinusoidal modulation."""
    def beta_inside(x, y, t):
        seasonal_factor = 1 + delta * np.cos(omega * t)
        return beta_0 / (1 + k1 * x) / (1 + k2 * y) * seasonal_factor
    return beta_inside

