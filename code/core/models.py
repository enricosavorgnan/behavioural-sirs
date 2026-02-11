# Enrico Savorgnan
# University of Trieste, Italy, 2025
# A new model in “Behavioral Epidemiology of Infectious Diseases”: SIRS model with social distancing

"""
The aim of this file is to provide several behavioural SIRS models.
Since each of them has its own parameters, there is no single class for all of them.
At the end of the file, two different methods to compute the β(M) parameter are provided.
"""
import numpy as np

class SIRS:

    def __init__(self, model_type, beta, gamma, mu, theta, a1=0, a2=0, k=1, delta=0):
        self.model_type = model_type
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.theta = theta
        self.a1 = a1
        self.a2 = a2
        self.k = k
        self.delta = delta

        self.model = self._build_model()

    def _build_model(self):
        if self.model_type == 'sirs':
            return sirs
        elif self.model_type == 'sirs_zero_layer':
            return sirs_zero_layer
        elif self.model_type == 'sirs_one_layer':
            return sirs_one_layer
        elif self.model_type == 'sirs_two_layer':
            return sirs_two_layer
        elif self.model_type == 'sirs_one_layer_incidence':
            return sirs_one_layer_incidence
        elif self.model_type == 'sirs_two_layer_incidence':
            return sirs_two_layer_incidence
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
    def sirs(self, t, X):
        """
        Standard SIRS model without memory and human behaviour.
        
        :param t: Description
        :param X: Description
        :param beta: Description
        :param gamma: Description
        :param mu: Description
        :param theta: Description
        """
        I, R = X

        dI = I*(self.beta - (self.mu + self.gamma)) - self.beta * R * I - self.beta * I**2
        dR = self.gamma * I - (self.mu + self.theta) * R

        return [dI, dR]
    
    def sirs_zero_layer(self, t, X):
        """
        SIRS model with only the current information taken into account.
        """
        I, R, M = X

        if self.delta != 0:
            # If seasonal forcing is present, pass time 't' to beta_1
            beta_1_func = self.beta(M, t)
        else:
            beta_1_func = self.beta(M)

        dI = I * (beta_1_func * (1.-I-R) - (self.mu + self.gamma))
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM = dI

        return [dI, dR, dM]
    
    def sirs_one_layer(self, t, X):
        """
        SIRS model with one layer of memory.
        """
        I, R, M = X

        if self.delta != 0:
            # If seasonal forcing is present, pass time 't' to beta_1
            beta_1_func = self.beta(M, t)
        else:
            print(self.delta)
            beta_1_func = self.beta(M)

        dI = I * ( beta_1_func * (1.-I-R) - (self.mu + self.gamma) )
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM = self.a1 * (I - M)

        return [dI, dR, dM]
    
    def sirs_two_layer(self, t, X):
        """
        SIRS model with two layers of memory, one more recent and one older.
        """
        I, R, M1, M2 = X

        if self.delta != 0:
            # If seasonal forcing is present, pass time 't' to beta_2
            beta_2_func = self.beta(M1, M2, t)
        else:
            beta_2_func = self.beta(M1, M2)

        dI = I * ( beta_2_func * (1.-I-R) - (self.mu + self.gamma) )
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM1 = self.a1 * (I - M1)
        dM2 = self.a2 * (M1 - M2)

        return [dI, dR, dM1, dM2]
    
    def sirs_one_layer_incidence(self, t, X):
        """
        SIRS model with one, incidence-based, layer of memory.
        """
        I, R, M = X

        if self.delta != 0:
            # If seasonal forcing is present, pass time 't' to beta_1
            beta_1_func = self.beta(M, t)
        else:
            beta_1_func = self.beta(M)

        dI = I*( beta_1_func*(1.-R-I) - (self.mu + self.gamma) )
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM = self.a1 * (beta_1_func*(1.-R-I)*I - M)

        return [dI, dR, dM]
    
    def sirs_two_layer_incidence(self, t, X):
        """
        SIRS model with two layers of memory, both incidence-based.
        """
        I, R, M1, M2 = X

        if self.delta != 0:
            # If seasonal forcing is present, pass time 't' to beta_2
            beta_2_func = self.beta(M1, M2, t)
        else:
            beta_2_func = self.beta(M1, M2)

        dI = I*( beta_2_func*(1.-R-I) - (self.mu + self.gamma) )
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM1 = self.a1 * (beta_2_func*(1.-R-I)*I - M1)
        dM2 = self.a2 * (M1 - M2)

        return [dI, dR, dM1, dM2]
    
    def beta_one_layer(self):
        """ Returns a function that computes the beta parameter for a one-layer memory model."""
        def beta_inside(x):
            return self.beta / (1 + self.k * x)
        return beta_inside
    
    def beta_two_layer(self):
        """ Returns a function that computes the beta parameter for a two-layers memory model."""
        def beta_inside(x, y):
            return self.beta / (1 + self.k * x) / (1 + self.k * y)
        return beta_inside
    
    def beta_one_layer_seasonal(self):
        """ Returns a function that computes the beta parameter for a one-layer memory model, adding a sinusoidal modulation."""
        def beta_inside(x, t):
            seasonal_factor = 1 + self.delta * np.cos(2 * np.pi * t / 365)
            return self.beta / (1 + self.k * x) * seasonal_factor
        return beta_inside
    
    def beta_two_layer_seasonal(self):
        """ Returns a function that computes the beta parameter for a two-layers memory model, adding a sinusoidal modulation."""
        def beta_inside(x, y, t):
            seasonal_factor = 1 + self.delta * np.cos(2 * np.pi * t / 365)
            return self.beta / (1 + self.k * x) / (1 + self.k * y) * seasonal_factor
        return beta_inside
    

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