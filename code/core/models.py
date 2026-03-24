# Enrico Savorgnan
# University of Trieste, Italy, 2025
# A new model in “Behavioral Epidemiology of Infectious Diseases”: SIRS model with social distancing

"""
The aim of this file is to provide several behavioural SIRS models.
Since each of them has its own parameters, there is no single class for all of them.
At the end of the file, two different methods to compute the β(M) parameter are provided.
"""
import numpy as np
from scipy.integrate import solve_ivp

class SIRSModels():
    def __init__(self, model_type: str = 'sirs', 
                 beta: float | None = 0., 
                 gamma: float | None = 0., 
                 mu: float | None = 0., 
                 theta: float | None = 0., 
                 a1: float | None = 0., 
                 a2: float | None = 0., 
                 k: float | None = 1, 
                 delta: float | None = 0.,
                 omega: float | None = 2 * np.pi / 365):
        
        self.model_type = model_type
        # RENAME: Store the float value as beta_val to avoid naming conflicts
        self.beta_val = beta 
        self.gamma = gamma
        self.mu = mu
        self.theta = theta
        self.a1 = a1
        self.a2 = a2
        self.k = k
        self.delta = delta
        self.omega = omega
        
        # PRE-CALCULATE: Instantiate the correct beta function immediately.
        # This function will always be callable as func(x, t) or func(x, y, t).
        self.beta_func = self._get_beta_function()

    def _get_beta_function(self):
        """
        Returns a closure (function) that computes beta.
        We standardize the signatures so the ODE methods don't need 'if' statements.
        """
        # --- Case 1: Two-Layer Models (Need M1, M2, t) ---
        if self.model_type in ['sirs_two_layer', 'sirs_two_layer_incidence']:
            if self.delta != 0:
                # Seasonal: Uses time 't'
                def beta_seasonal(x, y, t):
                    return (self.beta_val / ((1 + self.k * x) * (1 + self.k * y))) * \
                           (1 + self.delta * np.cos(self.omega * t))
                return beta_seasonal
            else:
                # Non-seasonal: Accepts 't' but ignores it
                def beta_const(x, y, t):
                    return self.beta_val / ((1 + self.k * x) * (1 + self.k * y))
                return beta_const

        # --- Case 2: Zero/One-Layer Models (Need M, t) ---
        elif self.model_type in ['sirs_zero_layer', 'sirs_zero_layer_incidence', 'sirs_one_layer', 'sirs_one_layer_incidence']:
            if self.delta != 0:
                # Seasonal: Uses time 't'
                def beta_seasonal(x, t):
                    return (self.beta_val / (1 + self.k * x)) * \
                           (1 + self.delta * np.cos(self.omega * t))
                return beta_seasonal
            else:
                # Non-seasonal: Accepts 't' but ignores it
                def beta_const(x, t):
                    return self.beta_val / (1 + self.k * x)
                return beta_const
        
        # --- Case 3: Standard SIRS (No beta function needed) ---
        else:
             raise ValueError(f"Unknown model type: {self.model_type}")

    # --- ODE Models ---

    def sirs(self, t, X):
        """ Standard SIRS model. Uses self.beta_val (float) directly. """
        I, R = X
        dI = I * (self.beta_val - (self.mu + self.gamma)) - self.beta_val * R * I - self.beta_val * I**2
        dR = self.gamma * I - (self.mu + self.theta) * R
        return [dI, dR]

    def sirs_zero_layer(self, t, X):
        I, R, M = X
        # Call beta_func with (M, t) regardless of seasonality
        beta_current = self.beta_func(M, t)

        dI = I * (beta_current * (1. - I - R) - (self.mu + self.gamma))
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM = dI
        return [dI, dR, dM]

    def sirs_one_layer(self, t, X):
        I, R, M = X
        # Call beta_func with (M, t)
        beta_current = self.beta_func(M, t)

        dI = I * (beta_current * (1. - I - R) - (self.mu + self.gamma))
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM = self.a1 * (I - M)
        return [dI, dR, dM]

    def sirs_two_layer(self, t, X):
        I, R, M1, M2 = X
        # Call beta_func with (M1, M2, t)
        beta_current = self.beta_func(M1, M2, t)

        dI = I * (beta_current * (1. - I - R) - (self.mu + self.gamma))
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM1 = self.a1 * (I - M1)
        dM2 = self.a2 * (M1 - M2)
        return [dI, dR, dM1, dM2]

    def sirs_zero_layer_incidence(self, t, X):
        I, R = X
        S = 1. - R - I

        M = (-1 + max(0, np.sqrt(1 + 4 * self.a1 * self.k * self.beta_val * I * S))) / (2*self.a1)
        beta_current = self.beta_func(M, t)
        incidence = beta_current * S * I

        dI = incidence - I * (self.mu + self.gamma)
        dR = self.gamma * I - (self.mu + self.theta) * R
        return [dI, dR]

    def sirs_one_layer_incidence(self, t, X):
        I, R, M = X
        beta_current = self.beta_func(M, t)

        S = 1. - R - I
        incidence = beta_current * S * I

        dI = incidence - I * (self.mu + self.gamma)
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM = self.a1 * (incidence - M)
        return [dI, dR, dM]

    def sirs_two_layer_incidence(self, t, X):
        I, R, M1, M2 = X
        beta_current = self.beta_func(M1, M2, t)

        S = 1. - R - I
        incidence = beta_current * S * I

        dI = incidence - I * (self.mu + self.gamma)
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM1 = self.a1 * (incidence - M1)
        dM2 = self.a2 * (M1 - M2)
        return [dI, dR, dM1, dM2]

class SIRS():
    def __init__(self, model_type: str = 'sirs', 
                 params: dict | None = None, 
                 beta: float | None = 0., 
                 gamma: float | None = 0., 
                 mu: float | None = 0.0, 
                 theta: float | None = 0., 
                 a1: float | None = 0., 
                 a2: float | None = 0, 
                 k: float | None = 1,   
                 delta: float | None = 0, 
                 omega: float | None = 2 * np.pi / 365):
        
        self.model_type = model_type

        self.beta = beta if params is None else params.get('beta', beta)
        self.gamma = gamma if params is None else params.get('gamma', gamma)
        self.mu = mu if params is None else params.get('mu', mu)
        self.theta = theta if params is None else params.get('theta', theta)
        self.a1 = a1 if params is None else params.get('a1', a1)
        self.a2 = a2 if params is None else params.get('a2', a2)
        self.k = k if params is None else params.get('k', k)
        self.delta = delta if params is None else params.get('delta', delta)
        self.omega = omega if params is None else params.get('omega', omega)
        
        # 1. Initialize the physics engine
        self._physics = SIRSModels(
            model_type=model_type, beta=self.beta, gamma=self.gamma, mu=self.mu, theta=self.theta,
            a1=self.a1, a2=self.a2, k=self.k, delta=self.delta, omega=self.omega
        )

        # 2. Bind the correct model method
        if hasattr(self._physics, model_type):
            self.model = getattr(self._physics, model_type)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        

    def _solve_odes(self, t, t_span, i, r, m1=None, m2=None):
        """
            Internal method to solve the ODEs based on the model type and parameters.

            Parameters:
            - t: array
                Time points at which to evaluate the solution.
            - t_span: list
                List of two elements [t_start, t_end] defining the time span for the simulation.
            - i: float
                Initial condition for the infected population.
            - r: float
                Initial condition for the recovered population.
            - m1: float, optional
                Initial condition for the first memory layer (if applicable).
            - m2: float, optional
                Initial condition for the second memory layer (if applicable).
        """
        # solution = solve_ivp(self.model, t_span, [i, r] + ([m1] if m1 is not None else []) + ([m2] if m2 is not None else []),
        #                     args=(self.beta, self.gamma, self.mu, self.theta, self.a1, self.a2, self.k, self.delta, self.omega), dense_output=True)

        solution = solve_ivp(self.model, t_span, [i, r] + ([m1] if m1 is not None else []) + ([m2] if m2 is not None else []), dense_output=True)
        return solution
    

    def cumulative_incidence(self, solution, t_span : list):
        """
            Internal method to calculate the cumulative incidence from the solution of the ODEs.

            Parameters:
            - solution: OdeSolution  
                The solution object returned by solve_ivp.
            - t_span: list
                List of two elements [t_start, t_end] defining the time span for the simulation.
            
            Returns:
            - cumulative_incidence: array
                Array of cumulative incidence values corresponding to the time points in the solution.
        """
        t = np.linspace(t_span[0], t_span[1], 20000)
        I = solution[0]
        R = solution[1]
        S = 1 - I - R
        M1 = solution[2] if len(solution) > 2 else None
        M2 = solution[3] if len(solution) > 3 else None
        beta_1 = self._physics.beta_one_layer(self.beta, self.k)
        beta_2 = self._physics.beta_two_layer(self.beta, self.k, self.k)

        if M1 is not None and M2 is not None:
            incidence = [beta_2(m1, m2) * (1 - r - i) * i for i, r, m1, m2 in zip(I, R, M1, M2)]
        elif M1 is not None:
            incidence = [beta_1(m1) * (1 - r - i) * i for i, r, m1 in zip(I, R, M1)]
        else:
            incidence = [self.beta * (1 - r - i) * i for i, r in zip(I, R)]

        # Integrating the incidence over time to get cumulative incidence, using the trapezoidal rule
        cumulative_incidence = np.trapezoid(incidence, t)

        return cumulative_incidence
    


    def simulate(self, t_span: list, initial_conditions: list):
        """
            Simulate the SIRS model over the given time span and initial conditions.
                
            Parameters:
            - t_span: list
                List of two elements [t_start, t_end] defining the time span for the simulation.
            - initial_conditions: list
                List of initial conditions for the model variables.
        """
        print(initial_conditions, len(initial_conditions), self.model_type)
        if self.model_type in ['sirs', 'sirs_zero_layer_incidence']:
            assert len(initial_conditions) == 2, "Expected 2 initial conditions for this model."
        elif self.model_type in ['sirs_zero_layer', 'sirs_one_layer', 'sirs_one_layer_incidence']:
            assert len(initial_conditions) == 3, "Expected 3 initial conditions for this model."
        elif self.model_type in ['sirs_two_layer', 'sirs_two_layer_incidence']:
            assert len(initial_conditions) == 4, "Expected 4 initial conditions for this model."

        t = np.linspace(t_span[0], t_span[1], 20000)

        i, r = initial_conditions[0], initial_conditions[1]
        m1 = initial_conditions[2] if len(initial_conditions) == 3 else None
        m2 = initial_conditions[3] if len(initial_conditions) == 4 else None

        print(m1, m2)
        solution = self._solve_odes(t, t_span, i, r, m1, m2)
        return solution.sol(t)






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