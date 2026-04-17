# Enrico Savorgnan
# University of Trieste, Italy, 2025
# A new model in “Behavioral Epidemiology of Infectious Diseases”: SIRS model with social distancing

"""
Several versions of Behavioural SIRS
"""
import numpy as np
from scipy.integrate import solve_ivp


class SIRS:
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
        beta_1 = self._physics.beta_func(self.beta, self.k)
        beta_2 = self._physics.beta_func(self.beta, self.k, self.k)

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