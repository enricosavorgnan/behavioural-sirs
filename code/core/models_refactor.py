# Enrico Savorgnan
# University of Trieste, Italy, 2025
# A new model in “Behavioral Epidemiology of Infectious Diseases”: SIRS model with social distancing

"""
Class method building utilities for simulating Behavioral SIRS models
"""

import numpy as np
import yaml
from scipy.integrate import solve_ivp

from code.core.sirs_models import SIRSModels


class SIRS:
    """
    SIRS Models
    """
    def __init__(self,
                 model_params: dict | None = None,
                 config_path : str | None = None):

        assert model_params is not None or config_path is not None, \
            f"One between model_params dictionary and config_path YAML file must be not None"

        if model_params is not None:
            self._set_params_with_dict(model_params = model_params)
        elif config_path is not None:
            self._set_params_with_yaml(config_path = config_path)

        self._physics = SIRSModels(
            model_type=self.model_type,
            r0=self.r0,
            gamma=self.gamma,
            mu=self.mu,
            theta=self.theta,
            a1=self.a1,
            a2=self.a2,
            k1=self.k1,
            k2 = self.k2,
            alpha1 = self.alpha1,
            alpha2 = self.alpha2,
            delta=self.delta,
            omega=self.omega )

        self.model = getattr(self._physics, self.model_type)


    def _set_params_with_dict(self, model_params: dict):
        self.model_type : str = model_params.get('model_type', 'sirs')
        self.r0 : float  = model_params.get('r0', 0.)
        self.mu : float = model_params.get('mu', 0.)
        self.gamma : float = model_params.get('gamma', 0.)
        self.theta : float = model_params.get('theta', 0.)
        self.a1 : float = model_params.get('a1', 0.)
        self.a2 : float = model_params.get('a2', 0.)
        self.k1 : float = model_params.get('k1', 1.)
        self.k2 : float = model_params.get('k2', 1.)
        self.alpha1 : float = model_params.get('alpha1', 1.)
        self.alpha2 : float = model_params.get('alpha2', 1.)
        self.delta : float = model_params.get('delta', 0)
        self.omega : float = model_params.get('omega', 2 * np.pi / 365)


    def _set_params_with_yaml(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.model_type : str = config.get('type', 'sirs')
        self.r0 : float  = config.get('r0', 0.)
        self.mu : float = config.get('mu', 0.)
        self.gamma : float = config.get('gamma', 0.)
        self.theta : float = config.get('theta', 0.)
        self.a1 : float = config.get('a1', 0.)
        self.a2 : float = config.get('a2', 0.)
        self.k1 : float = config.get('k1', 1.)
        self.k2 : float = config.get('k2', 1.)
        self.alpha1 : float = config.get('alpha1', 1.)
        self.alpha2 : float = config.get('alpha2', 1.)
        self.delta : float = config.get('delta', 0)
        self.omega : float = config.get('omega', 2 * np.pi / 365)


    def _solve_odes(self,
                    t : list | np.ndarray,
                    t_span : list[float | int],
                    i : float,
                    r : float,
                    m1 : float | None = None,
                    m2 : float | None = None,
                    method : str = 'RK45'):
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
        variables = [i, r] + ( [m1] if m1 is not None else [] ) + ( [m2] if m2 is not None else [] )
        solution = solve_ivp(self.model, t_span, variables, dense_output=True, method=method)
        return solution


    def cumulative_incidence(self, solution, t_span : list [float | int]):
        """
        Internal method to calculate the cumulative incidence from the solution of the ODEs.

        Parameters
        ----------
        solution: OdeSolution
            The solution object returned by solve_ivp.
        t_span: list
            List of two elements [t_start, t_end] defining the time span for the simulation.

        Returns
        -------
        cumulative_incidence: array
            Array of cumulative incidence values corresponding to the time points in the solution.
        """
        t = np.linspace(t_span[0], t_span[1], 20000)
        I = solution[0]
        R = solution[1]

        M1 = solution[2] if len(solution) > 2 else None
        M2 = solution[3] if len(solution) > 3 else None

        if M1 is not None and M2 is not None:
            incidence = [self._physics.beta(0, m1, m2) * (1 - r - i) * i for i, r, m1, m2 in zip(I, R, M1, M2)]
        elif M1 is not None:
            incidence = [self._physics.beta(0, m1) * (1 - r - i) * i for i, r, m1 in zip(I, R, M1)]
        else:
            incidence = [self._physics.beta(None) * (1 - r - i) * i for i, r in zip(I, R)]

        # Integrating the incidence over time to get cumulative incidence, using the trapezoidal rule
        cumulative_incidence = np.trapezoid(incidence, t)

        return cumulative_incidence


    def simulate(self, t_span: list, initial_conditions: list):
        """
        Simulate the SIRS model over the given time span and initial conditions.

        Parameters
        ----------
        t_span: list
            List of two elements [t_start, t_end] defining the time span for the simulation.
        initial_conditions: list
            List of initial conditions for the model variables.
        """
        if self.model_type in ['sirs', 'sirs_zero_layer_incidence']:
            assert len(initial_conditions) == 2, "Expected 2 initial conditions for this model."
        elif self.model_type in ['sirs_zero_layer', 'sirs_one_layer', 'sirs_one_layer_incidence']:
            assert len(initial_conditions) == 3, "Expected 3 initial conditions for this model."
        elif self.model_type in ['sirs_two_layer', 'sirs_two_layer_incidence', 'sirs_two_layers_one_memory', 'sirs_two_layers_one_memory_incidence']:
            assert len(initial_conditions) == 4, "Expected 4 initial conditions for this model."

        t = np.linspace(t_span[0], t_span[1], 20000)

        i, r = initial_conditions[0], initial_conditions[1]
        m1 = initial_conditions[2] if len(initial_conditions) == 3 else None
        m2 = initial_conditions[3] if len(initial_conditions) == 4 else None

        solution = self._solve_odes(t, t_span, i, r, m1, m2)
        return solution.sol(t)


    @property
    def params(self) -> dict:
        """
        Return the model parameters.
        """
        return {k:v for k,v in vars(self).items()}


    def __str__(self) -> str:
        parameters = self.model.params
        return f"Model Configuration:\n{parameters}"



if __name__ == "__main__":
    params = {
        'model_type': 'sirs_one_layer',
        'r0': 2.5,
        'gamma': 1/7,
        'mu': 1/80/365,
        'theta': 1/365,
        'a1': 0.5,
        'a2': 0.5,
        'k1': 1.,
        'k2': 1.,
        'alpha1': 1.,
        'alpha2': 1.,
        'delta': 0.,
        'omega': 0.
    }

    model = SIRS(
        model_params = params
    )

    print(model.params)