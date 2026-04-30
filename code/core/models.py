# Enrico Savorgnan
# University of Trieste, Italy, 2025
# A new model in “Behavioral Epidemiology of Infectious Diseases”: SIRS model with social distancing

"""
Class method building utilities for simulating Behavioral SIRS models
"""

import numpy as np
import yaml
from code.core.utils import expr
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from code.core.sirs_models import SIRSModels


class SIRS:
    """
    SIRS Models
    """
    def __init__(self,
                 model_params: dict | None = None,
                 config_path : str | None = None,
                 **kwargs):

        assert model_params is not None or config_path is not None, \
            f"One between model_params dictionary and config_path YAML file must be not None"

        if model_params is not None:
            self._set_params_with_dict(model_params = model_params)
        elif config_path is not None:
            self._set_params_with_yaml(config_path = config_path)
        # Useful for simulations with different values
        if kwargs is not None:
            self._override_params(**kwargs)

        self._physics = SIRSModels(
            model_type = self.model_type,
            r0 = self.r0,
            gamma = self.gamma,
            mu = self.mu,
            theta = self.theta,
            a1 = self.a1,
            a2 = self.a2,
            a3 = self.a3,
            k1 = self.k1,
            k2 = self.k2,
            k3 = self.k3,
            alpha1 = self.alpha1,
            alpha2 = self.alpha2,
            alpha3 = self.alpha3,
            delta = self.delta,
            omega = self.omega,
            T = self.T
        )

        self.model = getattr(self._physics, self.model_type)


    def _set_params_with_dict(self, model_params: dict):
        self.model_type : str = model_params.get('model_type', 'sirs')
        self.r0 : float  = model_params.get('r0', 0.)
        self.mu : float = model_params.get('mu', 0.)
        self.gamma : float = model_params.get('gamma', 0.)
        self.theta : float = model_params.get('theta', 0.)
        self.a1 : float = model_params.get('a1', 0.)
        self.a2 : float = model_params.get('a2', 0.)
        self.a3 : float = model_params.get('a3', 0.)
        self.k1 : float = model_params.get('k1', 1.)
        self.k2 : float = model_params.get('k2', 1.)
        self.k3 : float = model_params.get('k3', 1.)
        self.alpha1 : float = model_params.get('alpha1', 1.)
        self.alpha2 : float = model_params.get('alpha2', 1.)
        self.alpha3 : float = model_params.get('alpha3', 1.)
        self.delta : float = model_params.get('delta', 0)
        self.omega : float = model_params.get('omega', 2 * np.pi / 365)
        self.T : int = int(model_params.get('T', 14))

        # set values to float
        for k, v in vars(self).items():
            try:
                v = float(v)
                setattr(self, k, v)
            except (TypeError, ValueError):
                pass


    def _set_params_with_yaml(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.model_type : str = config.get('model_type', 'sirs')
        self.r0 : float  = config.get('r0', 0.)
        self.mu : float = config.get('mu', 0.)
        self.gamma : float = config.get('gamma', 0.)
        self.theta : float = config.get('theta', 0.)
        self.a1 : float = config.get('a1', 0.)
        self.a2 : float = config.get('a2', 0.)
        self.a3 : float = config.get('a3', 0.)
        self.k1 : float = config.get('k1', 1.)
        self.k2 : float = config.get('k2', 1.)
        self.k3 : float = config.get('k3', 1.)
        self.alpha1 : float = config.get('alpha1', 1.)
        self.alpha2 : float = config.get('alpha2', 1.)
        self.alpha3 : float = config.get('alpha3', 1.)
        self.delta : float = config.get('delta', 0)
        self.omega : float = config.get('omega', 2 * np.pi / 365)
        self.T : int = int(config.get('T', 14))

        # set values to float
        for k, v in vars(self).items():
            try:
                v = float(v)
                setattr(self, k, v)
            except (TypeError, ValueError):
                pass


    def _override_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Unknown parameter '{key}' provided in kwargs. No such attribute in SIRS class.")


    def _solve_odes(self,
                    t : list | np.ndarray,
                    t_span : list[float | int],
                    i : float,
                    r : float,
                    m1 : float | None = None,
                    m2 : float | None = None,
                    m3 : float | None = None,
                    method : str | None = 'RK45'):
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
        variables = [i, r] + ( [m1] if m1 is not None else [] ) + ( [m2] if m2 is not None else [] ) + ( [m3] if m3 is not None else [] )
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
        M3 = solution[4] if len(solution) > 4 else None

        if M1 is not None and M2 is not None and M3 is not None:
            incidence = [self._physics.beta(0, m1, m2, m3) * (1 - r - i) * i for i, r, m1, m2, m3 in zip(I, R, M1, M2, M3)]
        elif M1 is not None and M2 is not None:
            incidence = [self._physics.beta(0, m1, m2) * (1 - r - i) * i for i, r, m1, m2 in zip(I, R, M1, M2)]
        elif M1 is not None:
            incidence = [self._physics.beta(0, m1) * (1 - r - i) * i for i, r, m1 in zip(I, R, M1)]
        else:
            incidence = [self._physics.beta(None) * (1 - r - i) * i for i, r in zip(I, R)]

        # Integrating the incidence over time to get cumulative incidence, using the trapezoidal rule
        cumulative_incidence = np.trapezoid(incidence, t)

        return cumulative_incidence


    def find_equilibrium(self, initial_guess: list, t: float = 0.) -> np.ndarray:
        """
        Finds the exact mathematical equilibrium of the system using scipy.optimize.fsolve.

        Parameters
        ----------
        initial_guess : list
            A rough guess for the equilibrium state.
            For best results, use the theoretical endemic equilibrium of the
            standard SIR model as the base guess.
        t : float, optional
            The time at which to evaluate the model for finding the equilibrium.
            If None, the model will be evaluated at t=0.
        Returns
        -------
        np.ndarray
            The exact steady-state state variables [I*, R*, M1*, ...].
        """
        ode_func = lambda x: self.model(t, x)

        equilibrium, info, ier, mesg = fsolve(ode_func, initial_guess, full_output=True)

        if ier != 1:
            print(f"Warning: fsolve did not converge. Reason: {mesg}")

        return equilibrium


    def simulate(self, t_span: list, initial_conditions: list, n_points : int):
        """
        Simulate the SIRS model over the given time span and initial conditions.

        Parameters
        ----------
        t_span: list
            List of two elements [t_start, t_end] defining the time span for the simulation.
        initial_conditions: list
            List of initial conditions for the model variables.
        n_points : int
            Number of time points to evaluate the solution at.
        """
        if self.model_type in ['sirs']:
            assert len(initial_conditions) == 2, "Expected 2 initial conditions for this model."
        elif self.model_type in ['sirs_zero_layer', 'sirs_zero_layer_incidence', 'sirs_one_layer', 'sirs_one_layer_incidence', 'sirs_delay', 'sirs_delay_incidence']:
            assert len(initial_conditions) == 3, "Expected 3 initial conditions for this model."
        elif self.model_type in ['sirs_two_layer', 'sirs_two_layer_incidence', 'sirs_two_layers_one_memory', 'sirs_two_layers_one_memory_incidence']:
            assert len(initial_conditions) == 4, "Expected 4 initial conditions for this model."
        elif self.model_type in ['sirs_three_layer', 'sirs_three_layer_incidence,', 'sirs_three_layer_one_memory', 'sirs_three_layer_incidence_one_memory', 'sirs_three_layer_two_memory', 'sirs_three_layer_incidence_two_memory']:
            assert len(initial_conditions) == 5, "Expected 5 initial conditions for this model."

        t = np.linspace(t_span[0], t_span[1], num=n_points)

        i = initial_conditions[0]
        r = initial_conditions[1]
        m1 = initial_conditions[2] if len(initial_conditions) >= 3 else None
        m2 = initial_conditions[3] if len(initial_conditions) >= 4 else None
        m3 = initial_conditions[4] if len(initial_conditions) == 5 else None

        solution = self._solve_odes(t=t, t_span=t_span, i=i, r=r, m1=m1, m2=m2, m3=m3)
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
        'model_type': 'sirs_delay_incidence',
        'r0': 2.5,
        'gamma': 1/7,
        'mu': 1/80/365,
        'theta': 1/365,
        'a1': 1/15,
        'a2': 1/90,
        'a3' : 1/365/10,
        'k1': 1.,
        'k2': 1.,
        'k3' : 1.,
        'alpha1': 1.,
        'alpha2': 1.,
        'alpha3' : 0.,
        'delta': 0.,
        'omega': 0.,
        'T': 14
    }

    model = SIRS(
        model_params = params
    )

    print(model.params)