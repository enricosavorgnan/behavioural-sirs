# Enrico Savorgnan
# University of Trieste, Italy, 2025
# A new model in “Behavioral Epidemiology of Infectious Diseases”: SIRS model with social distancing

"""
Provides simulation methods used in the thesis and in the jupyter file `simulations.ipynb`.
"""
import yaml
from code.core.models import SIRS
from code.core.plots import Plots
from code.core.utils import expr
import matplotlib.pyplot as plt


class Simulations:
    """
    Simulations method
    """

    def simulation_1(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 1: Different R₀ and θ parameters.
        """
        with open(config_path, 'r') as f:
            yaml.SafeLoader.add_constructor('!expr', expr)
            config = yaml.safe_load(f)

        # Simulate models
        r0s, thetas = config.get('r0', 1), config.get('theta', 1/365)
        assert len(r0s) == len(thetas), f"R0 and Theta list must have the same length"
        t_span = config.get('t_span', [0, 20000])
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])

        solutions = []
        for r0, theta in zip(r0s, thetas):
            params = {'r0': r0, 'theta': theta}
            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, initial_conditions=initial_conditions)
            solutions.append(solution)

        # Plot results
        show_cum_incidence = config.get('show_cumulative_incidence', False)
        show_params = config.get('show_params', False)
        show_title = config.get('show_title', False)

        if config.get('plot_together', False):
            params = {'image_path': f'./img/simulation_1/all_r0_{r0s}_theta_{thetas}.png'}
            fig = Plots(show_cum_incidence, show_params, show_title, True).plot_simulations(solutions=solutions, t_span=t_span, **params)
            return fig

        else:
            for i, solution in enumerate(solutions):
                params = {'image_path': f'./img/simulation_1/img_{i+1}/{len(solutions)}_r0_{round(r0s[i], 3)}_theta_{round(thetas[i], 3)}.png'}

                fig = Plots(show_cum_incidence, show_params, show_title, True).plot_simulation(solution, t_span, **params)
                fig.show()
            return None


    def simulation_2_memory(self, config_path : str) -> plt.Figure | None:
        """
        Plots the memory for different (R₀, θ).
        """

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Simulate models
        r0s, thetas = config.get('r0', 1), config.get('theta', 1/365)
        assert len(r0s) == len(thetas), f"R0 and Theta list must have the same length"
        t_span = config.get('t_span', [0, 20000])
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])

        solutions = []
        for r0, theta in zip(r0s, thetas):
            params = {'r0': r0, 'theta': theta}
            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, initial_conditions=initial_conditions)
            solutions.append(solution)

        # Plot results
        show_cum_incidence = config.get('show_cumulative_incidence', False)
        show_params = config.get('show_params', False)
        show_title = config.get('show_title', False)

        if config.get('plot_together', False):
            params = {'image_path': f'./img/simulation_2/all_r0_{r0s}_theta_{thetas}.png'}

            fig = Plots(show_cum_incidence, show_params, show_title, True).plot_memory(solutions=solutions, t_span=t_span, **params)
            return fig

        else:
            for i, solution in enumerate(solutions):
                params = {'image_path': f'./img/simulation_2/img_{i+1}/{len(solutions)}_r0_{r0s[i]}_theta_{thetas[i]}.png'}

                fig = Plots(show_cum_incidence, show_params, show_title, True).plot_memory(solution, t_span, **params)
                fig.show()
            return None


if __name__ == '__main__':
    Simulations().simulation_1(config_path = '../config/config_1.yaml')
