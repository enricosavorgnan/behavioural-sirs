# Enrico Savorgnan
# University of Trieste, Italy, 2025
# A new model in “Behavioral Epidemiology of Infectious Diseases”: SIRS model with social distancing

"""
Provides simulation methods used in the thesis and in the jupyter file `simulations.ipynb`.
"""
import numpy as np
import yaml
from code.core.models import SIRS
from code.core.plots import Plots
from code.core.utils import expr
import matplotlib.pyplot as plt
import datetime


class Simulations:
    """
    Simulations method
    """

    def load_yaml(self, config_path : str) -> dict:
        """
        Safe Load of a YAML file
        """
        with open(config_path, 'r') as f:
            yaml.SafeLoader.add_constructor('!expr', expr)
            config = yaml.safe_load(f)
        return config



    def retrieve_initial_conditions_given_model_type(self, initial_conditions : list[float | int], model_type : str) -> list[float]:
        """
        Retrieve the initial conditions given the model type.
        """
        if model_type == 'sirs':
            return initial_conditions[:2]
        elif model_type in ['sirs_two_layer', 'sirs_two_layer_incidence', 'sirs_two_layer_one_memory', 'sirs_two_layer_two_memory']:
            return initial_conditions[:4]
        else:
            return initial_conditions[:3]



    def simulation_0(self, config_path) -> plt.Figure | None:
        """
        Simulation 0: JASM (Just Another Sirs Model)
        """
        config = self.load_yaml(config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        plot_t_span = config.get('plot_t_span', t_span)
        plot_n_points = (n_points * (plot_t_span[1]-plot_t_span[0])) // (t_span[1]-t_span[0]) + 1

        model = SIRS(config_path=config_path)
        solution = model.simulate(t_span=t_span,
                                  initial_conditions=config.get('initial_conditions', [0.99, 0.01, 0]),
                                  n_points=config.get('n_points', 20000))
        S = 1. - solution[0] - solution[1]
        solution = np.vstack((S, solution))

        params = {'image_path': f'../img/simulation_0/simulation_0_r0_{config.get('r0', 2.5)}.pdf'}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=solution,
                                                         t_span=plot_t_span,
                                                         n_points=plot_n_points, **params)
        return fig



    def simulation_1(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 1: Different R₀ and θ parameters.
        """
        config = self.load_yaml(config_path)

        # Simulate models
        r0s, thetas = config.get('r0', 1), config.get('theta', 1/365)
        assert len(r0s) == len(thetas), f"R0 and Theta list must have the same length"

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])

        solutions = []
        for r0, theta in zip(r0s, thetas):
            params = {'r0': r0, 'theta': theta}
            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, initial_conditions=initial_conditions, n_points=n_points)
            solutions.append(solution)

        # Plot results
        plot_t_span = config.get('plot_t_span', t_span)
        plot_n_points = (n_points * (plot_t_span[1]-plot_t_span[0])) // (t_span[1]-t_span[0]) + 1

        if config.get('plot_together', False):
            params = {'image_path': f'../img/simulation_1/all_r0_{r0s}_theta_{[round(theta, 3) for theta in thetas]}.pdf'}
            fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                        show_params = config.get('show_params', False),
                        show_title = config.get('show_title', False),
                        save_figures = True).plot_simulations(solutions=solutions, t_span=plot_t_span, n_points=plot_n_points,  **params)
            return fig

        else:
            for i, solution in enumerate(solutions):
                params = {'image_path': f'../img/simulation_1/img_{i+1}-{len(solutions)}_r0_{round(r0s[i], 3)}_theta_{round(thetas[i], 3)}.pdf'}
                fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                            show_params = config.get('show_params', False),
                            show_title = config.get('show_title', False),
                            save_figures = True).plot_simulation(solution=solution, t_span=plot_t_span, n_points=plot_n_points, **params)
                fig.show()
            return None



    def simulation_2(self, config_path : str) -> plt.Figure | None:
        """
        Plots the memory for different (R₀, θ).
        """
        config = self.load_yaml(config_path)

        # Simulate models
        r0s, thetas = config.get('r0', 1), config.get('theta', 1/365)
        assert len(r0s) == len(thetas), f"R0 and Theta list must have the same length"

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])

        solutions = []
        for r0, theta in zip(r0s, thetas):
            params = {'r0': r0, 'theta': theta}
            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, initial_conditions=initial_conditions, n_points=n_points)
            solutions.append(solution)


        # Plot results
        plot_t_span = config.get('plot_t_span', t_span)
        plot_n_points = (n_points * (plot_t_span[1]-plot_t_span[0])) // (t_span[1]-t_span[0]) + 1

        if config.get('plot_together', False):
            params = {'image_path': f'../img/simulation_2/all_r0_{r0s}_theta_{[round(theta, 3) for theta in thetas]}.pdf'}
            fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                        show_params = config.get('show_params', False),
                        show_title = config.get('show_title', False),
                        save_figures = True).plot_memory(solutions=solutions, t_span=plot_t_span, n_points=plot_n_points, **params)
            return fig

        else:
            for i, solution in enumerate(solutions):
                params = {'image_path': f'../img/simulation_2/img_{i+1}-{len(solutions)}_r0_{r0s[i]}_theta_{round(thetas[i], 3)}.pdf'}
                fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                            show_params = config.get('show_params', False),
                            show_title = config.get('show_title', False),
                            save_figures = True).plot_memory(solutions=solution, t_span=plot_t_span, n_points=plot_n_points, **params)
                fig.show()
            return None


    def simulation_3(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 3: different k
        """
        config = self.load_yaml(config_path)

        # simulate models
        alphas = config.get('alpha1', 2)
        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        plot_t_span = config.get('plot_t_span', t_span)
        plot_n_points = (n_points * (plot_t_span[1]-plot_t_span[0])) // (t_span[1]-t_span[0]) + 1
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])

        solutions = []
        for alpha in alphas:
            params = {'alpha1': alpha, 'alpha2': alpha}
            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conditions)
            solutions.append(solution[0])

        # Plot results
        params = {'image_path': f'../img/simulation_3/all_alpha_{alphas}.pdf'}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=solutions, t_span=plot_t_span, n_points=plot_n_points, **params)
        fig.show()
        return fig



    def simulation_4(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 4: different models types, parameters fixed
        """
        config = self.load_yaml(config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        plot_t_span = config.get('plot_t_span', t_span)
        plot_n_points = (n_points * (plot_t_span[1]-plot_t_span[0])) // (t_span[1]-t_span[0]) + 1
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])

        solutions = []
        for mod_type in config.get('model_type', 'sirs'):
            params = {'model_type': mod_type}
            initial_conds = self.retrieve_initial_conditions_given_model_type(initial_conditions, mod_type)

            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conds)
            solutions.append(solution[0])


        params = {'image_path': f'../img/simulation_4/all_models_{config.get("model_type", "sirs")}.pdf'}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=solutions, t_span=plot_t_span, n_points=plot_n_points, **params)
        fig.show()

        return fig



    def simulation_5(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 5: different model types, different R0, theta combinations
        It is like Simulation 1, but with additional focus on differnt models.
        """
        config = self.load_yaml(config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        plot_t_span = [[0, 2000], [0, 1250], [0, 400], [0, 400]]
        plot_n_points = [(n_points * (plot_t_span[i][1]-plot_t_span[i][0])) // (t_span[1]-t_span[0]) + 1 for i in range(4)]
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])

        r0s, thetas = config.get('r0', 2.), config.get('theta', 2.)
        model_types = config.get('model_type', 'sirs')

        solutions = []
        for r0, theta in zip(r0s, thetas):
            solutions_mid = []
            for mod_type in model_types:
                initial_conds = self.retrieve_initial_conditions_given_model_type(initial_conditions, mod_type)
                params = {'model_type': mod_type, 'r0': r0, 'theta': theta}

                model = SIRS(config_path=config_path, **params)
                solution =  model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conds)
                solutions_mid.append(solution[0])
            solutions.append(solutions_mid)

        if config.get('plot_together', False):
            params = {'image_path': f'../img/simulation_5/all_model_type_{model_types}_r0_{r0s}_theta_{[round(theta, 3) for theta in thetas]}.pdf'}
            fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                        show_params = config.get('show_params', False),
                        show_title = config.get('show_title', False),
                        save_figures = True).plot_simulations(solutions=solutions, t_span=plot_t_span, n_points=plot_n_points, **params)
            return fig

        else:
            for i, solution in enumerate(solutions):
                params = {'image_path': f'../img/simulation_5/img_{i+1}-{len(solutions)}_model_type_{model_types[i]}_r0_{r0s[i]}_theta_{round(thetas[i], 3)}.pdf'}
                fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                            show_params = config.get('show_params', False),
                            show_title = config.get('show_title', False),
                            save_figures = True).plot_simulation(solution=solution, t_span=plot_t_span[i], n_points=plot_n_points[i], **params)
                fig.show()
            return None



    def simulation_6(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 6: different a1, a2
        """
        config = self.load_yaml(config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        plot_t_span = config.get('plot_t_span', t_span)
        plot_n_points = (n_points * (plot_t_span[1]-plot_t_span[0])) // (t_span[1]-t_span[0]) + 1
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])

        a1s = config.get('a1', 1/30)
        a2s = config.get('a2', 1/90)

        solutions = []
        for a1, a2 in zip(a1s, a2s):
            params = {'a1': a1, 'a2': a2}
            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conditions)
            solutions.append(solution[2] + solution[3])

        # Plot results
        params = {'image_path': f'../img/simulation_6/all_a1_{[round(a1, 3) for a1 in a1s]}_a2_{[round(a2, 3) for a2 in a2s]}_time_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=solutions, t_span=plot_t_span, n_points=plot_n_points, **params)
        fig.show()
        return fig




if __name__ == '__main__':
    Simulations().simulation_6(config_path = '../config/config_6.yaml')
