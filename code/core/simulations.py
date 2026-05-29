# Enrico Savorgnan
# University of Trieste, Italy, 2025
# A new model in “Behavioral Epidemiology of Infectious Diseases”: SIRS model with social distancing

"""
Provides simulation methods used in the thesis and in the jupyter file `simulations.ipynb`.
"""
import numpy as np
import yaml
import datetime
import argparse
import matplotlib.pyplot as plt

from code.core.models import SIRS

from code.core.maths import RH_FifthOrder, RH_ForthOrder, RH_ThirdOrder
from code.core.maths import EigenStability, DelayStability

from code.core.plots import Plots
from code.core.utils import expr



class Simulations:
    """
    Simulations method
    """

    def _load_yaml(self, config_path : str) -> dict:
        """
        Safe Load of a YAML file
        """
        with open(config_path, 'r') as f:
            yaml.SafeLoader.add_constructor('!expr', expr)
            config = yaml.safe_load(f)
        return config



    def _retrieve_initial_conditions_given_model_type(self, initial_conditions : list[float | int], model_type : str) -> list[float]:
        """
        Retrieve the initial conditions given the model type.
        """
        if model_type == 'sirs':
            return initial_conditions[:2]
        elif model_type in ['sirs_two_layer', 'sirs_two_layer_incidence', 'sirs_two_layer_one_memory', 'sirs_two_layer_incidence_one_memory']:
            return initial_conditions[:4]
        else:
            return initial_conditions[:3]



    def _retrieve_model_string(self, model_types : str | list) -> str | list | None:
        """
        Retrieve the model string given the model type.
        """
        match_type = {
            'sirs' : 'sirs',
            'sirs_zero_layer' : '0L',
            'sirs_one_layer' : '1L',
            'sirs_two_layer' : '2L',
            'sirs_three_layer': '3L',

            'sirs_zero_layer_incidence' : '0LI',
            'sirs_one_layer_incidence' : '1LI',
            'sirs_two_layer_incidence' : '2LI',
            'sirs_three_layer_incidence' : '3LI',

            'sirs_two_layer_one_memory' : '2L1M',
            'sirs_two_layer_incidence_one_memory' : '2LI1M',
            'sirs_three_layer_one_memory' : '3L1M',
            'sirs_three_layer_incidence_one_memory' : '3LI1M',
            'sirs_three_layer_two_memory' : '3L2M',
            'sirs_three_layer_incidence_two_memory' : '3LI2M',

            'sirs_delay' : 'D',
            'sirs_delay_incidence' : 'DI'
        }
        if type(model_types) == str:
            return match_type[model_types]
        elif type(model_types) == list:
            return [match_type[mod_type] for mod_type in model_types]
        return None



    def _retrieve_img_path(self, config_path : str, n_simulation : int | str, sim_17_idx : list[str] | None = None, **kwargs) -> str:
        """
        Retrieve Image Path given the configuration file and the simulation's number
        """
        config = self._load_yaml(config_path)

        model_type = config.get('model_type', 'sirs')
        str_model_type = self._retrieve_model_string(model_type)

        r0 = config.get('r0', 2.5)
        theta = round(config.get('theta', 1/365), 3) if type(config.get('theta', 1/365)) == float else [round(theta, 3) for theta in config.get('theta', 1/365)]
        alpha = round(config.get('alpha1', 50.), 3) if type(config.get('alpha1', 50.)) == float else [round(alpha, 3) for alpha in config.get('alpha1', 50.)]
        k = round(config.get('k1', 1), 3) if type(config.get('k1', 1)) == float else [round(k, 3) for k in config.get('k1', 1)]
        a1 = round(config.get('a1', 1/30), 3) if type(config.get('a1', 1/30)) == float else [round(a1, 3) for a1 in config.get('a1', 1/30)]
        a2 = round(config.get('a2', 1/90), 3) if type(config.get('a2', 1/90)) == float else [round(a2, 3) for a2 in config.get('a2', 1/90)]

        img_folder = config.get("plot_path", f'../img/simulation_{n_simulation}/')
        img_path = f'model_{str_model_type}_r0_{r0}_theta_{theta}_k_{k}_a1_{a1}_a2_{a2}_alpha_{alpha}_time_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'

        if n_simulation == '17' and kwargs.get('plot_type', '3D') == 'Combo':
            img_path = f'combo_{sim_17_idx[0]}_{sim_17_idx[1]}_model_{model_type}_time_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        if model_type in ['sirs_delay', 'sirs_delay_incidence']:
            T = int(config.get('T', 14) )
            img_path = f'model_{str_model_type}_r0_{r0}_theta_{theta}_T_{T}_alpha_{alpha}_time_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        path = img_folder + img_path
        return path



    def _retrieve_list_img_path(self, config_path : str, n_simulation : int | str, n_imgs : int, curr_i : int):
        """
        Retrieve the image path if the image is a member of a list of images
        """
        img_path = self._retrieve_img_path(config_path=config_path, n_simulation=n_simulation)
        img_path_chunks = img_path.rsplit('/', 3)
        img_folder = img_path_chunks[0] + '/' + img_path_chunks[1] + '/' + img_path_chunks[2] + '/'
        img_file = f'img_{curr_i+1}-{n_imgs}' + img_path_chunks[-1]

        return img_folder + img_file



    def simulation_0(self, config_path) -> plt.Figure | None:
        """
        Simulation 0: JASM (Just Another Sirs Model)
        """
        config = self._load_yaml(config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        plot_t_span = config.get('plot_t_span', t_span)
        plot_n_points = config.get('plot_n_points', (n_points * (plot_t_span[1]-plot_t_span[0])) // (t_span[1]-t_span[0]) + 1)

        model = SIRS(config_path=config_path)
        solution = model.simulate(t_span=t_span,
                                  initial_conditions=config.get('initial_conditions', [0.99, 0.01, 0]),
                                  n_points=config.get('n_points', 20000))
        S = 1. - solution[0] - solution[1]
        solution = np.vstack((S, solution))
        solution = solution[:3]

        params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='00')}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=solution,
                                                         t_span=plot_t_span,
                                                         n_points=plot_n_points, **params)
        fig.show()

        return fig



    def simulation_1(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 1: Different R₀ and θ parameters.
        """
        config = self._load_yaml(config_path)

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
            params = {'image_path': f'../img/simulation_01/all_r0_{r0s}_theta_{[round(theta, 3) for theta in thetas]}.pdf'}
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
                return None



    def simulation_2(self, config_path : str) -> plt.Figure | None:
        """
        Plots the memory for different (R₀, θ).
        """
        config = self._load_yaml(config_path)

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
            params = {'image_path': f'../img/simulation_02/all_r0_{r0s}_theta_{[round(theta, 3) for theta in thetas]}.pdf'}
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
            return None


    def simulation_3(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 3: different k
        """
        config = self._load_yaml(config_path)

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
        params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='03')}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=solutions, t_span=plot_t_span, n_points=plot_n_points, **params)
        return fig



    def simulation_4(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 4: different models types, parameters fixed
        """
        config = self._load_yaml(config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        plot_t_span = config.get('plot_t_span', t_span)
        plot_n_points = (n_points * (plot_t_span[1]-plot_t_span[0])) // (t_span[1]-t_span[0]) + 1
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])

        solutions = []
        for mod_type in config.get('model_type', 'sirs'):
            params = {'model_type': mod_type}
            initial_conds = self._retrieve_initial_conditions_given_model_type(initial_conditions, mod_type)

            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conds)
            solutions.append(solution[0])


        params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='04')}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=solutions, t_span=plot_t_span, n_points=plot_n_points, **params)

        return fig



    def simulation_5(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 5: different model types, different R0, theta combinations
        It is like Simulation 1, but with additional focus on different models.
        """
        config = self._load_yaml(config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        plot_t_span = [[0, 2500], [0, 1250], [0, 400], [0, 400]]
        plot_n_points = [(n_points * (plot_t_span[i][1]-plot_t_span[i][0])) // (t_span[1]-t_span[0]) + 1 for i in range(4)]
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])

        r0s, thetas = config.get('r0', 2.), config.get('theta', 2.)
        model_types = config.get('model_type', 'sirs')

        solutions = []
        for r0, theta in zip(r0s, thetas):
            solutions_mid = []
            for mod_type in model_types:
                initial_conds = self._retrieve_initial_conditions_given_model_type(initial_conditions, mod_type)
                params = {'model_type': mod_type, 'r0': r0, 'theta': theta}

                model = SIRS(config_path=config_path, **params)
                solution =  model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conds)
                solutions_mid.append(solution[0])
            solutions.append(solutions_mid)

        if config.get('plot_together', False):
            params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='05')}
            fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                        show_params = config.get('show_params', False),
                        show_title = config.get('show_title', False),
                        save_figures = True).plot_simulations(solutions=solutions, t_span=plot_t_span, n_points=plot_n_points, **params)
            return fig

        else:
            for i, solution in enumerate(solutions):
                params = {'image_path': self._retrieve_list_img_path(config_path=config_path, n_simulation='05', n_imgs=len(solutions), curr_i=i)}
                fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                            show_params = config.get('show_params', False),
                            show_title = config.get('show_title', False),
                            save_figures = True).plot_simulation(solution=solution, t_span=plot_t_span[i], n_points=plot_n_points[i], **params)
            return None



    def simulation_6(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 6: different a1, a2
        """
        config = self._load_yaml(config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        plot_t_span = config.get('plot_t_span', t_span)
        plot_n_points = (n_points * (plot_t_span[1]-plot_t_span[0])) // (t_span[1]-t_span[0]) + 1
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])
        model_type = config.get('model_type', 'sirs')

        a1s = config.get('a1', 1/30)
        a2s = config.get('a2', 1/90)

        solutions = []
        for a1, a2 in zip(a1s, a2s):
            params = {'a1': a1, 'a2': a2}
            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conditions)
            if model_type.startswith('sirs_two_layer'):
                solutions.append(solution[2] + solution[3])
            elif model_type.startswith('sirs_one_layer'):
                solutions.append(solution[2])

        # Plot results
        params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='06')}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=solutions, t_span=plot_t_span, n_points=plot_n_points, **params)
        return fig



    def simulation_7(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 7: different a1, a2, model types
        """
        config = self._load_yaml(config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        plot_t_span = [[0, 500], [0, 500], [0, 1500], [0, 1500]]
        plot_n_points = [(n_points * (plot_t_span[i][1]-plot_t_span[i][0])) // (t_span[1]-t_span[0]) + 1 for i in range(4)]
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])

        a1s, a2s = config.get('a1', 2.), config.get('a2', 2.)
        model_types = config.get('model_type', 'sirs')

        solutions = []
        for a1, a2 in zip(a1s, a2s):
            solutions_mid = []
            for mod_type in model_types:
                initial_conds = self._retrieve_initial_conditions_given_model_type(initial_conditions, mod_type)
                params = {'model_type': mod_type, 'a1': a1, 'a2': a2}

                model = SIRS(config_path=config_path, **params)
                solution =  model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conds)
                solutions_mid.append(solution[0])
            solutions.append(solutions_mid)

        if config.get('plot_together', False):
            params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='07')}
            fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                        show_params = config.get('show_params', False),
                        show_title = config.get('show_title', False),
                        save_figures = True).plot_simulations(solutions=solutions, t_span=plot_t_span, n_points=plot_n_points, **params)
            return fig

        else:
            for i, solution in enumerate(solutions):
                params = {'image_path': self._retrieve_list_img_path(config_path=config_path, n_simulation='07', n_imgs=len(solutions), curr_i=i)}
                fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                            show_params = config.get('show_params', False),
                            show_title = config.get('show_title', False),
                            save_figures = True).plot_simulation(solution=solution, t_span=plot_t_span[i], n_points=plot_n_points[i], **params)
            return None


    def simulation_8(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 8: different alpha1, alpha2, model types
        """
        config = self._load_yaml(config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        plot_t_span = [[0, 1500], [0, 1500], [0, 750], [0, 750]]
        plot_n_points = [(n_points * (plot_t_span[i][1]-plot_t_span[i][0])) // (t_span[1]-t_span[0]) + 1 for i in range(4)]
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])

        alpha1s, alpha2s = config.get('alpha1', 2.), config.get('alpha2', 2.)
        model_types = config.get('model_type', 'sirs')

        solutions = []
        for alpha1, alpha2 in zip(alpha1s, alpha2s):
            solutions_mid = []
            for mod_type in model_types:
                initial_conds = self._retrieve_initial_conditions_given_model_type(initial_conditions, mod_type)
                params = {'model_type': mod_type, 'alpha1': alpha1, 'alpha2': alpha2}

                model = SIRS(config_path=config_path, **params)
                solution =  model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conds)
                solutions_mid.append(solution[0])
            solutions.append(solutions_mid)

        if config.get('plot_together', False):
            params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='08')}
            fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                        show_params = config.get('show_params', False),
                        show_title = config.get('show_title', False),
                        save_figures = True).plot_simulations(solutions=solutions, t_span=plot_t_span, n_points=plot_n_points, **params)
            return fig

        else:
            for i, solution in enumerate(solutions):
                params = {'image_path': self._retrieve_list_img_path(config_path=config_path, n_simulation='08', n_imgs=len(solutions), curr_i=i),
                          'legend': self._retrieve_model_string(model_types=config.get('model_type', 'sirs')) }
                fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                            show_params = config.get('show_params', False),
                            show_title = config.get('show_title', False),
                            show_legend= config.get('show_legend', False),
                            save_figures = True).plot_simulation(solution=solution, t_span=plot_t_span[i], n_points=plot_n_points[i], **params)
            return None


    def simulation_9(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 9: different alphas, plot I_e
        """
        config = self._load_yaml(config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])
        range_alpha = config.get('range_alpha', [0., 100.])
        n_alpha = config.get('n_alpha', 100)

        alphas = np.linspace(range_alpha[0], range_alpha[1], n_alpha)

        ies = []
        for alpha in alphas:
            params = {'alpha1': alpha, 'alpha2': alpha}

            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conditions)
            ies.append(solution[0][-1])

        params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='09')}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=ies, t_span=range_alpha, n_points=n_alpha, **params)
        return fig


    def simulation_10(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 10: different mu, plot EE_{mu=0} - EE_{mu!=0}
        """
        config = self._load_yaml(config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])
        range_r0 = config.get('range_r0', [1.001, 10.])
        n_r0 = config.get('n_r0', 100)

        r0s = np.linspace(range_r0[0], range_r0[1], n_r0)

        r0_res = []
        for r0 in r0s:
            params = {'r0': r0}

            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conditions)

            params_mu_zero = {'mu': 0, 'r0': r0}
            model_mu_zero = SIRS(config_path=config_path, **params_mu_zero)
            solution_mu_zero = model_mu_zero.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conditions)

            Ie = solution[0][-1]
            Ie_zero = solution_mu_zero[0][-1]

            Re = solution[1][-1]
            Re_zero = solution_mu_zero[1][-1]

            Se = 1 - Ie - Re
            Se_zero = 1 - Ie_zero - Re_zero

            norm = np.linalg.norm([Se-Se_zero, Ie - Ie_zero, Re - Re_zero])
            r0_res.append(norm)

        params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='10')}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=r0_res, t_span=range_r0, n_points=n_r0, **params)

        gamma = config.get('gamma', 0.1)
        mu = config.get('mu', 0.1)
        theta = config.get('theta', 0.1)
        alpha = config.get('alpha1', 0.1)
        t = np.linspace(range_r0[0], range_r0[1], n_r0)
        y = lambda x: (x - 1) / ( x * (gamma/(mu + theta) +1) + alpha)
        z = lambda x: 1 / (gamma/(mu + theta) + 1 ) * (1 - 1/2/x)
        ax = fig.gca()
        ax.plot(t, y(t))
        ax.plot(t, z(t))
        fig.show()
        return fig



    def simulation_11(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 11: different alphas, plot EE_{alpha=0} - EE_{alpha!=0}
        """
        config = self._load_yaml(config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        plot_t_span = config.get('plot_t_span', t_span)
        plot_n_points = (n_points * (plot_t_span[1]-plot_t_span[0])) // (t_span[1]-t_span[0]) + 1
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])

        solutions = []
        for mod_type in config.get('model_type', 'sirs'):
            params = {'model_type': mod_type}
            initial_conds = self._retrieve_initial_conditions_given_model_type(initial_conditions, mod_type)

            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conds)
            solutions.append(solution[0])

        sirs_zero = SIRS(config_path=config_path, **{'mu': 0, 'model_type': 'sirs'})
        initial_conds_zero = self._retrieve_initial_conditions_given_model_type(initial_conditions, 'sirs')
        solution_zero = sirs_zero.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conds_zero)
        solutions.append([solution_zero[0][-1]]*len(solution_zero[0]))

        params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='11')}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=solutions, t_span=plot_t_span, n_points=plot_n_points, **params)

        return fig


    def simulation_12(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 12: different a1, a2, model_type
        """
        config = self._load_yaml(config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)
        plot_t_span = config.get('plot_t_span', t_span)
        plot_n_points = (n_points * (plot_t_span[1]-plot_t_span[0])) // (t_span[1]-t_span[0]) + 1
        initial_conditions = config.get('initial_conditions', [0.99, 0.01, 0])
        model_type = config.get('model_type', 'sirs')

        a1s = config.get('a1', 1/30)
        a2s = config.get('a2', 1/90)

        solutions = []
        for a1, a2 in zip(a1s, a2s):
            params = {'a1': a1, 'a2': a2}
            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conditions)
            solutions.append(solution[0])

        # Plot results
        params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='12')}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=solutions, t_span=plot_t_span, n_points=plot_n_points, **params)
        return fig


    def simulation_13(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 13: Routh-Hurwitz stability for 4th order polynomial
        """
        config = self._load_yaml(config_path=config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)

        target = config.get('target', 'a1')
        assert target in ['a1'], f"Unknown target: {target}"
        target_span = config.get('target_span', [0, 1])
        target_n_points = config.get('target_n_points', 10)
        initial_conditions = config.get('initial_conditions', [0.999, 0.001, 0.])

        targets = np.linspace(target_span[0], target_span[1], target_n_points)
        rh_cond_s = []
        for targ in targets:
            params = {target: targ}
            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conditions)
            equilibrium = [solution[0][-1], solution[1][-1], solution[2][-1]]
            rh_cond = RH_ThirdOrder(target=target, equilibrium=equilibrium, model=model).compute(x=targ)
            rh_cond_s.append(rh_cond)

        rhs = np.array(rh_cond_s)

        params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='13')}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=rhs, t_span=target_span, n_points=target_n_points, **params)
        return fig


    def simulation_14(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 14: Routh-Hurwitz stability for 5th order polynomial
        """
        config = self._load_yaml(config_path=config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)

        target = config.get('target', 'a1')
        assert target in ['a1', 'a2'], f"Unknown target: {target}"
        target_span = config.get('target_span', [0, 1])
        target_n_points = config.get('target_n_points', 10)
        initial_conditions = config.get('initial_conditions', [0.999, 0.001, 0.])

        targets = np.linspace(target_span[0], target_span[1], target_n_points)
        rh_cond1_s, rh_cond2_s = [], []
        for targ in targets:
            params = {target: targ}
            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conditions)
            equilibrium = [solution[0][-1], solution[1][-1], solution[2][-1], solution[3][-1]]
            rh_cond_1, rh_cond_2 = RH_ForthOrder(target=target, equilibrium=equilibrium, model=model).compute(x=targ)
            rh_cond1_s.append(rh_cond_1)
            rh_cond2_s.append(rh_cond_2)

        rhs = np.array([rh_cond1_s, rh_cond2_s])
        # rhs = np.array([rh_cond2_s])

        params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='14')}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=rhs, t_span=target_span, n_points=target_n_points, **params)
        return fig


    def simulation_15(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 15: Routh-Hurwitz stability for 5th order polynomial
        """
        config = self._load_yaml(config_path=config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)

        target = config.get('target', 'a1')
        assert target in ['a1', 'a2', 'a3'], f"Unknown target: {target}"
        target_span = config.get('target_span', [0, 1])
        target_n_points = config.get('target_n_points', 10)
        initial_conditions = config.get('initial_conditions', [0.999, 0.001, 0.])

        targets = np.linspace(target_span[0], target_span[1], target_n_points)
        rh_cond1_s, rh_cond2_s = [], []
        for targ in targets:
            params = {target: targ}
            model = SIRS(config_path=config_path, **params)
            solution = model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conditions)
            equilibrium = [solution[0][-1], solution[1][-1], solution[2][-1], solution[3][-1], solution[4][-1]]
            rh_cond_1, rh_cond_2 = RH_FifthOrder(target=target, equilibrium=equilibrium, model=model).compute(x=targ)
            rh_cond1_s.append(rh_cond_1)
            rh_cond2_s.append(rh_cond_2)

        # rhs = np.array([rh_cond1_s, rh_cond2_s])
        rhs = np.array([rh_cond2_s])

        params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='15')}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=rhs, t_span=target_span, n_points=target_n_points, **params)
        return fig



    def simulation_16(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 16: Routh-Hurwitz stability for 2nd order polynomial. Plot as a function of a1 and a2 together.
        """
        config = self._load_yaml(config_path=config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)

        target_span = config.get('target_span', [0, 1])
        target_n_points = config.get('target_n_points', 10)
        initial_conditions = config.get('initial_conditions', [0.999, 0.001, 0.])

        rhs = []
        targets = np.linspace(target_span[0], target_span[1], target_n_points)
        for i, targ1 in enumerate(targets):
            print(f"\n\n\nSimulating for a1 = {targ1:.3f} ({i+1}/{target_n_points})")
            for j, targ2 in enumerate(targets):
                if j%20 == 0:
                    print(f"Simulating for a2 = {targ2:.3f} ({j+1}/{target_n_points})")
                params = {'a1': targ1, 'a2': targ2}
                model = SIRS(config_path=config_path, **params)
                solution = model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conditions)

                equilibrium = [solution[0][-1], solution[1][-1], solution[2][-1], solution[3][-1]]
                _, rh_cond_2 = RH_ForthOrder(target=['a1', 'a2'], equilibrium=equilibrium, model=model).compute(x=targ1, y=targ2, **{'verbose': True})
                if rh_cond_2 < 0:
                    print(f"I FOUND YOU: a1 = {targ1:.3f}, a2 = {targ2:.3f}, rh_cond_2 = {rh_cond_2:.3f}")
                rhs.append(rh_cond_2)


        params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='16')}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_legend = config.get('show_legend', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_rh(solution=rhs, x_span=target_span, y_span=target_span, n_points=target_n_points, **params)
        return fig


    def simulation_17(self, config_path : str) -> plt.Figure | None:
        """
        Simulation 17: Routh-Hurwitz stability for 4th order polynomial. Plot as a function of a1 and a2 together.
        """
        config = self._load_yaml(config_path=config_path)

        t_span = config.get('t_span', [0, 20000])
        n_points = config.get('n_points', 20000)

        target_span = config.get('target_span', [0, 1])
        target_n_points = config.get('target_n_points', 10)
        initial_conditions = config.get('initial_conditions', [0.999, 0.001, 0.])

        rhs = []
        targets = np.linspace(target_span[0], target_span[1], target_n_points)
        for i, targ1 in enumerate(targets):
            print(f"\n\n\nSimulating for a1 = {targ1:.3f} ({i+1}/{target_n_points})")
            for j, targ2 in enumerate(targets):
                if j%20 == 0:
                    print(f"Simulating for a2 = {targ2:.3f} ({j+1}/{target_n_points})")
                for k, targ3 in enumerate(targets):
                    if k%20 == 0:
                        print(f"Simulating for a3 = {targ3:.3f} ({k+1}/{target_n_points})")
                    params = {'a1': targ1, 'a2': targ2, 'a3': targ3}
                    model = SIRS(config_path=config_path, **params)
                    solution = model.simulate(t_span=t_span, n_points=n_points, initial_conditions=initial_conditions)

                    equilibrium = [solution[0][-1], solution[1][-1], solution[2][-1], solution[3][-1], solution[4][-1]]
                    rh_cond_2, _  = RH_FifthOrder(target=['a1', 'a2', 'a3'], equilibrium=equilibrium, model=model).compute(x=targ1, y=targ2, z=targ3, **{'verbose': False})
                    if rh_cond_2 < 0:
                        print(f"I FOUND YOU: a1 = {targ1:.12f}, a2 = {targ2:.12f}, rh_cond_2 = {rh_cond_2:.12f}")
                    rhs.append(rh_cond_2)

        rhs_3d = np.array(rhs).reshape((target_n_points, target_n_points, target_n_points))
        all_targets = ['a1', 'a2', 'a3']

        plot_type = config.get('plot_type', '3D') # Options: 'Combo', '3D', 'Voxel'
        print(f"Plotting with plot_type = {plot_type}")
        if plot_type == '3D':
            params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='17')}
            fig = Plots(show_cumulative_incidence=config.get('show_cumulative_incidence', False),
                        show_params=config.get('show_params', False),
                        show_legend=config.get('show_legend', False),
                        show_title=config.get('show_title', False),
                        save_figures=True).plot_rh_3d(
                solution=rhs_3d,
                x_span=target_span,
                y_span=target_span,
                z_span=target_span,
                n_points=target_n_points,
                **params
            )
            return fig

        elif plot_type == 'Voxel':
            params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='17')}
            fig = Plots(show_cumulative_incidence=config.get('show_cumulative_incidence', False),
                        show_params=config.get('show_params', False),
                        show_legend=config.get('show_legend', False),
                        show_title=config.get('show_title', False),
                        save_figures=True).plot_rh_voxel(
                solution=rhs_3d,
                x_span=target_span,
                y_span=target_span,
                z_span=target_span,
                n_points=target_n_points,
                **params
            )
            return fig

        else:
            for combo_idx in [['a1', 'a2'], ['a1', 'a3'], ['a2', 'a3']]:

                # Identify which parameter axis is NOT in the current combination
                missing_axis = [idx for idx, target in enumerate(all_targets) if target not in combo_idx][0]

                # Project the 3D array down to 2D for plotting.
                # Using np.min to visualize worst-case stability over the hidden dimension.
                # (You can change this to np.mean() or take a middle slice like rhs_3d.take(target_n_points//2, axis=missing_axis))
                rhs_combo = np.min(rhs_3d, axis=missing_axis)

                rhs_flat = rhs_combo.flatten()

                # 3. Plot the 2D grid
                params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='17', sim_17_idx=combo_idx)}
                fig = Plots(show_cumulative_incidence=config.get('show_cumulative_incidence', False),
                            show_params=config.get('show_params', False),
                            show_legend=config.get('show_legend', False),
                            show_title=config.get('show_title', False),
                            save_figures=True).plot_rh(
                    solution=rhs_flat,
                    x_span=target_span,
                    y_span=target_span,
                    n_points=target_n_points,
                    **params
                )

        return fig


    def simulation_18(self, config_path: str) -> plt.Figure | None:
        """
        Simulation 18: Routh-Hurwitz stability for 5th order polynomial. Plot as a function of a1, a2, a3 together.
        Like simulation 17, but plotting the eigenvalue directly.
        """
        config = self._load_yaml(config_path=config_path)

        target_span = config.get('target_span', [0, 1])
        target_n_points = config.get('target_n_points', 10) # Change this to ~50 and use np.logspace in production!

        # Base analytical estimate from standard SIR endemic state to feed to fsolve
        r0 = config.get('r0', 2.5)
        mu = config.get('mu', 1/80/365)
        theta = config.get('theta', 1/365)
        gamma = config.get('gamma', 1/7)
        S_guess = 1 / r0
        I_guess = (mu + theta) * (1 - S_guess) / (mu + gamma + theta)
        R_guess = 1 - S_guess - I_guess
        # Best guess for memory is I itself
        initial_guess = [I_guess, R_guess, I_guess, I_guess, I_guess]

        targets = np.linspace(target_span[0], target_span[1], target_n_points)

        rhs_3d = np.zeros((target_n_points, target_n_points, target_n_points))
        all_targets = ['a1', 'a2', 'a3']

        for i, targ1 in enumerate(targets):
            if i%10 == 0:
                print(f"\n\n\n\n\n\nSimulating for a1 = {targ1:.3f} ({i+1}/{target_n_points})")
            for j, targ2 in enumerate(targets):
                if j%10 == 0:
                    print(f"\n\nSimulating for a2 = {targ2:.3f} ({j+1}/{target_n_points})")
                for k, targ3 in enumerate(targets):
                    if k%10 == 0:
                        print(f"Simulating for a3 = {targ3:.3f} ({k+1}/{target_n_points})")
                    params = {'a1': targ1, 'a2': targ2, 'a3': targ3}
                    model = SIRS(config_path=config_path, **params)

                    equilibrium = model.find_equilibrium(initial_guess=initial_guess)
                    stability_calculator = EigenStability(model=model, equilibrium=equilibrium)
                    max_real_eigenvalue = stability_calculator.compute()

                    if max_real_eigenvalue > 0:
                        print(f"HOPF BIFURCATION FOUND: a1={targ1:.3f}, a2={targ2:.3f}, a3={targ3:.3f} | Max Real Eig: {max_real_eigenvalue:.6e}")
                    rhs_3d[i, j, k] = -max_real_eigenvalue

        fig = None
        for combo_idx in [['a1', 'a2'], ['a1', 'a3'], ['a2', 'a3']]:

            missing_axis = [idx for idx, target in enumerate(all_targets) if target not in combo_idx][0]

            # Using np.min to visualize worst-case stability over the hidden dimension.
            rhs_combo = np.min(rhs_3d, axis=missing_axis)
            rhs_flat = rhs_combo.flatten()

            params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='18', sim_17_idx=combo_idx)}
            fig = Plots(show_cumulative_incidence=config.get('show_cumulative_incidence', False),
                        show_params=config.get('show_params', False),
                        show_legend=config.get('show_legend', False),
                        show_title=config.get('show_title', False),
                        save_figures=True).plot_rh(
                solution=rhs_flat,
                x_span=target_span,
                y_span=target_span,
                n_points=target_n_points,
                **params
            )

        return fig


    def simulation_19(self, config_path: str) -> plt.Figure | None:
        """
        Simulation 19: Solve the Delay Equation to spot frequencies and periods of oscillations for Delayed SIRS.
        """
        config = self._load_yaml(config_path=config_path)

        # Base analytical estimate from standard SIR endemic state to feed to fsolve
        r0 = config.get('r0', 2.5)
        mu = config.get('mu', 1/80/365)
        theta = config.get('theta', 1/365)
        gamma = config.get('gamma', 1/7)
        S_guess = 1 / r0
        I_guess = (mu + theta) * (1 - S_guess) / (mu + gamma + theta)
        R_guess = 1 - S_guess - I_guess
        initial_guess = [I_guess, R_guess, I_guess]

        # Model run
        model = SIRS(config_path=config_path)
        equilibrium = model.find_equilibrium(initial_guess=initial_guess)
        frequencies, periods = DelayStability(model=model, equilibrium=equilibrium).compute()
        # solutions = model.simulate(t_span=[0, 20000], n_points=20000, initial_conditions=[0.001, 0., 0.])
        # frequencies, periods = DelayStability(model=model, equilibrium=solutions[:, -1]).compute()

        print(f"Frequencies:\t\t{frequencies}\nPeriods: \t\t{periods}")

        return





if __name__ == '__main__':
    parser = argparse.ArgumentParser("Behavioural SIRS")
    parser.add_argument("--simulation", type=int, default=0)
    parser.add_argument("--config", default="../config/config_0.yaml")
    args = parser.parse_args()

    config_path = args.config if args.config else "../config/config_0.yaml"
    simulation = args.simulation if args.simulation is not None else 0

    method = getattr(Simulations(), f"simulation_{simulation}", None)
    assert method is not None, f"Simulation {simulation} not found. Please choose a valid simulation number."
    method(config_path=config_path)