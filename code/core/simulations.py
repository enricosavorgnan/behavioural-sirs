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
            'sirs_zero_layer_incidence' : '0LI',
            'sirs_one_layer_incidence' : '1LI',
            'sirs_two_layer_incidence' : '2LI',
            'sirs_two_layer_one_memory' : '2L1M',
            'sirs_two_layer_incidence_one_memory' : '2LI1M'
        }
        if type(model_types) == str:
            return match_type[model_types]
        elif type(model_types) == list:
            return [match_type[mod_type] for mod_type in model_types]
        return None



    def _retrieve_img_path(self, config_path : str, n_simulation : int | str) -> str:
        """
        Retrieve Image Path given the configuration file and the number of simulation
        """
        config = self._load_yaml(config_path)

        model_type = config.get('model_type', 'sirs')
        model_type = self._retrieve_model_string(model_type)

        r0 = config.get('r0', 2.5)
        theta = round(config.get('theta', 1/365), 3) if type(config.get('theta', 1/365)) == float else [round(theta, 3) for theta in config.get('theta', 1/365)]
        alpha = round(config.get('alpha1', 50.), 3) if type(config.get('alpha1', 50.)) == float else [round(alpha, 3) for alpha in config.get('alpha1', 50.)]
        k = round(config.get('k1', 1), 3) if type(config.get('k1', 1)) == float else [round(k, 3) for k in config.get('k1', 1)]
        a1 = round(config.get('a1', 1/30), 3) if type(config.get('a1', 1/30)) == float else [round(a1, 3) for a1 in config.get('a1', 1/30)]
        a2 = round(config.get('a2', 1/90), 3) if type(config.get('a2', 1/90)) == float else [round(a2, 3) for a2 in config.get('a2', 1/90)]

        img_folder = f'../img/simulation_{n_simulation}/'
        img_path = f'model_{model_type}_r0_{r0}_theta_{theta}_k_{k}_a1_{a1}_a2_{a2}_alpha_{alpha}_time_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        return img_folder + img_path



    def _retrieve_list_img_path(self, config_path : str, n_simulation : int | str, n_imgs : int, curr_i : int):
        """
        Retrieve the image path if the image is member of a list of images
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
        plot_n_points = (n_points * (plot_t_span[1]-plot_t_span[0])) // (t_span[1]-t_span[0]) + 1

        model = SIRS(config_path=config_path)
        solution = model.simulate(t_span=t_span,
                                  initial_conditions=config.get('initial_conditions', [0.99, 0.01, 0]),
                                  n_points=config.get('n_points', 20000))
        S = 1. - solution[0] - solution[1]
        solution = np.vstack((S, solution))

        params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='00')}
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
        It is like Simulation 1, but with additional focus on differnt models.
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
        Simulation 9: different alphas, plot Ie
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
            solutions.append(solution[0])

        # Plot results
        params = {'image_path': self._retrieve_img_path(config_path=config_path, n_simulation='12')}
        fig = Plots(show_cumulative_incidence = config.get('show_cumulative_incidence', False),
                    show_params = config.get('show_params', False),
                    show_title = config.get('show_title', False),
                    save_figures = True).plot_simulation(solution=solutions, t_span=plot_t_span, n_points=plot_n_points, **params)
        return fig





if __name__ == '__main__':
    Simulations().simulation_8(config_path ='../config/config_8.yaml')
