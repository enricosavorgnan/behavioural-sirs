# Enrico Savorgnan
# University of Trieste, Italy, 2025
# A new model in “Behavioral Epidemiology of Infectious Diseases”: SIRS model with social distancing

import numpy as np
import matplotlib.pyplot as plt

from code.core.old.models import *

"""
The aim of this file is to provide the utils methods used by simulations.py and in jupyter file 'simulations.ipynb' to study the different dynamics of several behavioural SIRS models.
"""


class Plots:
    """
    This class provides methods to plot the results of the simulations.
    """

    def __init__(self,
                 show_cumulative_incidence : bool = False,
                 show_params : bool = False,
                 show_title : bool = False,
                 show_legend : bool = False,
                 save_figures : bool = False,
                 ):

        self.show_cumulative_incidence = show_cumulative_incidence
        self.show_params = show_params
        self.show_title = show_title
        self.show_legend = show_legend
        self.save_figures = save_figures


    def _manage_plot_settings(self, fig : plt.Figure, solution : list[np.ndarray] | np.ndarray, t_span : list[float | int], **kwargs):
        """
        Add fancy stuff to plots.

        Parameters
        ----------
        fig : plt.Figure
            The figure to be modified.
        solution : list[np.ndarray] | np.ndarray
            Array or list of array containing dynamics of a class during time.
            All the series in the parameter will be plotted.
        t_span : list[float | int]
            List containing the start and end time of the simulation.
            Used for plotting the x-axis.
        **kwargs
            Additional parameters to be used for plotting many details.
            - model : SIRS | None, default None
                An instance of the SIRS model.
                Used for plotting title and model parameters
            - title : str
                The title of the plot.
                Must be provided if show_title is set to True
            - xlabel : str
                What to print into xlabel ticks
            - ylabel  :str
                What to print into ylabel ticks
            - image_path : str
                image_path where to store the image.
                Must be provided if save_figures is set to True
        """
        ax = fig.gca()

        if self.show_title:
            ax.set_title(kwargs['title'] if 'title' in kwargs else 'SIRS model simulation')

        if self.show_params:
            # print params as subtitle
            model = kwargs['model'] if 'model' in kwargs else None
            if model is not None:
                R0 = round(model.r0, 3) if hasattr(model, 'r0') else None
                theta = round(model.theta, 3) if hasattr(model, 'theta') else None
                mu = round(model.mu, 3) if hasattr(model, 'mu') else None
                gamma = round(model.gamma, 3) if hasattr(model, 'gamma') else None
                a1 = round(model.a1, 3) if hasattr(model, 'a1') else None
                a2 = round(model.a2, 3) if hasattr(model, 'a2') else None
                k1 = round(model.k1, 3) if hasattr(model, 'k1') else None
                k2 = round(model.k2, 3) if hasattr(model, 'k2') else None
                delta = round(model.delta, 3) if hasattr(model, 'delta') else None
                omega = round(model.omega, 3) if hasattr(model, 'omega') else None

                params_text = f"Parameters: R₀={R0}, θ={theta}"
                if mu is not None:
                    params_text += f", mu={mu}"
                if gamma is not None:
                    params_text += f", k={gamma}"
                if a1 is not None:
                    params_text += f", a1={a1}"
                if a2 is not None:
                    params_text += f", a2={a2}"
                if k1 is not None:
                    params_text += f", k={k1}"
                if k2 is not None:
                    params_text += f", k={k2}"
                if delta is not None:
                    params_text += f", k={delta}"
                if omega is not None:
                    params_text += f", k={omega}"

                ax.text(0.5, -0.15, params_text, ha='center', va='center', transform=ax.transAxes)

        if self.show_cumulative_incidence:
            model = kwargs['model'] if 'model' in kwargs else None
            if isinstance(model, SIRS):
                ci = model.cumulative_incidence(solution, t_span)
                ax.text(0.5, -0.25, f"Cumulative incidence: {ci:.4f}", ha='center', va='center', transform=ax.transAxes)

        if self.show_legend:
            legend_labels = kwargs.get('legend', [])
            print(f'legend:labels: {legend_labels}')
            axes = fig.get_axes()

            for ax in axes:
                lines = ax.get_lines()
                if lines and legend_labels:
                    for line, label in zip(lines, legend_labels):
                        line.set_label(label)
                    # loc='best' sometimes fails if the plot is crowded
                    ax.legend(loc='upper right', fontsize='small')

        if self.save_figures:
            plt.savefig(kwargs.get('image_path', './img/img1.png'), dpi=450, format='pdf')


        return


    def plot_simulation(self, solution : list[np.ndarray] | np.ndarray, t_span : list[float | int], n_points : int, **kwargs) -> plt.Figure:
        """
        Plot the results of a single simulation of a SIRS model.

        Parameters
        ----------
        solution : list[np.ndarray] | np.ndarray
            Array or list of array containing dynamics of a class during time.
            All the series in the parameter will be plotted.
        t_span : list[float | int]
            List containing the start and end time of the simulation.
            Used for plotting the x-axis.
        n_points : int
            Number of points where to evaluate the functions.
        **kwargs
            Additional parameters to be used for plotting many details.
            See self._manage_plot_settings for more details
        """
        t = np.linspace(t_span[0], t_span[1], n_points)

        plt.rcParams['text.usetex'] = True
        plt.figure(figsize=(5, 5))
        plt.xlabel(r'$t$')
        plt.ylabel(r'$f(t)$')
        # plt.ylim(-0.05, 1.05)
        plt.grid(linestyle='dotted')

        for ts in solution:
            try:
                plt.plot(t, ts[: n_points])
            except IndexError:
                plt.plot(t, solution)

        self._manage_plot_settings(plt.gcf(), solution, t_span, **kwargs)
        plt.tight_layout()

        return plt.gcf()


    def plot_simulations(self, solutions: list[np.ndarray], t_span: list[float | int], n_points : int, **kwargs) -> plt.Figure:
        """
        Plots results in a grid where rows represent unique R0 values
        and columns represent unique theta values.

        Parameters
        ----------
        solutions : list[np.ndarray]
            List of arrays containing the dynamics of a class during time for each simulation.
        t_span : list[float | int]
            Start and end time for the x-axis.
        n_points : int
            Number of points where to evaluate the functions.
        """
        # 1. Extract unique parameters to define grid dimensions
        n_rows = len(solutions) // 2
        n_cols = len(solutions) // 2

        plt.rcParams['text.usetex'] = True
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                 sharex=True, sharey=True, squeeze=False)

        t = np.linspace(t_span[0], t_span[1], n_points)

        # 2. Map results to the grid
        for i, solution in enumerate(solutions):

            # Find index in grid
            row_idx = i // n_cols
            col_idx = i % n_cols
            ax = axes[row_idx, col_idx]

            # Plot each trajectory in the solution (S, I, R, etc.)
            for j, ts in enumerate(solution):
                ax.plot(t, ts[:n_points])


            ax.grid(linestyle='dotted')
            # ax.set_ylim(-0.05, 1.05)

        # Global labels
        fig.supxlabel(r'$t$')
        fig.supylabel(r'$f(t)$')

        plt.tight_layout()

        self._manage_plot_settings(fig, solutions[0], t_span, **kwargs)  # Pass first solution for settings

        return fig

    def plot_memory(self, solutions, t_span, n_points, **kwargs) -> plt.Figure:
        """
        Specific wrapper that prepares memory-specific labels/styles
        before calling the general plotting logic.
        """
        # Define specific styles for memory vs infectious
        kwargs.setdefault('title', 'Memory Compartment Analysis')

        # Pass only I, M1, M2
        memory_solutions = [sol[[0, 2, 3]] for sol in solutions]

        return self.plot_simulations(memory_solutions, t_span, n_points, **kwargs)