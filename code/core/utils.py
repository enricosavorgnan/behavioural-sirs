# Enrico Savorgnan
# University of Trieste, Italy, 2025
# A new model in “Behavioral Epidemiology of Infectious Diseases”: SIRS model with social distancing

from importlib.resources import path
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

from code.core.models import *

"""
The aim of this file is to provide the utils methods used by simulations.py and in jupyter file 'simulations.ipynb' to study the different dynamics of several behavioural SIRS models.
"""


class Plots:
    """
    This class provides methods to plot the results of the simulations.
    """

    def __init__(self, show_cumulative_incidence : bool =False, show_params : bool = False, save_figures : bool = False, path_figures : str = 'figures'):
        self.show_cumulative_incidence = show_cumulative_incidence
        self.show_params = show_params
        self.save_figures = save_figures
        self.path_figures = path_figures

    
    def save_figure(self, fig, filename):
        """
        Save the figure to a file.
        :param fig:       plotly figure object
        :param filename:  name of the file to save the figure to
        """
        fig.write_image(f"{self.path_figures}/{filename}", scale=2, width=1500, height=800, format='png')
        print(f"Figure saved to {self.path_figures}/{filename}")
        return


    def plot_infectious(self, t, I, model_name : str | None = None):
        """
        Plot the infectious compartment over time.
        :param t:           time points (numpy array)
        :param I:           infectious compartment values (numpy array)
        :param model_name:  name of the model to use in the plot title
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Infectious'))
        fig.update_layout(title=f'Infectious compartment over time for {model_name}' if model_name else 'Infectious compartment over time',
                          xaxis_title='Time (days)',
                          yaxis_title='Proportion of population',
                          template='plotly_white',
                          width=1000, height=600,
                          )
        fig.show()

        if self.save_figures and model_name:
            filename = f"{model_name}_infectious.png"
            self.save_figure(fig, filename)
        return
    

    def plot_simulation(self, model, solution, params):
        """
        Plot the results of a single simulation of a SIRS model.
        """
        t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)
        I = solution[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Infectious'))
        fig.update_layout(title=f'Infectious compartment over time for {name_model(model)}',
                          xaxis_title='Time (days)',
                            yaxis_title='Proportion of population',
                            template='plotly_white',
                            width=1000, height=600,
                          )
        fig.show()

        if self.save_figures:
            filename = f"name_{name_model(model)}_beta_{params['beta']}_gamma_{params['gamma']}_simulation.png"
            self.save_figure(fig, filename)
        return
    
    
    def plot_many_simulations(self, models, solutions, params):
        """
        Plot the results of multiple simulations of different SIRS models in a single plot.
        """

        # Defining the start and horizon for the simulation
        start = params['start_simulation'] if 'start_simulation' in params else 0
        horizon = params['end_simulation'] if 'end_simulation' in params else 20000
        colors = ['blue', 'red', 'green', 'orange']

        # Define the figure
        fig = go.Figure()
        for i, (model, sol, col) in enumerate(zip(models, solutions, colors), start=1):
            print(f'\n\nSimulating {name_model(model)}')
            t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)[start:horizon]
            I = sol[0][start:horizon]

            fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name=name_model(model), line=dict(color=col)))

        fig.update_layout(title='Infectious compartments of different models',
                        xaxis_title='Time (days)',
                        yaxis_title='Proportion of population',
                        legend_title='Models',
                        template='plotly_white',
                        width=1000, height=600,
                        )

        # adding all the cumulative incidence as an annotation
        if self.show_cumulative_incidence:
            fig.add_annotation(
                text='Cumulative incidence',
                xref="paper", yref="paper",
                x=0.5, y=0.9,
                showarrow=False,
                font=dict(size=16),
            )

            for i, (model, sol) in enumerate(zip(models, solutions), start=1):
                cumulative_incidence = calculate_cumulative_incidence(sol, params)
                fig.add_annotation(
                    text=f'{name_model(model)}: {cumulative_incidence:.4f}',
                    xref="paper", yref="paper",
                    x=0.5, y=0.9 - i * 0.1,
                    showarrow=False,
                    font=dict(size=16),
                )

        # adding the parameters as annotation
        if self.show_params:
            text = f"Parameters: R₀={round(params['beta'] / (params['mu'] + params['gamma']), 3)}, θ={round(params['theta'], 3)}"
            if 'a1' in params:
                text += f", a1={round(params['a1'],3)}"
            if 'a2' in params:
                text += f", a2={round(params['a2'], 3)}"
            if 'k' in params:
                text += f", k={round(params['k'], 3)}"
            fig.add_annotation(
                text=text,
                xref="paper", yref="paper",
                x=0.01, y=1.1,
                showarrow=False,
                font=dict(size=16),
            )

        fig.show()

        if self.save_figures:
            filename = f"comparison_simulation.png"
            self.save_figure(fig, self.path_figures, filename)
        return
    

    def plot_memory(self, models, solutions, params=None):
        """
        This method only plots the memory compartments and the infectious class of the different models.
        The purpose is to compare the behaviour of memory while changing parameters.
        """
        # Defining the start and horizon for the simulation
        start = params['start_simulation'] if params and 'start_simulation' in params else 0
        horizon = params['horizon'] if params and 'horizon' in params else 20000

        colors = ['blue', 'orange', 'green', 'red']

        fig = go.Figure()

        for i, (model, sol, col) in enumerate(zip(models, solutions, colors), start=1):
            t = np.linspace(start, horizon, 20000)
            I = sol[0][start:horizon]
            M1 = sol[2][start:horizon] if len(sol) > 2 else None
            M2 = sol[3][start:horizon] if len(sol) > 3 else None

            if M1 is not None and M2 is not None:
                fig.add_trace(go.Scatter(x=t, y=M1, mode='lines', name=f'{name_model(model)} 1st Memory Layer', line=dict(color=col)))
                fig.add_trace(go.Scatter(x=t, y=M2, mode='lines', name=f'{name_model(model)} 2nd Memory Layer', line=dict(color=col)))
            elif M1 is not None:
                fig.add_trace(go.Scatter
                    (x=t, y=M1, mode='lines', name=f'{name_model(model)} 1st Memory Layer', line=dict(color=col)))
            else:
                pass

            # Plotting the infectious compartment
            fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name=f'{name_model(model)} Infectious', line=dict(color=col, dash='dash')))


        fig.update_layout(title='Memory compartments of different models',
                        xaxis_title='Time (days)',
                        yaxis_title='Proportion of population',
                        legend_title='Models',
                        template='plotly_white',
                        width=1000, height=600,
                        )
        fig.show()

        if self.save_figures:
            filename = f"memory_compartments.png"
            save_figure(fig, self.path_figures, filename)

        return fig
    




def simulate(model, params):
    """
    Simulate the SIRS model with given parameters.
    :param model:                     - the name of the model to simulate
    :param params:      β, γ, μ, θ, a1, a2, k, t_span, δ   (dict)      - parameters for the model
    :return:                                                        - the solution of the ODEs as a tuple of numpy arrays
    """

    # Extract parameters
    beta_0 = params['beta']
    gamma = params['gamma']
    mu = params['mu']
    theta = params['theta']
    a1 = params.get('a1', 0)
    a2 = params.get('a2', 0)
    k = params.get('k', 0)
    delta = params.get('delta', 0)
    omega = params.get('omega', 2 * np.pi / 365) if delta != 0 else 0

    # Time span for the simulation
    t_span = params['t_span']
    t = np.linspace(t_span[0], t_span[1], 20000)

    # Initial conditions
    I0 = .001
    R0 = 0
    M0 = 0

    # Choose the model, set initial conditions accordingly, and solve the ODEs
    if model == sirs_zero_layer or model == sirs_one_layer or model == sirs_one_layer_incidence:
        # These models required one-layer memory
        X0 = [I0, R0, M0]
        if delta == 0 or None:
            beta_one = beta_one_layer(beta_0, k)
            sol = solve_ivp(model, t_span, X0, args=(beta_one, gamma, mu, theta, a1), dense_output=True)
        else:
            beta_one = beta_one_layer_seasonal(beta_0, k, delta, omega)
            sol = solve_ivp(model, t_span, X0, args=(beta_one, gamma, mu, theta, a1, k, delta), dense_output=True)

    elif model == sirs_two_layer or model == sirs_two_layer_incidence:
        # These models required two-layer memory
        X0 = [I0, R0, M0, M0]
        if delta == 0 or None:
            beta_two = beta_two_layer(beta_0, k, k)
            sol = solve_ivp(model, t_span, X0, args=(beta_two, gamma, mu, theta, a1, a2, k), dense_output=True)
        else:
            beta_two = beta_two_layer_seasonal(beta_0, k, k, delta, omega)
            sol = solve_ivp(model, t_span, X0, args=(beta_two, gamma, mu, theta, a1, a2, k, delta), dense_output=True)


    else:
        # Standard SIRS model without memory
        X0 = [I0, R0]
        sol = solve_ivp(model, t_span, X0, args=(beta_0, gamma, mu, theta), dense_output=True)

    # Solve ODE
    solution = sol.sol(t)

    return solution



def calculate_cumulative_incidence(solution, params):
    """
    Calculate the cumulative incidence from the solution of the SIRS model.
    """

    # Extracting parameters
    t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)
    I = solution[0]
    R = solution[1]
    S = 1 - I - R
    M1 = solution[2] if len(solution) > 2 else None
    M2 = solution[3] if len(solution) > 3 else None

    # Calculating cumulative incidence
    beta_1 = beta_one_layer(params['beta'], params['k'])
    beta_2 = beta_two_layer(params['beta'], params['k'], params['k'])
    
    if M1 is not None and M2 is not None:
        incidence = [beta_2(m1, m2) * (1 - r - i) * i for i, r, m1, m2 in zip(I, R, M1, M2)]
    elif M1 is not None:
        incidence = [beta_1(m1) * (1 - r - i) * i for i, r, m1 in zip(I, R, M1)]
    else:
        incidence = [params['beta'] * (1 - r - i) * i for i, r in zip(I, R)]

    # Integrating the incidence over time to get cumulative incidence, using the trapezoidal rule
    cumulative_incidence = np.trapezoid(incidence, t)

    return cumulative_incidence



def name_model(model):
    """
    Return the name of the model as a string.
    """
    if model == sirs:
        return 'SIRS'
    elif model == sirs_one_layer:
        return 'SIRS 1-layer'
    elif model == sirs_two_layer:
        return 'SIRS 2-layer'
    elif model == sirs_zero_layer:
        return 'SIRS 0-layer'
    elif model == sirs_one_layer_incidence:
        return 'SIRS 1-layer incidence'
    elif model == sirs_two_layer_incidence:
        return 'SIRS 2-layer incidence'
    else:
        raise ValueError("Unknown model type. Please provide a valid SIRS model.")






def plot_many_simulations(models, solutions, params, show_cumulative_incidence=False, show_params=True):
    """
    Plot the results of multiple simulations of different SIRS models in a single plot.
    """

    # Defining the start and horizon for the simulation
    start = params['start_simulation'] if 'start_simulation' in params else 0
    horizon = params['end_simulation'] if 'end_simulation' in params else 20000
    colors = ['blue', 'red', 'green', 'orange']

    # Define the figure
    fig = go.Figure()
    for i, (model, sol, col) in enumerate(zip(models, solutions, colors), start=1):
        print(f'\n\nSimulating {name_model(model)}')
        t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)[start:horizon]
        I = sol[0][start:horizon]

        fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name=name_model(model), line=dict(color=col)))

    fig.update_layout(title='Infectious compartments of different models',
                      xaxis_title='Time (days)',
                      yaxis_title='Proportion of population',
                      legend_title='Models',
                      template='plotly_white',
                      width=1000, height=600,
                      )

    # adding all the cumulative incidence as an annotation
    if show_cumulative_incidence:
        fig.add_annotation(
            text='Cumulative incidence',
            xref="paper", yref="paper",
            x=0.5, y=0.9,
            showarrow=False,
            font=dict(size=16),
        )

        for i, (model, sol) in enumerate(zip(models, solutions), start=1):
            cumulative_incidence = calculate_cumulative_incidence(sol, params)
            fig.add_annotation(
                text=f'{name_model(model)}: {cumulative_incidence:.4f}',
                xref="paper", yref="paper",
                x=0.5, y=0.9 - i * 0.1,
                showarrow=False,
                font=dict(size=16),
            )

    # adding the parameters as annotation
    if show_params:
        text = f"Parameters: R₀={round(params['beta'] / (params['mu'] + params['gamma']), 3)}, θ={round(params['theta'], 3)}"
        if 'a1' in params:
            text += f", a1={round(params['a1'],3)}"
        if 'a2' in params:
            text += f", a2={round(params['a2'], 3)}"
        if 'k' in params:
            text += f", k={round(params['k'], 3)}"
        fig.add_annotation(
            text=text,
            xref="paper", yref="paper",
            x=0.01, y=1.1,
            showarrow=False,
            font=dict(size=16),
        )

    fig.show()

    return



def plot_memory(models, solutions, params=None):
    """
    This method only plots the memory compartments and the infectious class of the different models.
    The purpose is to compare the behaviour of memory while changing parameters.
    """
    # Defining the start and horizon for the simulation
    start = params['start_simulation'] if params and 'start_simulation' in params else 0
    horizon = params['horizon'] if params and 'horizon' in params else 20000

    colors = ['blue', 'orange', 'green', 'red']

    fig = go.Figure()

    for i, (model, sol, col) in enumerate(zip(models, solutions, colors), start=1):
        t = np.linspace(start, horizon, 20000)
        I = sol[0][start:horizon]
        M1 = sol[2][start:horizon] if len(sol) > 2 else None
        M2 = sol[3][start:horizon] if len(sol) > 3 else None

        if M1 is not None and M2 is not None:
            fig.add_trace(go.Scatter(x=t, y=M1, mode='lines', name=f'{name_model(model)} 1st Memory Layer', line=dict(color=col)))
            fig.add_trace(go.Scatter(x=t, y=M2, mode='lines', name=f'{name_model(model)} 2nd Memory Layer', line=dict(color=col)))
        elif M1 is not None:
            fig.add_trace(go.Scatter
                (x=t, y=M1, mode='lines', name=f'{name_model(model)} 1st Memory Layer', line=dict(color=col)))
        else:
            pass

        # Plotting the infectious compartment
        fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name=f'{name_model(model)} Infectious', line=dict(color=col, dash='dash')))


    fig.update_layout(title='Memory compartments of different models',
                      xaxis_title='Time (days)',
                      yaxis_title='Proportion of population',
                      legend_title='Models',
                      template='plotly_white',
                      width=1000, height=600,
                      )
    fig.show()

    return fig