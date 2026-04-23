# Enrico Savorgnan
# University of Trieste, Italy, 2025
# A new model in “Behavioral Epidemiology of Infectious Diseases”: SIRS model with social distancing

"""
The aim of this file is to provide the main simulation methods used in the thesis and in the jupyter file `simulations.ipynb`.
"""
from code.core.utils import *
from code.core.old.models import *
from plotly.subplots import make_subplots

class Simulations:

    def __init__(self, show_cumulative_incidence=False, save_figures=False, path_figures='figures'):
        self.show_cumulative_incidence = show_cumulative_incidence
        self.save_figures = save_figures
        self.path_figures = path_figures


    def simulation_1_r0_and_theta(self, models, r0_theta_list, params, show_cumulative_incidence=False):
        """
        Simulation 1. of the thesis.
        This method simulates the SIRS models with different R₀ and θ parameters.
        """
        assert len(r0_theta_list) == 4, "You need exactly 4 couples (R₀, θ)"

        start = params['start_simulation'] if 'start_simulation' in params else 0
        horizon = params['horizon'] if 'horizon' in params else 20000
        # print(horizon)

        # Defining subplots titles
        subplot_titles = [
            f"R₀ = {R0:.2f}, θ = {theta:.5f}"
            for R0, theta in r0_theta_list
        ]

        # Create 2D canvas grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=subplot_titles,
            shared_xaxes=False, shared_yaxes=False,
            horizontal_spacing=0.08, vertical_spacing=0.12
        )

        colors = ['blue', 'red', 'green', 'orange']

        # Keep track of the first plot to avoid repeating legends
        first_plot = True

        for idx, (R0, theta) in enumerate(r0_theta_list):
            # Assign row and column based on index
            row = idx // 2 + 1
            col = idx % 2 + 1

            # Prepare parameters for the current simulation
            params = params.copy()
            params['theta'] = theta
            params['beta']  = R0 * (params['mu'] + params['gamma'])

            t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)

            # istantiating the models with the current parameters
            solutions = [SIRS(model_type=m, params=params).simulate(initial_conditions=params['initial_conditions'], t_span=params['t_span']) for m in models]

            # adding traces to the figure
            for _j, (model, sol, color) in enumerate(zip(models, solutions, colors)):
                I = sol[0] # sol[0] = infetti
                fig.add_trace(
                    go.Scatter(
                        x=t[:horizon], y=I[:horizon],
                        mode='lines',
                        name=name_model(model),
                        legendgroup=name_model(model),
                        showlegend=first_plot,
                        line=dict(color=color)
                    ),
                    row=row, col=col
                )

                if self.show_cumulative_incidence:
                    cum_inc = calculate_cumulative_incidence(sol, params)
                    print(f"Cumulative incidence for {name_model(model)} with R₀={round(R0, 3)}, θ={round(theta, 3)}: {cum_inc:.4f}")

                    fig.add_annotation(
                        text=f"Cumulative Incidence for({name_model(model)}):\t {cum_inc:.4f}",
                        xref='x domain', yref='y domain',
                        y=0.7 - 0.05*_j, x=0.5,
                        showarrow=False,
                        font=dict(size=10),
                        row=row, col=col
                    )

            first_plot = False  # after the first plot, we don't need to show the legend again


        # global layout settings
        fig.update_layout(
            title_text="Confronto dei modelli SIRS sui vari scenari (R₀, θ)",
            template='plotly_white',
            width=1200, height=1000,
            legend_title="Modelli",
            margin=dict(t=100, b=50, l=50, r=50)
        )

        # uniform x and y axes titles
        fig.update_xaxes(title_text="Time (days)", row=2, col=1)
        fig.update_xaxes(title_text="Time (days)", row=2, col=2)
        fig.update_yaxes(title_text="Proportion infected", row=1, col=1)
        fig.update_yaxes(title_text="Proportion infected", row=2, col=1)

        fig.show()

        return fig



    def simulation_2_memory(self, models, params):
        """
        Simulation 2. of the thesis.
        This method plots the memory compartments of given SIRS models.
        """

        solutions = []
        for _, model in enumerate(models):
            print(f'Simulating {name_model(model)}\n')
            solution = simulate(model, params)
            solutions.append(solution)

        fig = Plots().plot_memory(models, solutions, params)

        return fig



    def simulation_3_r0_and_theta(self, models, params, r0_theta_list, show_cumulative_incidence=False):
        """
        Simulation 3. of the thesis.
        Same as simulation_01 but with more parameters.
        """

        assert len(r0_theta_list) == 9, "Servono esattamente 9 coppie (R0, θ)"

        horizon = params['horizon'] if 'horizon' in params else 20000

        # Defining subplots titles
        subplot_titles = [
            f"R₀ = {R0:.2f}, θ = {theta:.5f}"
            for R0, theta in r0_theta_list
        ]

        # Create 2D canvas grid
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=subplot_titles,
            shared_xaxes=False, shared_yaxes=False,
            horizontal_spacing=0.08, vertical_spacing=0.12
        )

        colors = ['blue', 'red', 'green', 'orange']

        # Keep track of the first plot to avoid repeating legends
        first_plot = True

        for idx, (R0, theta) in enumerate(r0_theta_list):
            # Assign row and column based on index
            row = idx // 3 + 1
            col = idx % 3 + 1

            # Prepare parameters for the current simulation
            params = params.copy()
            params['theta'] = theta
            params['beta']  = R0 * (params['mu'] + params['gamma'])

            t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)

            # Simulate the models for the current parameters
            solutions = [simulate(m, params) for m in models]

            # Add traces to the figure
            for _j, (model, sol, color) in enumerate(zip(models, solutions, colors)):
                I = sol[0] # sol[0] = infectious
                fig.add_trace(
                    go.Scatter(
                        x=t[:horizon], y=I[:horizon],
                        mode='lines',
                        name=name_model(model),
                        legendgroup=name_model(model),
                        showlegend=first_plot,
                        line=dict(color=color)
                    ),
                    row=row, col=col
                )

                if self.show_cumulative_incidence:
                    cum_inc = calculate_cumulative_incidence(sol, params)
                    print(f"Cumulative incidence for {name_model(model)} with R₀={round(R0, 3)}, θ={round(theta, 3)}: {cum_inc:.4f}")
                    # eventually add cumulative incidence notation to the plot
                    fig.add_annotation(
                        text=f"Cumulative Incidence({name_model(model)}): {cum_inc:.4f}",
                        xref='x domain', yref='y domain',
                        y=0.9 - 0.05*_j, x=0.5,
                        showarrow=False,
                        font=dict(size=10),
                        row=row, col=col
                    )

            first_plot = False  # after the first plot, we don't need to show the legend again

        fig.update_layout(
            title_text="Confronto dei modelli SIRS sui vari scenari (R₀, θ)",
            template='plotly_white',
            width=1200, height=1000,
            legend_title="Modelli",
            margin=dict(t=100, b=50, l=50, r=50)
        )

        # uniform x and y axes titles
        fig.update_xaxes(title_text="Time (days)", row=2, col=1)
        fig.update_xaxes(title_text="Time (days)", row=2, col=2)
        fig.update_yaxes(title_text="Proportion infected", row=1, col=1)
        fig.update_yaxes(title_text="Proportion infected", row=2, col=1)

        fig.show()

        return fig



    def simulation_4_r0_and_theta_with_2_layers(self, models, params, R0s, thetas, a_list, show_cumulative_incidence=False):
        """
        Simulation 4. of the thesis.
        This method is similar to simulation_3b but it plots the SIRS models with 2 layers of memory for different (R₀, θ) scenarios.
        """

        t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)

        horizon = params['horizon'] if 'horizon' in params else 20000

        # Defining subplots titles
        subplot_titles = [
            f'R₀ = {R0:.2f}, θ = {theta:.5f}' for R0, theta in zip(R0s, thetas)
        ]

        # Create 2D canvas grid
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=subplot_titles,
            shared_xaxes=False, shared_yaxes=False,
            horizontal_spacing=0.08, vertical_spacing=0.12
        )

        colors = ['rgba(255, 0, 125, 0.5)', 'rgba(0, 0, 0, 0.5)', 'rgba(0, 125, 255, 0.5)', 'rgba(255, 125, 0, 0.5)']
        # i_colors = ['rgba(0, 0, 255, 1)', 'rgba(0, 0, 255, 0.5)', 'rgba(0, 0, 255, 0.2)', 'rgba(0, 56, 255, 0.5)']              # colors for the infectious class

        for idx, (a1, a2) in enumerate(a_list):
            params = params.copy()
            params['a1'] = a1
            params['a2'] = a2

            first_plot = True

            for _j, (R0, theta) in enumerate(zip(R0s, thetas)):
                # Prepare parameters for the current simulation
                params['theta'] = theta
                params['beta']  = R0 * (params['mu'] + params['gamma'])


                solutions = [simulate(m, params) for m in models]

                color = colors[idx]
                # i_color = i_colors[idx]
                x = 0.5
                y = 0.9 - 0.05*idx

                # Assign row and column based on index
                row = 1
                col = _j + 1

                # I = solutions[0][0]
                M1 = solutions[0][2]
                M2 = solutions[0][3]

                fig.add_trace(
                    go.Scatter(
                        x=t[:horizon], y=M1[:horizon] + M2[:horizon],
                        mode='lines',
                        name=f'a₁: {a1:.4f}, a₂: {a2:.4f}',
                        legendgroup=name_model(models[0]),
                        showlegend=first_plot,
                        line=dict(color=color)
                    ),
                    row=row, col=col
                )

                if self.show_cumulative_incidence:
                    cum_inc = calculate_cumulative_incidence(solutions[0], params)
                    print(f"Cumulative incidence for {name_model(models[0])} with R₀={round(R0, 3)}, θ={round(theta, 3)}: {cum_inc:.4f}")
                    fig.add_annotation(
                        text=f"CI({name_model(models[0])}): {cum_inc:.4f}",
                        xref='x domain', yref='y domain',
                        y=y, x=0.5,
                        showarrow=False,
                        font=dict(size=10),
                        row=row, col=col
                    )
                first_plot = False


        fig.update_layout(
            title_text="Confronto del modello SIRS con 2 layer di memoria sui vari scenari (a₁, a₂)",
            template='plotly_white',
            width=1500, height=800,
            legend_title="Modelli",
            margin=dict(t=100, b=50, l=50, r=50)
        )
        fig.update_xaxes(title_text="Time (days)", row=2, col=1)
        fig.update_xaxes(title_text="Time (days)", row=2, col=2)
        fig.update_yaxes(title_text="Proportion infected", row=1, col=1)
        fig.update_yaxes(title_text="Proportion infected", row=2, col=1)
        # share the y axes
        # fig.update_yaxes(matches='y')

        fig.show()

        return fig



    def simulation_5_k(self, models, params, k, show_cumulative_incidence=False):
        """
        Simulation 5. of the thesis.
        This method simulates the SIRS models with different k parameters.
        """

        t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)
        horizon = params['horizon'] if 'horizon' in params else 20000

        fig = go.Figure()

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for idx, k_val in enumerate(k):
            params = params.copy()
            params['k'] = k_val

            # Simulate the models for the current parameters
            solutions = [simulate(m, params) for m in models]
            solution_data = solutions[0]

            I = solution_data[0]

            fig.add_trace(
                go.Scatter(
                    x=t[:horizon],
                    y=I[:horizon],
                    mode='lines',
                    name=f'k = {k_val}',
                    line=dict(color=colors[idx % len(colors)])
                )
            )

            if self.show_cumulative_incidence:
                cum_inc = calculate_cumulative_incidence(solution_data, params)
                print(f"Cumulative incidence for {name_model(models[0])} with k={k_val}: {cum_inc:.4f}")
                if show_cumulative_incidence:
                    # eventually add cumulative incidence notation to the plot
                    fig.add_annotation(
                        text=f"CI({name_model(models[0])}): {cum_inc:.4f}",
                        xref='x domain', yref='y domain',
                        y=0.9 - 0.05*idx, x=0.5,
                        showarrow=False,
                        font=dict(size=10)
                    )

        # 4. Aggiornare il layout della figura singola
        fig.update_layout(
            title_text="Confronto del modello SIRS con 2 layer di memoria sui vari scenari (k) - Numero di infetti",
            xaxis_title="Time (days)",
            yaxis_title="Proportion infected",
            template='plotly_white',
            width=1200,
            height=600,
            legend_title_text="Valore di k" #
        )

        fig.show()

        return fig



    def simulation_6_r0_and_theta_prevalence_vs_incidence(self, models, params, R0s, thetas, show_cumulative_incidence=False):
        """
        Simulation 6. of the thesis.
        This method simulates prevalence-based and incidence-based SIRS models with different (R₀, θ) scenarios.
        """

        t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)
        start = 0.
        horizon = 20000

        # Defining subplots titles
        subplot_titles = [
            f'R₀ = {R0:.2f}, θ = {theta:.5f}' for R0, theta in zip(R0s, thetas)
        ]

        # Create 2D canvas grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=subplot_titles,
            shared_xaxes=False, shared_yaxes=False,
            horizontal_spacing=0.08, vertical_spacing=0.12
        )

        colors = ['rgba(255, 0, 125, 0.5)', 'rgba(0, 0, 0, 0.5)', 'rgba(0, 125, 255, 0.5)', 'rgba(255, 125, 0, 0.5)']

        first_plot = True

        for idx, (R0, theta) in enumerate(zip(R0s, thetas)):
            params = params.copy()
            params['theta'] = theta
            params['beta']  = R0 * (params['mu'] + params['gamma'])


            # Simulate the models for the current parameters
            solutions = [simulate(m, params) for m in models]

            # notes positions
            x = 0.5
            # Assign row and column based on index
            row = idx // 2 + 1
            col = idx % 2 + 1


            for _j, (model, sol, color) in enumerate(zip(models, solutions, colors)):
                y = 0.9 - 0.05*_j

                I = sol[0] # sol[0] = infetti
                fig.add_trace(
                    go.Scatter(
                        x=t[:horizon], y=I[:horizon],
                        mode='lines',
                        #name=name_model(model),
                        #legendgroup=name_model(model),
                        # showlegend=first_plot,
                        showlegend = False,
                        line=dict(color=color)
                    ),
                    row=row, col=col
                )
                cum_inc = calculate_cumulative_incidence(sol, params)
                print(f"Cumulative incidence for {name_model(model)} with R₀={round(R0, 3)}, θ={round(theta, 3)}: {cum_inc:.4f}")
                if show_cumulative_incidence:
                    # eventually add cumulative incidence notation to the plot
                    fig.add_annotation(
                        text=f"CI({name_model(model)}): {cum_inc:.4f}",
                        xref='x domain', yref='y domain',
                        y=y, x=0.5,
                        showarrow=False,
                        font=dict(size=10),
                        row=row, col=col
                    )

            first_plot = False  # dopo la prima cella, non ripetere le legende

        fig.update_layout(
            # title_text="Confronto dei modelli SIRS con memoria prevalence-based e incidence-based per diversi valori (R₀, θ)",
            title_text=" ",
            template='plotly_white',
            width=1500, height=800,
            legend_title="Modelli",
            margin=dict(t=100, b=50, l=50, r=50)
        )

        fig.update_xaxes(title_text="Time (days)", row=2, col=1)
        fig.update_xaxes(title_text="Time (days)", row=2, col=2)
        fig.update_yaxes(title_text="Proportion infected", row=1, col=1)
        fig.update_yaxes(title_text="Proportion infected", row=2, col=1)

        fig.show()

        return fig



    def simulation_7_a1_and_a2_prevalence_vs_incidence(self, models, params, a1s, a2s, show_cumulative_incidence=False):
        """
        Simulation 7. of the thesis.
        This method simulates prevalence-based and incidence-based SIRS models with different couples (a₁, a₂).
        """

        t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)
        start = 0.
        horizon = 20000

        # Defining subplots titles
        subplot_titles = [
            f'a₁ = {a1:.4f}, a₂ = {a2:.4f}' for a1, a2 in zip(a1s, a2s)
        ]

        # Create 2D canvas grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=subplot_titles,
            shared_xaxes=False, shared_yaxes=False,
            horizontal_spacing=0.08, vertical_spacing=0.12
        )

        colors = ['rgba(255, 0, 125, 0.5)', 'rgba(0, 0, 0, 0.5)', 'rgba(0, 125, 255, 0.5)', 'rgba(255, 125, 0, 0.5)']

        first_plot = True

        for idx, (a1, a2) in enumerate(zip(a1s, a2s)):
            params = params.copy()
            params['a1'] = a1
            params['a2'] = a2

            # Simulate the models for the current parameters
            solutions = [simulate(m, params) for m in models]

            # notes positions
            x = 0.5
            # Assign row and column based on index
            row = idx // 2 + 1
            col = idx % 2 + 1

            for _j, (model, sol, color) in enumerate(zip(models, solutions, colors)):
                y = 0.9 - 0.05*_j

                I = sol[0] # sol[0] = infetti
                fig.add_trace(
                    go.Scatter(
                        x=t[:horizon], y=I[:horizon],
                        mode='lines',
                        name=name_model(model),
                        legendgroup=name_model(model),
                        showlegend=first_plot,
                        line=dict(color=color)
                    ),
                    row=row, col=col
                )
                cum_inc = calculate_cumulative_incidence(sol, params)
                print(f"Cumulative incidence for {name_model(model)} with a₁={round(a1, 3)}, a₂={round(a2, 3)}: {cum_inc:.4f}")
                if show_cumulative_incidence:
                    # eventually add cumulative incidence notation to the plot
                    fig.add_annotation(
                        text=f"CI({name_model(model)}): {cum_inc:.4f}",
                        xref='x domain', yref='y domain',
                        y=y, x=0.5,
                        showarrow=False,
                        font=dict(size=10),
                        row=row, col=col
                    )

            first_plot = False  # after the first plot, we don't need to show the legend again

        fig.update_layout(
            title_text="Confronto dei modelli SIRS con memoria prevalence-based e incidence-based per diversi valori (a₁, a₂)",
            template='plotly_white',
            width=1500, height=800,
            legend_title="Modelli",
            margin=dict(t=100, b=50, l=50, r=50)
        )

        fig.update_xaxes(title_text="Time (days)", row=2, col=1)
        fig.update_xaxes(title_text="Time (days)", row=2, col=2)
        fig.update_yaxes(title_text="Proportion infected", row=1, col=1)
        fig.update_yaxes(title_text="Proportion infected", row=2, col=1)

        fig.show()

        return fig



    def simulation_8_k_prevalence_vs_incidence(self, models, params, ks, show_cumulative_incidence=False):
        """
        Simulation 8. of the thesis.
        This method simulates prevalence-based and incidence-based SIRS models with different k parameters.
        """

        t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)
        start = 0.
        horizon = 20000

        # Defining subplots titles
        subplot_titles = [
            f'k = {k_val}' for k_val in ks
        ]

        # Create 2D canvas grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=subplot_titles,
            shared_xaxes=False, shared_yaxes=False,
            horizontal_spacing=0.08, vertical_spacing=0.12
        )

        colors = ['rgba(255, 0, 125, 0.5)', 'rgba(0, 0, 0, 0.5)', 'rgba(0, 125, 255, 0.5)', 'rgba(255, 125, 0, 0.5)']

        first_plot = True

        for idx, k_val in enumerate(ks):
            params = params.copy()
            params['k'] = k_val

            # Simulate the models for the current parameters
            solutions = [simulate(m, params) for m in models]

            # notes positions
            x = 0.5
            # Assign row and column based on index
            row = idx // 2 + 1
            col = idx % 2 + 1

            for _j, (model, sol, color) in enumerate(zip(models, solutions, colors)):
                y = 0.9 - 0.05*_j

                I = sol[0] # sol[0] = infetti
                fig.add_trace(
                    go.Scatter(
                        x=t[:horizon], y=I[:horizon],
                        mode='lines',
                        name=name_model(model),
                        legendgroup=name_model(model),
                        showlegend=first_plot,
                        line=dict(color=color)
                    ),
                    row=row, col=col
                )
                cum_inc = calculate_cumulative_incidence(sol, params)
                print(f"Cumulative incidence for {name_model(model)} with k={round(k_val, 3)}: {cum_inc:.4f}")
                if show_cumulative_incidence:
                    # eventually add cumulative incidence notation to the plot
                    fig.add_annotation(
                        text=f"CI({name_model(model)}): {cum_inc:.4f}",
                        xref='x domain', yref='y domain',
                        y=y, x=0.65,
                        showarrow=False,
                        font=dict(size=10),
                        row=row, col=col
                    )

            first_plot = False  # after the first plot, we don't need to show the legend again

        fig.update_layout(
            title_text="Confronto dei modelli SIRS con memoria prevalence-based e incidence-based per diversi valori (k)",
            template='plotly_white',
            width=1500, height=1000,
            legend_title="Modelli",
            margin=dict(t=100, b=50, l=50, r=50)
        )

        fig.update_xaxes(title_text="Time (days)")
        fig.update_yaxes(title_text="Proportion infected")

        fig.show()

        return fig



    def simulation_9_periodic(self, models, params, deltas, show_cumulative_incidence=False):
        """
        The following simulation will analyze the behaviour of the models with a periodic component

        :param models:
        :param params:
        :param deltas:
        :param show_cumulative_incidence:
        :return:
        """

        t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)
        start = 0.
        horizon = 20000

        # Defining subplots titles
        subplot_titles = [
            f'δ = {delta}' for delta in deltas
        ]

        # Create 2D canvas grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=subplot_titles,
            shared_xaxes=False, shared_yaxes=False,
            horizontal_spacing=0.08, vertical_spacing=0.12
        )

        for delta in deltas:
            params = params.copy()
            params['delta'] = delta

            # Assign row and column based on index
            idx = deltas.index(delta)
            row = idx // 2 + 1
            col = idx % 2 + 1

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            # Simulate the models for the current parameters
            solutions = [simulate(m, params) for m in models]

            for idx, (model, sol) in enumerate(zip(models, solutions)):
                I = sol[0]
                S = 1 - sol[0] - sol[1]

                S_eq = (params['mu'] + params['gamma']) / params['beta']
                I_eq = I[-1]

                fig.add_trace(
                    go.Scatter(
                        x=S[:horizon] / ([S_eq] * horizon),
                        y=I[:horizon] / ([I_eq] * horizon),
                        mode='lines',
                        name=f'{name_model(model)}',
                        line=dict(color=colors[idx % len(colors)])
                    ),
                    row=row, col=col,
                )

        fig.update_layout(
            title_text="Andamento all'equilibrio per diversi modelli, con diversi valori di δ",
            template='plotly_white',
            width=1500, height=1000,
            legend_title="Modelli",
            margin=dict(t=100, b=50, l=50, r=50)
        )

        fig.update_xaxes(title_text="I / Ie")
        fig.update_yaxes(title_text="S / Se")
        fig.show()

        return fig



    def simulation_10_a1_and_a2(self, models, params, a_list, show_cumulative_incidence=False):
        """
        Simulation 10. (previous 3b) of the thesis.
        This method simulates the SIRS models with different a₁ and a₂ parameters.
        """

        horizon = params['horizon'] if 'horizon' in params else 20000

        # Defining subplots titles
        subplot_titles = [
            'SIRS con 1 Layer di Memoria',
            'SIRS con 2 Layer di Memoria'
        ]

        # Create 2D canvas grid
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=subplot_titles,
            shared_xaxes=False, shared_yaxes=False,
            horizontal_spacing=0.08, vertical_spacing=0.12
        )

        colors = ['rgba(255, 0, 125, 0.5)', 'rgba(0, 0, 0, 0.5)', 'rgba(0, 125, 255, 0.5)', 'rgba(255, 125, 0, 0.5)']

        for idx, (a1, a2) in enumerate(a_list):
            params = params.copy()
            params['a1'] = a1
            params['a2'] = a2

            t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)
            solutions = [simulate(m, params) for m in models]

            color = colors[idx]
            # notes positions
            x = 0.5
            y = 0.9 - 0.05*idx

            for _j, (model, sol) in enumerate(zip(models, solutions)):
                row = 1
                if model == sirs_one_layer:
                    col = 1

                    M = sol[2]

                    fig.add_trace(
                        go.Scatter(
                            x=t[:horizon], y=M[:horizon],
                            mode='lines',
                            name=f'a₁={a1:.4f}, a₂={a2:.4f}',
                            legendgroup=name_model(model),
                            showlegend=True,
                            line=dict(color=color)
                        ),
                        row=1, col=col
                    )

                    cum_inc = calculate_cumulative_incidence(sol, params)

                    if show_cumulative_incidence:
                        fig.add_annotation(
                            text=f"CI({name_model(model)}): {cum_inc:.4f}",
                            xref='x domain', yref='y domain',
                            y=y, x=0.5,
                            showarrow=False,
                            font=dict(size=10),
                            row=row, col=col
                        )
                        first_plot_sx = False

                elif model == sirs_two_layer:
                    M1 = sol[2]
                    M2 = sol[3]
                    col = 2

                    fig.add_trace(
                        go.Scatter(
                            x=t[:horizon], y=M1[:horizon] + M2[:horizon],
                            mode='lines',
                            name=f'a₁: {a1:.4f}, a₂: {a2:.4f}',
                            legendgroup=name_model(model),
                            showlegend=False,
                            line=dict(color=color)
                        ),
                        row=row, col=col
                    )
                    cum_inc = calculate_cumulative_incidence(sol, params)
                    if show_cumulative_incidence:
                        fig.add_annotation(
                            text=f"CI({name_model(model)}): {cum_inc:.4f}",
                            xref='x domain', yref='y domain',
                            y=y, x=0.5,
                            showarrow=False,
                            font=dict(size=10),
                            row=row, col=col
                        )

                first_plot_sx = False




        # Layout globale
        fig.update_layout(
            title_text="Confronto di modelli SIRS con memoria per varie coppie (a₁, a₂)",
            template='plotly_white',
            width=1500, height=800,
            legend_title="Coppie (a₁, a₂)",
            margin=dict(t=100, b=50, l=50, r=50)
        )

        fig.update_xaxes(title_text="Time (days)", row=2, col=1)
        fig.update_xaxes(title_text="Time (days)", row=2, col=2)
        fig.update_yaxes(title_text="Proportion infected", row=1, col=1)
        fig.update_yaxes(title_text="Proportion infected", row=2, col=1)
        # share the y axes
        fig.update_yaxes(matches='y')

        fig.show()

        return fig


    def simulation_11_ie_given_alpha(self, params, alpha_range : list):
        """
        Simulation 11. of the thesis.
        This method simulates the behaviour of the Infectious endemic equilibrium as a function of the alpha parameter, which represents the strength of the memory effect on the infectious class.
        """
        R0 = params['R0']
        alpha_start, alpha_end = alpha_range[0], alpha_range[1]
        alpha = np.linspace(alpha_start, alpha_end, 10000)

        I = (R0 - 1) / (R0 * (1 + alpha * (R0 - 1)))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=alpha, y=I,
                mode='lines',
                name='Infectious endemic equilibrium',
                line=dict(color='blue')
            )
        )

        fig.update_layout(
            title_text="Infectious Endemic Equilibrium given α",
            xaxis_title="α",
            yaxis_title="I*",
            template='plotly_white',
            width=1200, height=600,
            legend_title="Infectious Endemic Equilibrium"
        )

        fig.show()
        return fig


    def simulation_12_distance_between_ee(self, params, beta_range : list):
        """
        Simulation 12. of the thesis.
        This method simulates the distance between the endemic equilibrium of the SIRS model with memory with and without the vital dynamics, as a function of β.
        """
        beta_start, beta_end = beta_range[0], beta_range[1]
        beta_values = np.linspace(beta_start, beta_end, 1000)
        R0_vital = beta_values / (params['mu'] + params['gamma'])
        R0_no_vital = beta_values / params['gamma']

        I_ee_with_vital = (R0_vital - 1) / (R0_vital * (1 + params['alpha'] * (R0_vital - 1)))
        I_ee_without_vital = (R0_no_vital - 1) / (R0_no_vital * (1 + params['alpha'] * (R0_no_vital - 1)))

        S_ee_with_vital = 1 / R0_vital
        S_ee_without_vital = 1 / R0_no_vital

        R_ee_with_vital = 1 - S_ee_with_vital - I_ee_with_vital
        R_ee_without_vital = 1 - S_ee_without_vital - I_ee_without_vital

        M_ee_with_vital = I_ee_with_vital
        M_ee_without_vital = I_ee_without_vital


        distance = np.sqrt((S_ee_with_vital - S_ee_without_vital)**2 + (I_ee_with_vital - I_ee_without_vital)**2 + (R_ee_with_vital - R_ee_without_vital)**2 + (M_ee_with_vital - M_ee_without_vital)**2)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=beta_values, y=distance,
                mode='lines',
                name='Distance between endemic equilibria',
                line=dict(color='red')
            )
        )

        fig.update_layout(
            title_text="Distance between endemic equilibria given β",
            xaxis_title="β",
            yaxis_title="Distance",
            template='plotly_white',
            width=1200, height=600,
            legend_title="Distance between endemic equilibria"
        )

        fig.show()
        return fig


    def simulation_13_impact_r0(self, params, r0_list : list):
        """
        Simulation 13. of the thesis.
        This method simulates the impact of R₀ on the infectious endemic equilibrium, for both the SIRS model with memory and the classic SIRS model without memory.
        """
        def calculate_I_ee(r0, x, params):
            q = params['gamma'] / (params['mu'] + params['theta'])
            return 1/(q+1) * (1 - 1 / r0 * (np.sqrt(.25 + alpha * k * params['beta'] * x * (1 - (q+1) * x)) + .5))

        for r0 in r0_list:
            params['beta'] = r0 * (params['mu'] + params['gamma'])

            # Note: Only pass [I_0, R_0]
            sim_engine = SIRS(model_type='sirs_zero_layer_incidence', params=params)
            sol = sim_engine.simulate(t_span=params['t_span'], initial_conditions=[0.01, 0.0])

            I = sol[0]
            R = sol[1]
            S = 1 - I - R

            # Reconstruct M for analysis/plotting
            alpha = params['a1']
            beta_val = params['beta']
            k = params['k']
            q = params['gamma'] / (params['mu'] + params['theta'])

            print("R₀:", r0, "I_ee:", calculate_I_ee(r0, I[-1], params), "I_ee_approx:", 1/(q+1) * (1 - (1 / 2/ r0))  )




    def run_simulation(self, simulations, models=None, params=None,
                       r0_theta_list=None, a_list=None,
                       R0s=None, thetas=None, ks=None, deltas=None,
                       alpha_range=None, beta_range=None, r0_list=None):

        simulation_dict = {
            1: lambda: self.simulation_1_r0_and_theta(models=models, r0_theta_list=r0_theta_list, params=params),
            2: lambda: self.simulation_2_memory(models=models, params=params),
            3: lambda: self.simulation_3_r0_and_theta(models=models, params=params, r0_theta_list=r0_theta_list),
            4: lambda: self.simulation_4_r0_and_theta_with_2_layers(models=models, params=params, R0s=R0s, thetas=thetas, a_list=a_list),
            5: lambda: self.simulation_5_k(models=models, params=params, k=ks),
            6: lambda: self.simulation_6_r0_and_theta_prevalence_vs_incidence(models=models, params=params, R0s=R0s, thetas=thetas),
            7: lambda: self.simulation_7_a1_and_a2_prevalence_vs_incidence(models=models, params=params, a1s=a_list[0], a2s=a_list[1]),
            8: lambda: self.simulation_8_k_prevalence_vs_incidence(models=models, params=params, ks=ks),
            9: lambda: self.simulation_9_periodic(models=models, params=params, deltas=deltas),
            10: lambda: self.simulation_10_a1_and_a2(models=models, params=params, a_list=a_list),
            11: lambda: self.simulation_11_ie_given_alpha(params=params, alpha_range=alpha_range),
            12: lambda: self.simulation_12_distance_between_ee(params=params, beta_range=beta_range),
            13: lambda: self.simulation_13_impact_r0(params=params, r0_list=r0_list)
        }

        for simulation_number in simulations:
            # FIX: Add () to actually run the function now
            fig = simulation_dict[simulation_number]()

            # Save the figure
            if self.save_figures and fig is not None:
                fig.write_image(f"./img/simulation_{simulation_number}.png", scale=2)

        return fig



def simulation_1(models, r0_theta_list, params, show_cumulative_incidence=False):
    """
    Simulation 1. of the thesis.
    This method simulates the SIRS models with different R₀ and θ parameters.
    """
    assert len(r0_theta_list) == 4, "You need exactly 4 couples (R₀, θ)"

    start = params['start_simulation'] if 'start_simulation' in params else 0
    horizon = params['horizon'] if 'horizon' in params else 20000
    # print(horizon)

    # Defining subplots titles
    subplot_titles = [
        f"R₀ = {R0:.2f}, θ = {theta:.5f}"
        for R0, theta in r0_theta_list
    ]

    # Create 2D canvas grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.08, vertical_spacing=0.12
    )

    colors = ['blue', 'red', 'green', 'orange']

    # Keep track of the first plot to avoid repeating legends
    first_plot = True

    for idx, (R0, theta) in enumerate(r0_theta_list):
        # Assign row and column based on index
        row = idx // 2 + 1
        col = idx % 2 + 1

        # Prepare parameters for the current simulation
        params = params.copy()
        params['theta'] = theta
        params['beta']  = R0 * (params['mu'] + params['gamma'])

        t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)

        # Simulate the model
        solutions = [simulate(m, params) for m in models]

        # adding traces to the figure
        for _j, (model, sol, color) in enumerate(zip(models, solutions, colors)):
            I = sol[0] # sol[0] = infetti
            fig.add_trace(
                go.Scatter(
                    x=t[:horizon], y=I[:horizon],
                    mode='lines',
                    name=name_model(model),
                    legendgroup=name_model(model),
                    showlegend=first_plot,
                    line=dict(color=color)
                ),
                row=row, col=col
            )

            cum_inc = calculate_cumulative_incidence(sol, params)
            print(f"Cumulative incidence for {name_model(model)} with R₀={round(R0, 3)}, θ={round(theta, 3)}: {cum_inc:.4f}")

            # eventually add cumulative incidence notation to the plot
            if show_cumulative_incidence:
                fig.add_annotation(
                    text=f"Cumulative Incidence for({name_model(model)}):\t {cum_inc:.4f}",
                    xref='x domain', yref='y domain',
                    y=0.7 - 0.05*_j, x=0.5,
                    showarrow=False,
                    font=dict(size=10),
                    row=row, col=col
                )

        first_plot = False  # after the first plot, we don't need to show the legend again

    # global layout settings
    fig.update_layout(
        title_text="Confronto dei modelli SIRS sui vari scenari (R₀, θ)",
        template='plotly_white',
        width=1200, height=1000,
        legend_title="Modelli",
        margin=dict(t=100, b=50, l=50, r=50)
    )

    # uniform x and y axes titles
    fig.update_xaxes(title_text="Time (days)", row=2, col=1)
    fig.update_xaxes(title_text="Time (days)", row=2, col=2)
    fig.update_yaxes(title_text="Proportion infected", row=1, col=1)
    fig.update_yaxes(title_text="Proportion infected", row=2, col=1)

    fig.show()

    return fig

def simulation_2(models, params):
    """
    Simulation 2. of the thesis.
    This method plots the memory compartments of given SIRS models.
    """

    solutions = []
    for _, model in enumerate(models):
        print(f'Simulating {name_model(model)}\n')
        solution = simulate(model, params)
        solutions.append(solution)

    fig = Plots().plot_memory(models, solutions, params)

    return fig

def simulation_3a(models, params, r0_theta_list, show_cumulative_incidence=False):
    """
    Simulation 3. of the thesis.
    Same as simulation_01 but with more parameters.
    """

    assert len(r0_theta_list) == 9, "Servono esattamente 9 coppie (R0, θ)"

    horizon = params['horizon'] if 'horizon' in params else 20000

    # Defining subplots titles
    subplot_titles = [
        f"R₀ = {R0:.2f}, θ = {theta:.5f}"
        for R0, theta in r0_theta_list
    ]

    # Create 2D canvas grid
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=subplot_titles,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.08, vertical_spacing=0.12
    )

    colors = ['blue', 'red', 'green', 'orange']

    # Keep track of the first plot to avoid repeating legends
    first_plot = True

    for idx, (R0, theta) in enumerate(r0_theta_list):
        # Assign row and column based on index
        row = idx // 3 + 1
        col = idx % 3 + 1

        # Prepare parameters for the current simulation
        params = params.copy()
        params['theta'] = theta
        params['beta']  = R0 * (params['mu'] + params['gamma'])

        t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)

        # Simulate the models for the current parameters
        solutions = [simulate(m, params) for m in models]

        # Add traces to the figure
        for _j, (model, sol, color) in enumerate(zip(models, solutions, colors)):
            I = sol[0] # sol[0] = infectious
            fig.add_trace(
                go.Scatter(
                    x=t[:horizon], y=I[:horizon],
                    mode='lines',
                    name=name_model(model),
                    legendgroup=name_model(model),
                    showlegend=first_plot,
                    line=dict(color=color)
                ),
                row=row, col=col
            )


            cum_inc = calculate_cumulative_incidence(sol, params)
            print(f"Cumulative incidence for {name_model(model)} with R₀={round(R0, 3)}, θ={round(theta, 3)}: {cum_inc:.4f}")
            if show_cumulative_incidence:
                # eventually add cumulative incidence notation to the plot
                fig.add_annotation(
                    text=f"Cumulative Incidence({name_model(model)}): {cum_inc:.4f}",
                    xref='x domain', yref='y domain',
                    y=0.9 - 0.05*_j, x=0.5,
                    showarrow=False,
                    font=dict(size=10),
                    row=row, col=col
                )

        first_plot = False  # after the first plot, we don't need to show the legend again

    fig.update_layout(
        title_text="Confronto dei modelli SIRS sui vari scenari (R₀, θ)",
        template='plotly_white',
        width=1200, height=1000,
        legend_title="Modelli",
        margin=dict(t=100, b=50, l=50, r=50)
    )

    # uniform x and y axes titles
    fig.update_xaxes(title_text="Time (days)", row=2, col=1)
    fig.update_xaxes(title_text="Time (days)", row=2, col=2)
    fig.update_yaxes(title_text="Proportion infected", row=1, col=1)
    fig.update_yaxes(title_text="Proportion infected", row=2, col=1)

    fig.show()

    return fig

def simulation_3b(models, params, a_list, show_cumulative_incidence=False):
    """
    Simulation 3b. of the thesis.
    This method simulates the SIRS models with different a₁ and a₂ parameters.
    """

    horizon = params['horizon'] if 'horizon' in params else 20000

    # Defining subplots titles
    subplot_titles = [
        'SIRS con 1 Layer di Memoria',
        'SIRS con 2 Layer di Memoria'
    ]

    # Create 2D canvas grid
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.08, vertical_spacing=0.12
    )

    colors = ['rgba(255, 0, 125, 0.5)', 'rgba(0, 0, 0, 0.5)', 'rgba(0, 125, 255, 0.5)', 'rgba(255, 125, 0, 0.5)']

    for idx, (a1, a2) in enumerate(a_list):
        params = params.copy()
        params['a1'] = a1
        params['a2'] = a2

        t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)
        solutions = [simulate(m, params) for m in models]

        color = colors[idx]
        # notes positions
        x = 0.5
        y = 0.9 - 0.05*idx

        for _j, (model, sol) in enumerate(zip(models, solutions)):
            row = 1
            if model == sirs_one_layer:
                col = 1

                M = sol[2]

                fig.add_trace(
                    go.Scatter(
                        x=t[:horizon], y=M[:horizon],
                        mode='lines',
                        name=f'a₁={a1:.4f}, a₂={a2:.4f}',
                        legendgroup=name_model(model),
                        showlegend=True,
                        line=dict(color=color)
                    ),
                    row=1, col=col
                )

                cum_inc = calculate_cumulative_incidence(sol, params)

                if show_cumulative_incidence:
                    fig.add_annotation(
                        text=f"CI({name_model(model)}): {cum_inc:.4f}",
                        xref='x domain', yref='y domain',
                        y=y, x=0.5,
                        showarrow=False,
                        font=dict(size=10),
                        row=row, col=col
                    )
                    first_plot_sx = False

            elif model == sirs_two_layer:
                M1 = sol[2]
                M2 = sol[3]
                col = 2

                fig.add_trace(
                    go.Scatter(
                        x=t[:horizon], y=M1[:horizon] + M2[:horizon],
                        mode='lines',
                        name=f'a₁: {a1:.4f}, a₂: {a2:.4f}',
                        legendgroup=name_model(model),
                        showlegend=False,
                        line=dict(color=color)
                    ),
                    row=row, col=col
                )
                cum_inc = calculate_cumulative_incidence(sol, params)
                if show_cumulative_incidence:
                    fig.add_annotation(
                        text=f"CI({name_model(model)}): {cum_inc:.4f}",
                        xref='x domain', yref='y domain',
                        y=y, x=0.5,
                        showarrow=False,
                        font=dict(size=10),
                        row=row, col=col
                    )

            first_plot_sx = False




    # Layout globale
    fig.update_layout(
        title_text="Confronto di modelli SIRS con memoria per varie coppie (a₁, a₂)",
        template='plotly_white',
        width=1500, height=800,
        legend_title="Coppie (a₁, a₂)",
        margin=dict(t=100, b=50, l=50, r=50)
    )

    fig.update_xaxes(title_text="Time (days)", row=2, col=1)
    fig.update_xaxes(title_text="Time (days)", row=2, col=2)
    fig.update_yaxes(title_text="Proportion infected", row=1, col=1)
    fig.update_yaxes(title_text="Proportion infected", row=2, col=1)
    # share the y axes
    fig.update_yaxes(matches='y')

    fig.show()

    return fig

def simulation_4(models, params, R0s, thetas, a_list, show_cumulative_incidence=False):
    """
    Simulation 4. of the thesis.
    This method is similar to simulation_3b but it plots the SIRS models with 2 layers of memory for different (R₀, θ) scenarios.
    """

    t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)

    horizon = params['horizon'] if 'horizon' in params else 20000

    # Defining subplots titles
    subplot_titles = [
        f'R₀ = {R0:.2f}, θ = {theta:.5f}' for R0, theta in zip(R0s, thetas)
    ]

    # Create 2D canvas grid
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.08, vertical_spacing=0.12
    )

    colors = ['rgba(255, 0, 125, 0.5)', 'rgba(0, 0, 0, 0.5)', 'rgba(0, 125, 255, 0.5)', 'rgba(255, 125, 0, 0.5)']
    # i_colors = ['rgba(0, 0, 255, 1)', 'rgba(0, 0, 255, 0.5)', 'rgba(0, 0, 255, 0.2)', 'rgba(0, 56, 255, 0.5)']              # colors for the infectious class

    for idx, (a1, a2) in enumerate(a_list):
        params = params.copy()
        params['a1'] = a1
        params['a2'] = a2

        first_plot = True

        for _j, (R0, theta) in enumerate(zip(R0s, thetas)):
            # Prepare parameters for the current simulation
            params['theta'] = theta
            params['beta']  = R0 * (params['mu'] + params['gamma'])


            solutions = [simulate(m, params) for m in models]

            color = colors[idx]
            # i_color = i_colors[idx]
            x = 0.5
            y = 0.9 - 0.05*idx

            # Assign row and column based on index
            row = 1
            col = _j + 1

            # I = solutions[0][0]
            M1 = solutions[0][2]
            M2 = solutions[0][3]

            fig.add_trace(
                go.Scatter(
                    x=t[:horizon], y=M1[:horizon] + M2[:horizon],
                    mode='lines',
                    name=f'a₁: {a1:.4f}, a₂: {a2:.4f}',
                    legendgroup=name_model(models[0]),
                    showlegend=first_plot,
                    line=dict(color=color)
                ),
                row=row, col=col
            )
            # fig.add_trace(
            #     go.Scatter(
            #         x=t[:horizon], y=I[:horizon],
            #         mode='lines',
            #         name=f'a₁: {a1:.4f}, a₂: {a2:.4f}',
            #         legendgroup=name_model(models[0]),
            #         showlegend=False,
            #         line=dict(color=i_color)
            #     ),
            #     row=row, col=col
            # )

            cum_inc = calculate_cumulative_incidence(solutions[0], params)
            print(f"Cumulative incidence for {name_model(models[0])} with R₀={round(R0, 3)}, θ={round(theta, 3)}: {cum_inc:.4f}")
            if show_cumulative_incidence:
                fig.add_annotation(
                    text=f"CI({name_model(models[0])}): {cum_inc:.4f}",
                    xref='x domain', yref='y domain',
                    y=y, x=0.5,
                    showarrow=False,
                    font=dict(size=10),
                    row=row, col=col
                )
            first_plot = False


    fig.update_layout(
        title_text="Confronto del modello SIRS con 2 layer di memoria sui vari scenari (a₁, a₂)",
        template='plotly_white',
        width=1500, height=800,
        legend_title="Modelli",
        margin=dict(t=100, b=50, l=50, r=50)
    )
    fig.update_xaxes(title_text="Time (days)", row=2, col=1)
    fig.update_xaxes(title_text="Time (days)", row=2, col=2)
    fig.update_yaxes(title_text="Proportion infected", row=1, col=1)
    fig.update_yaxes(title_text="Proportion infected", row=2, col=1)
    # share the y axes
    # fig.update_yaxes(matches='y')

    fig.show()

    return fig

def simulation_5(models, params, k, show_cumulative_incidence=False):
    """
    Simulation 5. of the thesis.
    This method simulates the SIRS models with different k parameters.
    """

    t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)
    horizon = params['horizon'] if 'horizon' in params else 20000

    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, k_val in enumerate(k):
        params = params.copy()
        params['k'] = k_val

        # Simulate the models for the current parameters
        solutions = [simulate(m, params) for m in models]
        solution_data = solutions[0]

        I = solution_data[0]

        fig.add_trace(
            go.Scatter(
                x=t[:horizon],
                y=I[:horizon],
                mode='lines',
                name=f'k = {k_val}',
                line=dict(color=colors[idx % len(colors)])
            )
        )

        cum_inc = calculate_cumulative_incidence(solution_data, params)
        print(f"Cumulative incidence for {name_model(models[0])} with k={k_val}: {cum_inc:.4f}")
        if show_cumulative_incidence:
            # eventually add cumulative incidence notation to the plot
            fig.add_annotation(
                text=f"CI({name_model(models[0])}): {cum_inc:.4f}",
                xref='x domain', yref='y domain',
                y=0.9 - 0.05*idx, x=0.5,
                showarrow=False,
                font=dict(size=10)
            )

    # 4. Aggiornare il layout della figura singola
    fig.update_layout(
        title_text="Confronto del modello SIRS con 2 layer di memoria sui vari scenari (k) - Numero di infetti",
        xaxis_title="Time (days)",
        yaxis_title="Proportion infected",
        template='plotly_white',
        width=1200,
        height=600,
        legend_title_text="Valore di k" #
    )

    fig.show()

    return fig

def simulation_6(models, params, R0s, thetas, show_cumulative_incidence=False):
    """
    Simulation 6. of the thesis.
    This method simulates prevalence-based and incidence-based SIRS models with different (R₀, θ) scenarios.
    """

    t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)
    start = 0.
    horizon = 20000

    # Defining subplots titles
    subplot_titles = [
        f'R₀ = {R0:.2f}, θ = {theta:.5f}' for R0, theta in zip(R0s, thetas)
    ]

    # Create 2D canvas grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.08, vertical_spacing=0.12
    )

    colors = ['rgba(255, 0, 125, 0.5)', 'rgba(0, 0, 0, 0.5)', 'rgba(0, 125, 255, 0.5)', 'rgba(255, 125, 0, 0.5)']

    first_plot = True

    for idx, (R0, theta) in enumerate(zip(R0s, thetas)):
        params = params.copy()
        params['theta'] = theta
        params['beta']  = R0 * (params['mu'] + params['gamma'])


        # Simulate the models for the current parameters
        solutions = [simulate(m, params) for m in models]

        # notes positions
        x = 0.5
        # Assign row and column based on index
        row = idx // 2 + 1
        col = idx % 2 + 1


        for _j, (model, sol, color) in enumerate(zip(models, solutions, colors)):
            y = 0.9 - 0.05*_j

            I = sol[0] # sol[0] = infetti
            fig.add_trace(
                go.Scatter(
                    x=t[:horizon], y=I[:horizon],
                    mode='lines',
                    #name=name_model(model),
                    #legendgroup=name_model(model),
                    # showlegend=first_plot,
                    showlegend = False,
                    line=dict(color=color)
                ),
                row=row, col=col
            )
            cum_inc = calculate_cumulative_incidence(sol, params)
            print(f"Cumulative incidence for {name_model(model)} with R₀={round(R0, 3)}, θ={round(theta, 3)}: {cum_inc:.4f}")
            if show_cumulative_incidence:
                # eventually add cumulative incidence notation to the plot
                fig.add_annotation(
                    text=f"CI({name_model(model)}): {cum_inc:.4f}",
                    xref='x domain', yref='y domain',
                    y=y, x=0.5,
                    showarrow=False,
                    font=dict(size=10),
                    row=row, col=col
                )

        first_plot = False  # dopo la prima cella, non ripetere le legende

    fig.update_layout(
        # title_text="Confronto dei modelli SIRS con memoria prevalence-based e incidence-based per diversi valori (R₀, θ)",
        title_text=" ",
        template='plotly_white',
        width=1500, height=800,
        legend_title="Modelli",
        margin=dict(t=100, b=50, l=50, r=50)
    )

    fig.update_xaxes(title_text="Time (days)", row=2, col=1)
    fig.update_xaxes(title_text="Time (days)", row=2, col=2)
    fig.update_yaxes(title_text="Proportion infected", row=1, col=1)
    fig.update_yaxes(title_text="Proportion infected", row=2, col=1)

    fig.show()

    return fig

def simulation_7(models, params, a1s, a2s, show_cumulative_incidence=False):
    """
    Simulation 7. of the thesis.
    This method simulates prevalence-based and incidence-based SIRS models with different couples (a₁, a₂).
    """

    t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)
    start = 0.
    horizon = 20000

    # Defining subplots titles
    subplot_titles = [
        f'a₁ = {a1:.4f}, a₂ = {a2:.4f}' for a1, a2 in zip(a1s, a2s)
    ]

    # Create 2D canvas grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.08, vertical_spacing=0.12
    )

    colors = ['rgba(255, 0, 125, 0.5)', 'rgba(0, 0, 0, 0.5)', 'rgba(0, 125, 255, 0.5)', 'rgba(255, 125, 0, 0.5)']

    first_plot = True

    for idx, (a1, a2) in enumerate(zip(a1s, a2s)):
        params = params.copy()
        params['a1'] = a1
        params['a2'] = a2

        # Simulate the models for the current parameters
        solutions = [simulate(m, params) for m in models]

        # notes positions
        x = 0.5
        # Assign row and column based on index
        row = idx // 2 + 1
        col = idx % 2 + 1

        for _j, (model, sol, color) in enumerate(zip(models, solutions, colors)):
            y = 0.9 - 0.05*_j

            I = sol[0] # sol[0] = infetti
            fig.add_trace(
                go.Scatter(
                    x=t[:horizon], y=I[:horizon],
                    mode='lines',
                    name=name_model(model),
                    legendgroup=name_model(model),
                    showlegend=first_plot,
                    line=dict(color=color)
                ),
                row=row, col=col
            )
            cum_inc = calculate_cumulative_incidence(sol, params)
            print(f"Cumulative incidence for {name_model(model)} with a₁={round(a1, 3)}, a₂={round(a2, 3)}: {cum_inc:.4f}")
            if show_cumulative_incidence:
                # eventually add cumulative incidence notation to the plot
                fig.add_annotation(
                    text=f"CI({name_model(model)}): {cum_inc:.4f}",
                    xref='x domain', yref='y domain',
                    y=y, x=0.5,
                    showarrow=False,
                    font=dict(size=10),
                    row=row, col=col
                )

        first_plot = False  # after the first plot, we don't need to show the legend again

    fig.update_layout(
        title_text="Confronto dei modelli SIRS con memoria prevalence-based e incidence-based per diversi valori (a₁, a₂)",
        template='plotly_white',
        width=1500, height=800,
        legend_title="Modelli",
        margin=dict(t=100, b=50, l=50, r=50)
    )

    fig.update_xaxes(title_text="Time (days)", row=2, col=1)
    fig.update_xaxes(title_text="Time (days)", row=2, col=2)
    fig.update_yaxes(title_text="Proportion infected", row=1, col=1)
    fig.update_yaxes(title_text="Proportion infected", row=2, col=1)

    fig.show()

    return fig

def simulation_8(models, params, ks, show_cumulative_incidence=False):
    """
    Simulation 8. of the thesis.
    This method simulates prevalence-based and incidence-based SIRS models with different k parameters.
    """

    t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)
    start = 0.
    horizon = 20000

    # Defining subplots titles
    subplot_titles = [
        f'k = {k_val}' for k_val in ks
    ]

    # Create 2D canvas grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.08, vertical_spacing=0.12
    )

    colors = ['rgba(255, 0, 125, 0.5)', 'rgba(0, 0, 0, 0.5)', 'rgba(0, 125, 255, 0.5)', 'rgba(255, 125, 0, 0.5)']

    first_plot = True

    for idx, k_val in enumerate(ks):
        params = params.copy()
        params['k'] = k_val

        # Simulate the models for the current parameters
        solutions = [simulate(m, params) for m in models]

        # notes positions
        x = 0.5
        # Assign row and column based on index
        row = idx // 2 + 1
        col = idx % 2 + 1

        for _j, (model, sol, color) in enumerate(zip(models, solutions, colors)):
            y = 0.9 - 0.05*_j

            I = sol[0] # sol[0] = infetti
            fig.add_trace(
                go.Scatter(
                    x=t[:horizon], y=I[:horizon],
                    mode='lines',
                    name=name_model(model),
                    legendgroup=name_model(model),
                    showlegend=first_plot,
                    line=dict(color=color)
                ),
                row=row, col=col
            )
            cum_inc = calculate_cumulative_incidence(sol, params)
            print(f"Cumulative incidence for {name_model(model)} with k={round(k_val, 3)}: {cum_inc:.4f}")
            if show_cumulative_incidence:
                # eventually add cumulative incidence notation to the plot
                fig.add_annotation(
                    text=f"CI({name_model(model)}): {cum_inc:.4f}",
                    xref='x domain', yref='y domain',
                    y=y, x=0.65,
                    showarrow=False,
                    font=dict(size=10),
                    row=row, col=col
                )

        first_plot = False  # after the first plot, we don't need to show the legend again

    fig.update_layout(
        title_text="Confronto dei modelli SIRS con memoria prevalence-based e incidence-based per diversi valori (k)",
        template='plotly_white',
        width=1500, height=1000,
        legend_title="Modelli",
        margin=dict(t=100, b=50, l=50, r=50)
    )

    fig.update_xaxes(title_text="Time (days)")
    fig.update_yaxes(title_text="Proportion infected")

    fig.show()

    return fig

def simulation_9(models, params, deltas, show_cumulative_incidence=False):
    """
    The following simulation will analyze the behaviour of the models with a periodic component

    :param models:
    :param params:
    :param ks:
    :param show_cumulative_incidence:
    :return:
    """

    t = np.linspace(params['t_span'][0], params['t_span'][1], 20000)
    start = 0.
    horizon = 20000

    # Defining subplots titles
    subplot_titles = [
        f'δ = {delta}' for delta in deltas
    ]

    # Create 2D canvas grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.08, vertical_spacing=0.12
    )

    for delta in deltas:
        params = params.copy()
        params['delta'] = delta

        # Assign row and column based on index
        idx = deltas.index(delta)
        row = idx // 2 + 1
        col = idx % 2 + 1

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # Simulate the models for the current parameters
        solutions = [simulate(m, params) for m in models]

        for idx, (model, sol) in enumerate(zip(models, solutions)):
            I = sol[0]
            S = 1 - sol[0] - sol[1]

            S_eq = (params['mu'] + params['gamma']) / params['beta']
            I_eq = I[-1]

            fig.add_trace(
                go.Scatter(
                    x=S[:horizon] / ([S_eq] * horizon),
                    y=I[:horizon] / ([I_eq] * horizon),
                    mode='lines',
                    name=f'{name_model(model)}',
                    line=dict(color=colors[idx % len(colors)])
                ),
                row=row, col=col,
            )

    fig.update_layout(
        title_text="Andamento all'equilibrio per diversi modelli, con diversi valori di δ",
        template='plotly_white',
        width=1500, height=1000,
        legend_title="Modelli",
        margin=dict(t=100, b=50, l=50, r=50)
    )

    fig.update_xaxes(title_text="I / Ie")
    fig.update_yaxes(title_text="S / Se")
    fig.show()

    return fig