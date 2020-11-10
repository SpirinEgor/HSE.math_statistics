from typing import List

from plotly.graph_objects import Scatter, Figure

from assignment_1.experiment import Experiment


def create_mse_scatter(mse: List[float], distribution_name: str, theta: float) -> Scatter:
    x_range = [i for i in range(1, len(mse) + 1)]
    return Scatter(x=x_range, y=mse, mode='lines+markers', name=f"{distribution_name} (θ = {theta})")


def build_mse_dependency(
        distribution_name: str, thetas: List[float], n_moments: int, n_samples: int, n_runs: int
) -> Figure:
    figure = Figure()
    figure.update_layout(title=f"Dependency between moment and MSE for {distribution_name} distribution")
    figure.update_xaxes(title="k")
    figure.update_yaxes(title="MSE")
    for theta in thetas:
        experiment = Experiment(distribution_name, theta)
        mse = experiment.run(n_moments, n_samples, n_runs)
        figure.add_trace(create_mse_scatter(mse, distribution_name, theta))
    return figure
