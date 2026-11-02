"""Plots for research validation runs."""

from pathlib import Path


def plot_policy_metric(results, metric: str, output: str | Path) -> Path:
    import matplotlib.pyplot as plt
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = list(results)
    values = [getattr(results[name], metric) for name in names]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(names, values)
    axis.set_ylabel(metric.replace("_", " ").title())
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return destination
