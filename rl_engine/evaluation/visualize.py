"""Training and evaluation plots."""

from pathlib import Path


def plot_training(history, output: str | Path) -> Path:
    import matplotlib.pyplot as plt
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(9, 8))
    axes[0].plot(history.rewards)
    axes[0].set_ylabel("Reward")
    axes[1].plot(history.losses)
    axes[1].set_ylabel("Loss")
    axes[2].plot(history.epsilons)
    axes[2].set_ylabel("Epsilon")
    axes[2].set_xlabel("Episode")
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return destination


def plot_comparison(results, output: str | Path) -> Path:
    import matplotlib.pyplot as plt
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = list(results)
    values = [results[name].qoe for name in names]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(names, values)
    axis.set_ylabel("QoE")
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return destination
