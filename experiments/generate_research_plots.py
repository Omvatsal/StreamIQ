"""Generate verified research visualizations from multi-seed experiment output."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from network_lab.traces.models import NetworkTrace
from rl_engine.agents.dqn import DQNAgent
from rl_engine.environment.abr_env import ABREnvironment
from rl_engine.evaluation.baselines import buffer_based, fixed_bitrate, throughput_based
from rl_engine.evaluation.evaluator import evaluate_policy
from rl_engine.evaluation.metrics import summarize
from rl_engine.training.ablations import ABLATIONS


POLICIES = ("dqn", "fixed", "throughput", "buffer")
POLICY_LABELS = {
    "dqn": "DQN",
    "fixed": "Fixed 240p",
    "throughput": "Throughput",
    "buffer": "Buffer",
}
METRICS = (
    "qoe",
    "average_bitrate_mbps",
    "rebuffer_seconds",
    "bitrate_switches",
    "total_bitrate_variation_mbps",
)


def _plt():
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 220, "font.size": 10})
    return plt


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    _plt().close(figure)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _mean_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, 0.0
    from scipy.stats import t

    critical = float(t.ppf(0.975, len(array) - 1))
    return mean, float(critical * array.std(ddof=1) / np.sqrt(len(array)))


def _result_rows(output: Path) -> list[dict[str, object]]:
    data = json.loads((output / "metrics" / "results.json").read_text(encoding="utf-8"))
    rows = []
    for key, value in data.items():
        seed, ablation = key.split(":", 1)
        for policy, metrics in value["policies"].items():
            rows.append({"seed": int(seed), "ablation": ablation, "policy": policy, **metrics})
    return rows


def _training_plots(output: Path, plots: Path, seeds: list[int]) -> None:
    plt = _plt()
    by_ablation: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for seed in seeds:
        for ablation in ABLATIONS:
            path = output / "training" / str(seed) / f"{ablation.name}.csv"
            by_ablation[ablation.name].append(_read_csv(path))
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for name, histories in by_ablation.items():
        length = min(len(history) for history in histories)
        rewards = np.asarray(
            [[float(row["reward"]) for row in history[:length]] for history in histories]
        )
        losses = np.asarray(
            [[float(row["mean_loss"]) for row in history[:length]] for history in histories]
        )
        x = np.arange(length)
        for axis, values, label in ((axes[0], rewards, name), (axes[1], losses, name)):
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            axis.plot(x, mean, label=label)
            axis.fill_between(x, mean - std, mean + std, alpha=0.15)
    axes[0].set_title("Training reward across seeds")
    axes[0].set_ylabel("Episode reward")
    axes[1].set_title("Training loss across seeds")
    axes[1].set_ylabel("Mean loss")
    axes[1].set_xlabel("Episode")
    axes[0].legend(fontsize=8, ncol=2)
    _save(figure, plots / "training" / "reward_and_loss.png")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, histories in by_ablation.items():
        length = min(len(history) for history in histories)
        epsilons = np.asarray(
            [[float(row["epsilon"]) for row in history[:length]] for history in histories]
        )
        x = np.arange(length)
        axes[0].plot(x, epsilons.mean(axis=0), label=name)
    validation: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for seed in seeds:
        for ablation in ABLATIONS:
            rows = _read_csv(
                output / "training" / str(seed) / ablation.name / "validation.csv"
            )
            validation[ablation.name].append(
                (
                    np.asarray([float(row["validation_index"]) for row in rows]),
                    np.asarray([float(row["score"]) for row in rows]),
                )
            )
    for name, series in validation.items():
        if not series:
            continue
        x = series[0][0]
        values = np.asarray([item[1] for item in series])
        axes[1].plot(x, values.mean(axis=0), marker="o", label=name)
        axes[1].fill_between(
            x, values.mean(axis=0) - values.std(axis=0),
            values.mean(axis=0) + values.std(axis=0), alpha=0.15,
        )
    axes[0].set_title("Epsilon decay")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Epsilon")
    axes[1].set_title("Validation performance")
    axes[1].set_xlabel("Validation episode")
    axes[1].set_ylabel("Validation criterion")
    axes[0].legend(fontsize=8, ncol=2)
    _save(figure, plots / "training" / "epsilon_and_validation.png")


def _summary_plots(rows: list[dict[str, object]], plots: Path) -> None:
    plt = _plt()
    for metric, title, ylabel, filename in (
        ("qoe", "QoE comparison", "Independent QoE", "qoe_comparison.png"),
        ("average_bitrate_mbps", "Average bitrate comparison", "Average bitrate (Mbps)", "average_bitrate.png"),
        ("rebuffer_seconds", "Rebuffering duration comparison", "Rebuffering (seconds)", "rebuffering_duration.png"),
        ("bitrate_switches", "Bitrate switching frequency", "Switches per episode", "switching_frequency.png"),
        ("total_bitrate_variation_mbps", "Total bitrate variation", "Total variation (Mbps)", "total_variation.png"),
    ):
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row["ablation"] == "full":
                grouped[str(row["policy"])].append(float(row[metric]))
        names = list(POLICIES)
        means, errors = zip(*[_mean_ci(grouped[name]) for name in names])
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar([POLICY_LABELS[name] for name in names], means, yerr=errors, capsize=4)
        axis.set_title(title + " (mean ± 95% CI)")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=20)
        _save(figure, plots / "baselines" / filename)
    for metric, title, ylabel, filename in (
        ("average_bitrate_mbps", "DQN average bitrate by ablation", "Average bitrate (Mbps)", "dqn_average_bitrate_mbps.png"),
        ("rebuffer_seconds", "DQN rebuffering by ablation", "Rebuffering (seconds)", "dqn_rebuffer_seconds.png"),
        ("total_bitrate_variation_mbps", "DQN bitrate variation by ablation", "Total variation (Mbps)", "dqn_total_bitrate_variation_mbps.png"),
    ):
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row["policy"] == "dqn":
                grouped[str(row["ablation"])].append(float(row[metric]))
        names = [ablation.name for ablation in ABLATIONS]
        means, errors = zip(*[_mean_ci(grouped[name]) for name in names])
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.bar(names, means, yerr=errors, capsize=4)
        axis.set_title(title + " (mean ± 95% CI)")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=55)
        _save(figure, plots / filename)


def _holm_adjustments(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running_max
    return adjusted


def _format_mean_sd(values: list[float]) -> str:
    array = np.asarray(values, dtype=float)
    deviation = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return f"{array.mean():.3f} ± {deviation:.3f}"


def _report(output: Path, rows: list[dict[str, object]], seeds: list[int]) -> None:
    data = json.loads((output / "metrics" / "results.json").read_text(encoding="utf-8"))
    metrics = (
        ("average_bitrate_mbps", "Average bitrate (Mbps)"),
        ("rebuffer_seconds", "Rebuffering (seconds)"),
        ("bitrate_switches", "Bitrate switches"),
        ("total_bitrate_variation_mbps", "Total variation (Mbps)"),
        ("qoe", "Independent QoE"),
    )
    lines = [
        "# StreamIQ Research Validation Report",
        "",
        f"Dataset manifest: `experiments/results/research_multiseed/datasets_*/manifest.json`",
        "",
        "## Experiment scope",
        "",
        f"- Seeds: {', '.join(map(str, seeds))}; training duration: 20 episodes per seed and ablation.",
        "- Six ablations were independently retrained for each seed.",
        "- Tables report mean ± sample standard deviation across the three seeds.",
        "- QoE is independently calculated from bitrate, rebuffering, and bitrate variation.",
        "",
        "## Policy results on unseen test traces",
        "",
        "| Ablation | Policy | " + " | ".join(label for _, label in metrics) + " |",
        "|---|---|" + "|".join("---" for _ in metrics) + "|",
    ]
    for ablation in [item.name for item in ABLATIONS]:
        for policy in POLICIES:
            selected = [
                row for row in rows
                if row["ablation"] == ablation and row["policy"] == policy
            ]
            lines.append(
                "| "
                + " | ".join(
                    [ablation, POLICY_LABELS[policy]]
                    + [
                        _format_mean_sd([float(row[metric]) for row in selected])
                        for metric, _ in metrics
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## DQN ablation summary",
            "",
            "| Ablation | " + " | ".join(label for _, label in metrics) + " |",
            "|---|" + "|".join("---" for _ in metrics) + "|",
        ]
    )
    for ablation in [item.name for item in ABLATIONS]:
        selected = [
            row for row in rows
            if row["ablation"] == ablation and row["policy"] == "dqn"
        ]
        lines.append(
            "| "
            + " | ".join(
                [ablation]
                + [
                    _format_mean_sd([float(row[metric]) for row in selected])
                    for metric, _ in metrics
                ]
            )
            + " |"
        )

    comparisons = []
    for key, result in data.items():
        seed, ablation = key.split(":", 1)
        for baseline, statistics in result["paired_dqn_qoe"].items():
            comparisons.append(
                {
                    "seed": int(seed),
                    "ablation": ablation,
                    "baseline": baseline,
                    **statistics,
                }
            )
    adjusted = _holm_adjustments([float(item["p_value"]) for item in comparisons])
    lines.extend(
        [
            "",
            "## Paired QoE comparisons",
            "",
            "Each comparison pairs DQN and baseline QoE on the same unseen test "
            "traces. Raw p-values come from paired t-tests over those trace-level "
            "observations. Adjusted p-values use Holm-Bonferroni correction across "
            f"all {len(comparisons)} reported comparisons (six ablations × three seeds × three baselines).",
            "A p-value below 0.05 is not treated as conclusive evidence here because "
            "the experiment has only three training seeds and 20 episodes per run.",
            "",
            "| Seed | Ablation | Baseline | Mean DQN−baseline QoE | Cohen's dz | Raw p | Holm-adjusted p |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for item, adjusted_p in zip(comparisons, adjusted):
        lines.append(
            f"| {item['seed']} | {item['ablation']} | {item['baseline']} | "
            f"{item['mean_difference']:.3f} | {item['cohens_dz']:.3f} | "
            f"{item['p_value']:.6g} | {adjusted_p:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The run is smoke-scale: 20 training episodes and three independent seeds.",
            "- Results support descriptive comparisons, not claims of convergence or production superiority.",
            "- The paired tests use trace-level pairing within each seed/ablation; they do not replace a hierarchical analysis across training seeds.",
            "- Holm correction controls family-wise error for the 54 reported paired QoE comparisons, but no inferential tests were performed for the other metrics.",
            "",
            "## Visualization index",
            "",
            f"Generated from actual multi-seed results for seeds {', '.join(map(str, seeds))}.",
            "PNG files were checked for existence, non-zero size, and readable image dimensions.",
            "",
        ]
    )
    for path in sorted((output / "plots").rglob("*.png")):
        lines.append(f"- `{path.relative_to(output).as_posix()}`")
    (output / "reports").mkdir(parents=True, exist_ok=True)
    (output / "reports" / "research_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _load_test_traces(output: Path, seed: int) -> list[NetworkTrace]:
    return [
        NetworkTrace.load(path)
        for path in sorted((output / f"datasets_{seed}" / "test").glob("*.json"))
    ]


def _episode_rows(environment: ABREnvironment, policy, label: str) -> list[dict[str, float | int | str]]:
    state, _ = environment.reset(seed=environment.trace.seed)
    done = False
    while not done:
        state, _, terminated, truncated, _ = environment.step(policy(state, environment))
        done = terminated or truncated
    return [{"policy": label, **metric.__dict__} for metric in environment.metrics.segments]


def _streaming_plots(output: Path, plots: Path, seed: int) -> None:
    plt = _plt()
    checkpoint = output / "training" / str(seed) / "full" / "model_selected.pt"
    traces = _load_test_traces(output, seed)
    if not traces:
        raise FileNotFoundError("no test traces found")
    trace = traces[0]
    agent = DQNAgent(9, seed=seed, epsilon_start=0.0, epsilon_end=0.0)
    agent.load(checkpoint)
    dqn = lambda state, _env: agent.select_action(state, explore=False)
    policies = {"dqn": dqn, "fixed": fixed_bitrate(0), "throughput": throughput_based, "buffer": buffer_based}
    rows = []
    for name, policy in policies.items():
        rows.extend(_episode_rows(ABREnvironment(trace), policy, POLICY_LABELS[name]))
    representative = output / "metrics" / "representative_episode.csv"
    representative.parent.mkdir(parents=True, exist_ok=True)
    with representative.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = _plt().subplots(3, 1, figsize=(11, 10), sharex=True)
    for label in {row["policy"] for row in rows}:
        selected = [row for row in rows if row["policy"] == label]
        x = np.arange(len(selected))
        bandwidth = [float(row["bandwidth_mbps"]) for row in selected]
        bitrate = [float(row["bitrate_mbps"]) for row in selected]
        throughput = [float(row["throughput_mbps"]) for row in selected]
        buffer = [float(row["buffer_occupancy_seconds"]) for row in selected]
        rebuffer = [float(row["rebuffer_seconds"]) for row in selected]
        axes[0].plot(x, bandwidth, "--", alpha=0.65, label=f"Bandwidth ({label})")
        axes[0].plot(x, bitrate, label=f"Bitrate ({label})")
        axes[1].plot(x, throughput, label=label)
        axes[2].plot(x, buffer, label=f"Buffer ({label})")
        events = [index for index, value in enumerate(rebuffer) if value > 0]
        axes[2].scatter(events, [buffer[index] for index in events], marker="x", s=45)
    axes[0].set_title("Network bandwidth versus selected bitrate")
    axes[0].set_ylabel("Mbps")
    axes[1].set_title("Observed throughput over time")
    axes[1].set_ylabel("Mbps")
    axes[2].set_title("Buffer occupancy and rebuffering events")
    axes[2].set_ylabel("Buffer (seconds)")
    axes[2].set_xlabel("Segment index")
    axes[0].legend(fontsize=7, ncol=2)
    _save(axes[0].figure, plots / "streaming" / "representative_network_and_buffer.png")

    def line_plot(title: str, ylabel: str, series: dict[str, list[float]], filename: str) -> None:
        figure, axis = plt.subplots(figsize=(10, 5))
        for label, values in series.items():
            axis.plot(np.arange(len(values)), values, label=label)
        axis.set_title(title)
        axis.set_xlabel("Segment index")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
        _save(figure, plots / "streaming" / filename)

    line_plot(
        "Network bandwidth versus selected bitrate",
        "Mbps",
        {
            f"Bandwidth ({label})": [float(row["bandwidth_mbps"]) for row in rows if row["policy"] == label]
            for label in {row["policy"] for row in rows}
        }
        | {
            f"Bitrate ({label})": [float(row["bitrate_mbps"]) for row in rows if row["policy"] == label]
            for label in {row["policy"] for row in rows}
        },
        "bandwidth_vs_bitrate.png",
    )
    line_plot(
        "Observed throughput over time",
        "Throughput (Mbps)",
        {
            label: [float(row["throughput_mbps"]) for row in rows if row["policy"] == label]
            for label in {row["policy"] for row in rows}
        },
        "throughput_over_time.png",
    )
    line_plot(
        "Buffer occupancy and rebuffering events",
        "Buffer occupancy (seconds)",
        {
            label: [float(row["buffer_occupancy_seconds"]) for row in rows if row["policy"] == label]
            for label in {row["policy"] for row in rows}
        },
        "buffer_occupancy.png",
    )


def _ablation_plot(rows: list[dict[str, object]], plots: Path) -> None:
    plt = _plt()
    metrics = ("qoe", "total_bitrate_variation_mbps", "rebuffer_seconds")
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = [ablation.name for ablation in ABLATIONS]
    for axis, metric in zip(axes, metrics):
        means, errors = [], []
        for name in names:
            values = [
                float(row[metric])
                for row in rows
                if row["policy"] == "dqn" and row["ablation"] == name
            ]
            mean, error = _mean_ci(values)
            means.append(mean)
            errors.append(error)
        axis.bar(names, means, yerr=errors, capsize=4)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylabel("Mean ± 95% CI")
        axis.tick_params(axis="x", rotation=55)
    _save(figure, plots / "ablations" / "reward_and_state_ablations.png")


def generate(output: Path, seed: int) -> list[Path]:
    rows = _result_rows(output)
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) < 2:
        raise ValueError("multi-seed results require at least two seeds")
    plots = output / "plots"
    _training_plots(output, plots, seeds)
    _summary_plots(rows, plots)
    _streaming_plots(output, plots, seed)
    _ablation_plot(rows, plots)
    generated = sorted(plots.rglob("*.png"))
    if len(generated) < 10 or any(path.stat().st_size == 0 for path in generated):
        raise RuntimeError("plot generation did not produce non-empty PNG files")
    _report(output, rows, seeds)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/results/research_multiseed"))
    parser.add_argument("--representative-seed", type=int, default=11)
    args = parser.parse_args()
    generated = generate(args.output, args.representative_seed)
    print(f"generated {len(generated)} plots")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
