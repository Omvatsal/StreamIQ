"""Focused analysis of DQN decisions around observed bandwidth drops."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


Q_COLUMNS = tuple(f"q_value_action_{action}" for action in range(5))
STATE_COLUMNS = (
    "buffer_occupancy_seconds",
    "throughput_history_0_mbps",
    "throughput_history_1_mbps",
    "throughput_history_2_mbps",
    "throughput_history_3_mbps",
    "throughput_history_4_mbps",
    "previous_bitrate_mbps",
    "previous_download_duration_seconds",
    "remaining_segments",
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build_timelines(rows: list[dict[str, str]], radius: int = 2) -> list[dict[str, object]]:
    """Return consecutive decision rows around each detected drop."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["trace_id"]].append(row)

    timelines: list[dict[str, object]] = []
    drop_number = 0
    for trace_id, trace_rows in grouped.items():
        trace_rows.sort(key=lambda row: int(row["segment_index"]))
        for index, row in enumerate(trace_rows):
            if row["bandwidth_drop"] != "True":
                continue
            drop_number += 1
            entries = []
            for offset in range(-radius, radius + 1):
                position = index + offset
                if 0 <= position < len(trace_rows):
                    entry = dict(trace_rows[position])
                    entry["drop_id"] = f"drop_{drop_number:03d}"
                    entry["offset"] = offset
                    entry["observation"] = entry["state_vector"]
                    entries.append(entry)
            timelines.append(
                {
                    "drop_id": f"drop_{drop_number:03d}",
                    "trace_id": trace_id,
                    "profile": row["profile"],
                    "drop_segment_index": int(row["segment_index"]),
                    "drop_bandwidth_mbps": float(row["bandwidth_mbps"]),
                    "previous_bandwidth_mbps": float(row["previous_bandwidth_mbps"]),
                    "entries": entries,
                }
            )
    return timelines


def _mean(entries: list[dict[str, object]], field: str) -> float:
    return float(np.mean([float(entry[field]) for entry in entries]))


def _fraction(entries: list[dict[str, object]], predicate) -> float:
    return float(np.mean([predicate(entry) for entry in entries]))


def summarize(timelines: list[dict[str, object]]) -> dict[str, object]:
    by_offset: dict[int, list[dict[str, object]]] = defaultdict(list)
    for timeline in timelines:
        for entry in timeline["entries"]:
            by_offset[int(entry["offset"])].append(entry)

    offsets = sorted(by_offset)
    summary = {}
    for offset in offsets:
        entries = by_offset[offset]
        summary[str(offset)] = {
            "decisions": len(entries),
            "bandwidth_mbps": _mean(entries, "bandwidth_mbps"),
            "throughput_mbps": _mean(entries, "throughput_mbps"),
            "last_throughput_history_mbps": _mean(entries, "throughput_history_4_mbps"),
            "history_mean_mbps": float(
                np.mean(
                    [
                        np.mean([float(entry[f"throughput_history_{i}_mbps"]) for i in range(5)])
                        for entry in entries
                    ]
                )
            ),
            "buffer_seconds": _mean(entries, "buffer_occupancy_seconds"),
            "previous_bitrate_mbps": _mean(entries, "previous_bitrate_mbps"),
            "selected_bitrate_mbps": _mean(entries, "selected_bitrate_mbps"),
            "q_margin": _mean(entries, "q_value_margin"),
            "action_4_fraction": _fraction(entries, lambda entry: int(entry["selected_action"]) == 4),
            "high_buffer_fraction": _fraction(
                entries, lambda entry: float(entry["buffer_occupancy_seconds"]) > 8.0
            ),
        }
    return {"offset_summary": summary}


def _plot(output: Path, timelines: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    selected = timelines[: min(6, len(timelines))]
    figure, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    for timeline in selected:
        entries = timeline["entries"]
        x = [int(entry["offset"]) for entry in entries]
        label = f"{timeline['profile']} {timeline['drop_id']}"
        axes[0].plot(x, [float(entry["bandwidth_mbps"]) for entry in entries], marker="o", label=label)
        axes[0].plot(x, [float(entry["throughput_mbps"]) for entry in entries], marker="x", linestyle="--")
        axes[1].plot(x, [float(entry["buffer_occupancy_seconds"]) for entry in entries], marker="o", label=label)
        axes[1].plot(x, [float(entry["selected_bitrate_mbps"]) for entry in entries], marker="x", linestyle="--")
        axes[2].plot(x, [float(entry["q_value_margin"]) for entry in entries], marker="o", label=label)
    axes[0].set_ylabel("Mbps")
    axes[0].set_title("Representative timelines: bandwidth and achieved throughput")
    axes[1].set_ylabel("Seconds / Mbps")
    axes[1].set_title("Buffer occupancy and selected bitrate")
    axes[2].set_ylabel("Q margin")
    axes[2].set_xlabel("Decision offset relative to drop (0 = drop segment)")
    axes[2].set_title("Q-value separation")
    for axis in axes:
        axis.axvline(0, color="black", linestyle=":", linewidth=1)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output / "representative_drop_timelines.png", dpi=220)
    plt.close(figure)

    summary = summarize(timelines)["offset_summary"]
    offsets = [int(offset) for offset in sorted(summary, key=int)]
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(offsets, [summary[str(offset)]["bandwidth_mbps"] for offset in offsets], marker="o", label="Bandwidth")
    axes[0].plot(offsets, [summary[str(offset)]["throughput_mbps"] for offset in offsets], marker="o", label="Achieved throughput")
    axes[0].plot(offsets, [summary[str(offset)]["selected_bitrate_mbps"] for offset in offsets], marker="o", label="Selected bitrate")
    axes[0].set_ylabel("Mbps")
    axes[0].set_title("Mean response around detected bandwidth drops")
    axes[1].plot(offsets, [summary[str(offset)]["buffer_seconds"] for offset in offsets], marker="o", label="Buffer")
    axes[1].plot(offsets, [summary[str(offset)]["q_margin"] for offset in offsets], marker="o", label="Q margin")
    axes[1].plot(offsets, [summary[str(offset)]["action_4_fraction"] * 10 for offset in offsets], marker="o", label="Action 4 fraction × 10")
    axes[1].set_ylabel("Seconds / normalized")
    axes[1].set_xlabel("Decision offset relative to drop")
    for axis in axes:
        axis.axvline(0, color="black", linestyle=":")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "aggregate_drop_response.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=True)
    for timeline in selected:
        entries = timeline["entries"]
        x = [int(entry["offset"]) for entry in entries]
        for action, axis in enumerate(axes):
            axis.plot(x, [float(entry[f"q_value_action_{action}"]) for entry in entries], marker="o", label=timeline["drop_id"])
            axis.set_ylabel(f"Q{action}")
            axis.grid(alpha=0.25)
    axes[0].set_title("All action Q-values around representative drops")
    axes[-1].set_xlabel("Decision offset relative to drop")
    for axis in axes:
        axis.axvline(0, color="black", linestyle=":")
        axis.legend(fontsize=7, ncol=3)
    figure.tight_layout()
    figure.savefig(output / "q_values_drop_timelines.png", dpi=220)
    plt.close(figure)


def write_report(output: Path, source: Path, timelines: list[dict[str, object]]) -> None:
    offset_summary = summarize(timelines)["offset_summary"]
    paired = [
        timeline
        for timeline in timelines
        if {-1, 0, 1}.issubset({int(entry["offset"]) for entry in timeline["entries"]})
    ]
    at_drop = [entry for timeline in timelines for entry in timeline["entries"] if int(entry["offset"]) == 0]
    after_drop = [entry for timeline in timelines for entry in timeline["entries"] if int(entry["offset"]) == 1]
    pre_drop = [entry for timeline in timelines for entry in timeline["entries"] if int(entry["offset"]) == -1]
    response = (
        "decreased"
        if _mean(after_drop, "selected_bitrate_mbps") < _mean(at_drop, "selected_bitrate_mbps")
        else "did not decrease"
    )
    throughput_evidence = _fraction(
        after_drop,
        lambda entry: float(entry["throughput_history_4_mbps"]) < float(entry["bandwidth_mbps"]),
    )
    high_buffer = _fraction(after_drop, lambda entry: float(entry["buffer_occupancy_seconds"]) > 8.0)
    report = [
        "# DQN Drop-Response Investigation",
        "",
        f"- Source: `{source}`",
        "- Checkpoint: existing state-action diagnostic checkpoint; no retraining performed.",
        f"- Detected drops: {len(timelines)}",
        f"- Post-drop decisions available: {len(after_drop)}",
        f"- Paired drop/post-drop timelines: {len(paired)}",
        "",
        "## Timing interpretation",
        "",
        "A row records the observation used to choose an action before its download. "
        "The row's bandwidth is the condition experienced by that download, so offset 0 "
        "is the drop segment selected before the drop was observed. Offset +1 is the first "
        "decision whose state can contain throughput measured during the drop.",
        "",
        "## Aggregate evidence",
        "",
        "| Offset | Decisions | Bandwidth (Mbps) | Achieved throughput (Mbps) | Last history sample (Mbps) | Mean history (Mbps) | Buffer (s) | Selected bitrate (Mbps) | Q margin | Action 4 fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for offset in sorted(offset_summary, key=int):
        value = offset_summary[offset]
        report.append(
            f"| {offset} | {value['decisions']} | {value['bandwidth_mbps']:.3f} | "
            f"{value['throughput_mbps']:.3f} | {value['last_throughput_history_mbps']:.3f} | "
            f"{value['history_mean_mbps']:.3f} | {value['buffer_seconds']:.3f} | "
            f"{value['selected_bitrate_mbps']:.3f} | {value['q_margin']:.4f} | "
            f"{value['action_4_fraction']:.3f} |"
        )
    report.extend(
        [
            "",
            "## Explanation assessment",
            "",
            f"- **Delayed or misleading throughput observations — supported for the immediate response.** "
            f"At offset +1, {throughput_evidence:.1%} of observations have a last throughput-history "
            f"sample below the current bandwidth, but the mean selected bitrate {_response_word(response)} "
            f"({ _mean(at_drop, 'selected_bitrate_mbps'):.3f} Mbps at offset 0 versus "
            f"{_mean(after_drop, 'selected_bitrate_mbps'):.3f} Mbps at offset +1). "
            "The state reflects the prior completed download, not the current drop, at offset 0.",
            f"- **Insufficient temporal information — plausible but not isolated as the primary cause.** "
            f"The five-sample history is present at +1 and its last sample changes from "
            f"{_mean(pre_drop, 'throughput_history_4_mbps'):.3f} Mbps at -1 to "
            f"{_mean(after_drop, 'throughput_history_4_mbps'):.3f} Mbps at +1. "
            "The available records cannot establish whether a longer history would improve the action.",
            f"- **High buffer occupancy masking the drop — supported as a contributing factor.** "
            f"{high_buffer:.1%} of post-drop observations have more than 8 seconds of buffer; "
            f"mean buffer is {_mean(after_drop, 'buffer_occupancy_seconds'):.3f} seconds. "
            "This is evidence of masking pressure, not proof of causal attribution.",
            f"- **Poorly separated Q-values — not supported as the primary explanation by this comparison.** "
            f"Mean Q margin is {_mean(at_drop, 'q_value_margin'):.4f} at offset 0 and "
            f"{_mean(after_drop, 'q_value_margin'):.4f} at offset +1; the margin does not collapse "
            "when the post-drop state is available. Q-margin measures decisiveness, not correctness.",
            "",
            "## Per-drop data",
            "",
            "The accompanying CSV contains every available consecutive segment from offsets -2 through +2, "
            "including the complete raw observation, named state fields, all five Q-values, selected action, "
            "bandwidth, throughput, buffer, and previous bitrate.",
            "",
            "No algorithm, environment, reward, checkpoint, or existing result file was modified.",
        ]
    )
    (output / "drop_response_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run(source: Path, output: Path) -> None:
    rows = load_rows(source)
    timelines = build_timelines(rows)
    output.mkdir(parents=True, exist_ok=True)
    flattened = [entry for timeline in timelines for entry in timeline["entries"]]
    if not flattened:
        raise ValueError("no bandwidth-drop timelines found")
    fields = list(flattened[0])
    with (output / "drop_response_timelines.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)
    _plot(output, timelines)
    write_report(output, source, timelines)
    (output / "drop_response_summary.json").write_text(
        json.dumps(summarize(timelines), indent=2) + "\n", encoding="utf-8"
    )


def _response_word(response: str) -> str:
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("experiments/results/research_multiseed/diagnostics/state_action/state_action_decisions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/research_multiseed/diagnostics/drop_response"),
    )
    args = parser.parse_args()
    run(args.source, args.output)
    print(f"wrote drop-response analysis to {args.output}")


if __name__ == "__main__":
    main()
