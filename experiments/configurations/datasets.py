"""Generate disjoint trace datasets for research experiments."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from network_lab.traces.generator import PROFILE_NAMES, generate_trace
from network_lab.traces.models import NetworkTrace, TraceConfig


@dataclass(frozen=True)
class DatasetConfig:
    train_seeds: tuple[int, ...] = (101, 102, 103, 104, 105, 106)
    validation_seeds: tuple[int, ...] = (201, 202, 203, 204, 205, 206)
    test_seeds: tuple[int, ...] = (301, 302, 303, 304, 305, 306)
    duration_seconds: float = 60.0
    interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        groups = (self.train_seeds, self.validation_seeds, self.test_seeds)
        if any(len(group) == 0 for group in groups):
            raise ValueError("each dataset split must have at least one seed")
        if len(set().union(*map(set, groups))) != sum(map(len, groups)):
            raise ValueError("dataset seeds must not overlap between splits")


def generate_dataset(root: str | Path, config: DatasetConfig = DatasetConfig()) -> dict[str, list[NetworkTrace]]:
    root = Path(root)
    datasets: dict[str, list[NetworkTrace]] = {}
    for split, seeds in (
        ("training", config.train_seeds),
        ("validation", config.validation_seeds),
        ("test", config.test_seeds),
    ):
        traces = []
        for seed in seeds:
            for profile in PROFILE_NAMES:
                trace = generate_trace(
                    profile,
                    f"{split}_{profile}_{seed}",
                    TraceConfig(
                        config.duration_seconds,
                        config.interval_seconds,
                        seed=seed,
                    ),
                )
                trace.save(root / split / f"{trace.trace_id}.json")
                traces.append(trace)
        datasets[split] = traces
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "splits": {
                    split: [trace.trace_id for trace in traces]
                    for split, traces in datasets.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return datasets
