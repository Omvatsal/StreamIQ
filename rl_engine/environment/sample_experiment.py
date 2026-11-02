"""Run a deterministic trace-backed ABR sample without Mininet."""

import argparse
from pathlib import Path

from network_lab.traces.models import NetworkTrace
from network_lab.traces.generator import PROFILE_NAMES, generate_trace
from network_lab.traces.models import TraceConfig

from .abr_env import ABREnvironment
from .visualize import plot_episode


def run(profile: str, duration: float, seed: int, output_dir: Path) -> None:
    trace = generate_trace(profile, f"abr_{profile}_{seed}", TraceConfig(duration, seed=seed))
    trace_path = output_dir / f"{trace.trace_id}.json"
    trace.save(trace_path)
    environment = ABREnvironment(trace)
    environment.reset(seed=seed)
    while True:
        _, _, terminated, truncated, _ = environment.step(0)
        if terminated or truncated:
            break
    environment.metrics.to_csv(output_dir / f"{trace.trace_id}.csv")
    plot_episode(environment.metrics, output_dir / f"{trace.trace_id}.png")
    print(f"trace={trace_path}")
    print(f"segments={len(environment.metrics.segments)} reward={environment.metrics.accumulated_reward:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILE_NAMES, default="stable")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    run(args.profile, args.duration, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
