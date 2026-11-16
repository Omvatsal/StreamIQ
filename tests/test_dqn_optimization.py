from pathlib import Path

from experiments.dqn_optimization import CONFIGS, ci95


def test_optimization_sweep_is_explicit_and_runs_longer_than_smoke_baseline():
    assert len(CONFIGS) == 4
    assert CONFIGS[0].episodes == 20
    assert all(config.episodes >= 20 for config in CONFIGS)
    assert {config.name for config in CONFIGS} == {
        "baseline_20ep",
        "lower_lr_slow_decay",
        "discounted_stable",
        "larger_batch_slow_target",
    }


def test_ci95_returns_mean_sd_and_half_width():
    mean, sd, half_width = ci95([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert sd > 0
    assert half_width > sd
