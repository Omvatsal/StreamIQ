from experiments.configurations.datasets import DatasetConfig, generate_dataset
from rl_engine.evaluation.statistics import paired_statistics


def test_dataset_splits_are_disjoint_and_complete(tmp_path):
    datasets = generate_dataset(
        tmp_path,
        DatasetConfig(train_seeds=(1,), validation_seeds=(2,), test_seeds=(3,), duration_seconds=5),
    )
    ids = {trace.trace_id for traces in datasets.values() for trace in traces}
    assert len(ids) == 18
    assert len(datasets["training"]) == 6
    assert (tmp_path / "manifest.json").exists()


def test_paired_statistics_reports_ci_and_effect():
    result = paired_statistics([2, 4, 6], [1, 2, 3])
    assert result.mean_difference == 2
    assert result.confidence_interval_low < result.mean_difference < result.confidence_interval_high
    assert result.cohens_dz > 0
