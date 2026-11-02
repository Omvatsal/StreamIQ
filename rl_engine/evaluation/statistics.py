"""Paired descriptive and inferential statistics for policy comparisons."""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PairedStatistics:
    mean_difference: float
    standard_deviation: float
    confidence_interval_low: float
    confidence_interval_high: float
    cohens_dz: float
    p_value: float | None


def paired_statistics(
    first: Sequence[float],
    second: Sequence[float],
    confidence: float = 0.95,
) -> PairedStatistics:
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    if differences.size == 0 or len(first) != len(second):
        raise ValueError("paired samples must have equal non-zero lengths")
    mean = float(np.mean(differences))
    standard_deviation = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(len(differences))
    try:
        from scipy.stats import t, ttest_rel
        critical = float(t.ppf((1 + confidence) / 2, len(differences) - 1))
        p_value = float(ttest_rel(first, second).pvalue)
    except (ImportError, ValueError):
        critical = 1.96
        p_value = None
    effect = mean / standard_deviation if standard_deviation > 0 else 0.0
    return PairedStatistics(
        mean,
        standard_deviation,
        mean - critical * standard_error,
        mean + critical * standard_error,
        effect,
        p_value,
    )
