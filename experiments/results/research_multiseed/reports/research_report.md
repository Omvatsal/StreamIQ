# StreamIQ Research Validation Report

Dataset manifest: `experiments/results/research_multiseed/datasets_*/manifest.json`

## Experiment scope

- Seeds: 11, 12, 13; training duration: 20 episodes per seed and ablation.
- Six ablations were independently retrained for each seed.
- Tables report mean ± sample standard deviation across the three seeds.
- QoE is independently calculated from bitrate, rebuffering, and bitrate variation.

## Policy results on unseen test traces

| Ablation | Policy | Average bitrate (Mbps) | Rebuffering (seconds) | Bitrate switches | Total variation (Mbps) | Independent QoE |
|---|---|---|---|---|---|---|
| full | DQN | 3.841 ± 0.386 | 1.102 ± 0.099 | 1.333 ± 0.577 | 3.800 ± 0.000 | -2.467 ± 0.722 |
| full | Fixed 240p | 0.300 ± 0.000 | 0.242 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.668 ± 0.000 |
| full | Throughput | 4.917 ± 0.000 | 6.100 ± 0.000 | 0.000 ± 0.000 | 0.417 ± 0.000 | -19.690 ± 0.000 |
| full | Buffer | 1.027 ± 0.000 | 0.242 ± 0.000 | 3.000 ± 0.000 | 1.983 ± 0.000 | -0.933 ± 0.000 |
| without_switch_penalty | DQN | 3.327 ± 0.851 | 1.052 ± 0.437 | 2.000 ± 0.000 | 5.090 ± 2.046 | -3.427 ± 1.564 |
| without_switch_penalty | Fixed 240p | 0.300 ± 0.000 | 0.242 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.668 ± 0.000 |
| without_switch_penalty | Throughput | 4.917 ± 0.000 | 6.100 ± 0.000 | 0.000 ± 0.000 | 0.417 ± 0.000 | -19.690 ± 0.000 |
| without_switch_penalty | Buffer | 1.027 ± 0.000 | 0.242 ± 0.000 | 3.000 ± 0.000 | 1.983 ± 0.000 | -0.933 ± 0.000 |
| without_rebuffer_penalty | DQN | 4.521 ± 0.830 | 5.225 ± 2.700 | 0.667 ± 1.155 | 1.483 ± 2.569 | -17.119 ± 8.686 |
| without_rebuffer_penalty | Fixed 240p | 0.300 ± 0.000 | 0.242 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.668 ± 0.000 |
| without_rebuffer_penalty | Throughput | 4.917 ± 0.000 | 6.100 ± 0.000 | 0.000 ± 0.000 | 0.417 ± 0.000 | -19.690 ± 0.000 |
| without_rebuffer_penalty | Buffer | 1.027 ± 0.000 | 0.242 ± 0.000 | 3.000 ± 0.000 | 1.983 ± 0.000 | -0.933 ± 0.000 |
| without_throughput_history | DQN | 3.818 ± 0.073 | 2.087 ± 0.722 | 1.667 ± 0.577 | 4.541 ± 1.283 | -6.802 ± 2.176 |
| without_throughput_history | Fixed 240p | 0.300 ± 0.000 | 0.242 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.668 ± 0.000 |
| without_throughput_history | Throughput | 3.647 ± 0.000 | 5.957 ± 0.000 | 3.000 ± 0.000 | 3.867 ± 0.000 | -22.114 ± 0.000 |
| without_throughput_history | Buffer | 1.027 ± 0.000 | 0.242 ± 0.000 | 3.000 ± 0.000 | 1.983 ± 0.000 | -0.933 ± 0.000 |
| without_previous_bitrate | DQN | 3.597 ± 0.226 | 1.214 ± 0.499 | 1.667 ± 0.577 | 4.257 ± 0.792 | -3.387 ± 2.044 |
| without_previous_bitrate | Fixed 240p | 0.300 ± 0.000 | 0.242 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.668 ± 0.000 |
| without_previous_bitrate | Throughput | 4.917 ± 0.000 | 6.100 ± 0.000 | 0.000 ± 0.000 | 0.417 ± 0.000 | -19.690 ± 0.000 |
| without_previous_bitrate | Buffer | 1.027 ± 0.000 | 0.242 ± 0.000 | 3.000 ± 0.000 | 1.983 ± 0.000 | -0.933 ± 0.000 |
| high_switch_penalty | DQN | 3.324 ± 0.282 | 0.792 ± 0.211 | 1.000 ± 0.000 | 3.172 ± 0.597 | -1.432 ± 0.826 |
| high_switch_penalty | Fixed 240p | 0.300 ± 0.000 | 0.242 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.668 ± 0.000 |
| high_switch_penalty | Throughput | 4.917 ± 0.000 | 6.100 ± 0.000 | 0.000 ± 0.000 | 0.417 ± 0.000 | -19.690 ± 0.000 |
| high_switch_penalty | Buffer | 1.027 ± 0.000 | 0.242 ± 0.000 | 3.000 ± 0.000 | 1.983 ± 0.000 | -0.933 ± 0.000 |

## DQN ablation summary

| Ablation | Average bitrate (Mbps) | Rebuffering (seconds) | Bitrate switches | Total variation (Mbps) | Independent QoE |
|---|---|---|---|---|---|
| full | 3.841 ± 0.386 | 1.102 ± 0.099 | 1.333 ± 0.577 | 3.800 ± 0.000 | -2.467 ± 0.722 |
| without_switch_penalty | 3.327 ± 0.851 | 1.052 ± 0.437 | 2.000 ± 0.000 | 5.090 ± 2.046 | -3.427 ± 1.564 |
| without_rebuffer_penalty | 4.521 ± 0.830 | 5.225 ± 2.700 | 0.667 ± 1.155 | 1.483 ± 2.569 | -17.119 ± 8.686 |
| without_throughput_history | 3.818 ± 0.073 | 2.087 ± 0.722 | 1.667 ± 0.577 | 4.541 ± 1.283 | -6.802 ± 2.176 |
| without_previous_bitrate | 3.597 ± 0.226 | 1.214 ± 0.499 | 1.667 ± 0.577 | 4.257 ± 0.792 | -3.387 ± 2.044 |
| high_switch_penalty | 3.324 ± 0.282 | 0.792 ± 0.211 | 1.000 ± 0.000 | 3.172 ± 0.597 | -1.432 ± 0.826 |

## Paired QoE comparisons

Each comparison pairs DQN and baseline QoE on the same unseen test traces. Raw p-values come from paired t-tests over those trace-level observations. Adjusted p-values use Holm-Bonferroni correction across all 54 reported comparisons (six ablations × three seeds × three baselines).
A p-value below 0.05 is not treated as conclusive evidence here because the experiment has only three training seeds and 20 episodes per run.

| Seed | Ablation | Baseline | Mean DQN−baseline QoE | Cohen's dz | Raw p | Holm-adjusted p |
|---:|---|---|---:|---:|---:|---:|
| 11 | full | fixed | -2.455 | -0.692 | 0.000201027 | 0.00934804 |
| 11 | full | throughput | 16.567 | 0.653 | 0.000395326 | 0.0142317 |
| 11 | full | buffer | -2.190 | -0.598 | 0.00101468 | 0.0304405 |
| 11 | without_switch_penalty | fixed | -0.956 | -0.450 | 0.0105511 | 0.158267 |
| 11 | without_switch_penalty | throughput | 18.066 | 0.676 | 0.000263823 | 0.0110805 |
| 11 | without_switch_penalty | buffer | -0.691 | -0.307 | 0.0741023 | 0.45737 |
| 11 | without_rebuffer_penalty | fixed | -6.421 | -1.059 | 2.62408e-07 | 1.36452e-05 |
| 11 | without_rebuffer_penalty | throughput | 12.601 | 0.532 | 0.00298698 | 0.0806485 |
| 11 | without_rebuffer_penalty | buffer | -6.156 | -1.001 | 7.54023e-07 | 3.69472e-05 |
| 11 | without_throughput_history | fixed | -7.431 | -0.498 | 0.00511839 | 0.111627 |
| 11 | without_throughput_history | throughput | 14.015 | 1.057 | 2.74948e-07 | 1.40223e-05 |
| 11 | without_throughput_history | buffer | -7.166 | -0.475 | 0.00723361 | 0.122459 |
| 11 | without_previous_bitrate | fixed | -1.048 | -0.435 | 0.0131879 | 0.158267 |
| 11 | without_previous_bitrate | throughput | 17.974 | 0.668 | 0.000304247 | 0.0121699 |
| 11 | without_previous_bitrate | buffer | -0.783 | -0.317 | 0.0653386 | 0.45737 |
| 11 | high_switch_penalty | fixed | -1.314 | -0.518 | 0.00373642 | 0.0934104 |
| 11 | high_switch_penalty | throughput | 17.708 | 0.659 | 0.000358755 | 0.0132739 |
| 11 | high_switch_penalty | buffer | -1.049 | -0.405 | 0.02039 | 0.2039 |
| 12 | full | fixed | -1.026 | -0.341 | 0.0484 | 0.4356 |
| 12 | full | throughput | 17.996 | 0.673 | 0.000279434 | 0.0114568 |
| 12 | full | buffer | -0.761 | -0.250 | 0.143041 | 0.45737 |
| 12 | without_switch_penalty | fixed | -3.735 | -0.530 | 0.0030901 | 0.0806485 |
| 12 | without_switch_penalty | throughput | 15.287 | 0.684 | 0.000230032 | 0.0101214 |
| 12 | without_switch_penalty | buffer | -3.470 | -0.483 | 0.00640462 | 0.115283 |
| 12 | without_rebuffer_penalty | fixed | -21.466 | -0.649 | 0.000423787 | 0.0148325 |
| 12 | without_rebuffer_penalty | throughput | -2.444 | -0.288 | 0.0929587 | 0.45737 |
| 12 | without_rebuffer_penalty | buffer | -21.201 | -0.638 | 0.000509084 | 0.0167998 |
| 12 | without_throughput_history | fixed | -3.622 | -0.540 | 0.00261618 | 0.073253 |
| 12 | without_throughput_history | throughput | 17.824 | 0.841 | 1.39847e-05 | 0.000671265 |
| 12 | without_throughput_history | buffer | -3.357 | -0.491 | 0.0056918 | 0.113268 |
| 12 | without_previous_bitrate | fixed | -2.110 | -1.302 | 3.55214e-09 | 1.91816e-07 |
| 12 | without_previous_bitrate | throughput | 16.911 | 0.613 | 0.000778218 | 0.0241248 |
| 12 | without_previous_bitrate | buffer | -1.846 | -1.100 | 1.27323e-07 | 6.74812e-06 |
| 12 | high_switch_penalty | fixed | 0.186 | 0.124 | 0.46054 | 0.46054 |
| 12 | high_switch_penalty | throughput | 19.208 | 0.693 | 0.000198894 | 0.00934804 |
| 12 | high_switch_penalty | buffer | 0.451 | 0.291 | 0.0895403 | 0.45737 |
| 13 | full | fixed | -1.914 | -0.502 | 0.00481535 | 0.110753 |
| 13 | full | throughput | 17.108 | 0.681 | 0.000242097 | 0.0104102 |
| 13 | full | buffer | -1.649 | -0.420 | 0.0165685 | 0.182254 |
| 13 | without_switch_penalty | fixed | -3.587 | -0.491 | 0.0056634 | 0.113268 |
| 13 | without_switch_penalty | throughput | 15.435 | 0.686 | 0.0002248 | 0.010116 |
| 13 | without_switch_penalty | buffer | -3.322 | -0.448 | 0.0109432 | 0.158267 |
| 13 | without_rebuffer_penalty | fixed | -21.466 | -0.649 | 0.000423787 | 0.0148325 |
| 13 | without_rebuffer_penalty | throughput | -2.444 | -0.288 | 0.0929587 | 0.45737 |
| 13 | without_rebuffer_penalty | buffer | -21.201 | -0.638 | 0.000509084 | 0.0167998 |
| 13 | without_throughput_history | fixed | -7.347 | -0.498 | 0.00507396 | 0.111627 |
| 13 | without_throughput_history | throughput | 14.099 | 1.051 | 3.08271e-07 | 1.54135e-05 |
| 13 | without_throughput_history | buffer | -7.083 | -0.476 | 0.00720346 | 0.122459 |
| 13 | without_previous_bitrate | fixed | -4.997 | -0.543 | 0.00249088 | 0.0722356 |
| 13 | without_previous_bitrate | throughput | 14.024 | 0.664 | 0.000324909 | 0.0126714 |
| 13 | without_previous_bitrate | buffer | -4.733 | -0.507 | 0.00445332 | 0.10688 |
| 13 | high_switch_penalty | fixed | -1.163 | -0.449 | 0.0107483 | 0.158267 |
| 13 | high_switch_penalty | throughput | 17.858 | 0.662 | 0.000337576 | 0.0128279 |
| 13 | high_switch_penalty | buffer | -0.899 | -0.340 | 0.048784 | 0.4356 |

## Limitations

- The run is smoke-scale: 20 training episodes and three independent seeds.
- Results support descriptive comparisons, not claims of convergence or production superiority.
- The paired tests use trace-level pairing within each seed/ablation; they do not replace a hierarchical analysis across training seeds.
- Holm correction controls family-wise error for the 54 reported paired QoE comparisons, but no inferential tests were performed for the other metrics.

## Visualization index

Generated from actual multi-seed results for seeds 11, 12, 13.
PNG files were checked for existence, non-zero size, and readable image dimensions.

- `plots/ablations/reward_and_state_ablations.png`
- `plots/baselines/average_bitrate.png`
- `plots/baselines/qoe_comparison.png`
- `plots/baselines/rebuffering_duration.png`
- `plots/baselines/switching_frequency.png`
- `plots/baselines/total_variation.png`
- `plots/dqn_average_bitrate_mbps.png`
- `plots/dqn_rebuffer_seconds.png`
- `plots/dqn_total_bitrate_variation_mbps.png`
- `plots/streaming/bandwidth_vs_bitrate.png`
- `plots/streaming/buffer_occupancy.png`
- `plots/streaming/representative_network_and_buffer.png`
- `plots/streaming/throughput_over_time.png`
- `plots/training/11/full.png`
- `plots/training/11/high_switch_penalty.png`
- `plots/training/11/without_previous_bitrate.png`
- `plots/training/11/without_rebuffer_penalty.png`
- `plots/training/11/without_switch_penalty.png`
- `plots/training/11/without_throughput_history.png`
- `plots/training/12/full.png`
- `plots/training/12/high_switch_penalty.png`
- `plots/training/12/without_previous_bitrate.png`
- `plots/training/12/without_rebuffer_penalty.png`
- `plots/training/12/without_switch_penalty.png`
- `plots/training/12/without_throughput_history.png`
- `plots/training/13/full.png`
- `plots/training/13/high_switch_penalty.png`
- `plots/training/13/without_previous_bitrate.png`
- `plots/training/13/without_rebuffer_penalty.png`
- `plots/training/13/without_switch_penalty.png`
- `plots/training/13/without_throughput_history.png`
- `plots/training/epsilon_and_validation.png`
- `plots/training/reward_and_loss.png`
