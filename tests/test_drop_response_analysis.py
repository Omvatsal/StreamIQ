import json

from experiments.drop_response_analysis import build_timelines, summarize


def test_build_timelines_keeps_consecutive_offsets():
    rows = []
    for index in range(4):
        rows.append(
            {
                "trace_id": "trace",
                "profile": "stable",
                "segment_index": str(index),
                "bandwidth_drop": "True" if index == 2 else "False",
                "bandwidth_mbps": "5",
                "previous_bandwidth_mbps": "10",
                "throughput_mbps": "4",
                "throughput_history_4_mbps": "4",
                "throughput_history_0_mbps": "4",
                "throughput_history_1_mbps": "4",
                "throughput_history_2_mbps": "4",
                "throughput_history_3_mbps": "4",
                "buffer_occupancy_seconds": "10",
                "previous_bitrate_mbps": "2.5",
                "selected_bitrate_mbps": "2.5",
                "selected_action": "4",
                "q_value_margin": "0.2",
                "state_vector": json.dumps([10, 4, 4, 4, 4, 4, 2.5, 1, 5]),
            }
        )

    timelines = build_timelines(rows, radius=1)

    assert [entry["offset"] for entry in timelines[0]["entries"]] == [-1, 0, 1]
    assert timelines[0]["drop_bandwidth_mbps"] == 5.0


def test_summary_computes_action_and_buffer_rates():
    rows = []
    for index, action in enumerate(("4", "4", "2")):
        rows.append(
            {
                "trace_id": "trace",
                "profile": "stable",
                "segment_index": str(index),
                "bandwidth_drop": "True" if index == 1 else "False",
                "bandwidth_mbps": "5",
                "previous_bandwidth_mbps": "10",
                "throughput_mbps": "4",
                "throughput_history_4_mbps": "3",
                    "throughput_history_0_mbps": "3",
                    "throughput_history_1_mbps": "3",
                    "throughput_history_2_mbps": "3",
                    "throughput_history_3_mbps": "3",
                    "buffer_occupancy_seconds": "10",
                "previous_bitrate_mbps": "2.5",
                "selected_bitrate_mbps": "5" if action == "4" else "1.2",
                "selected_action": action,
                "q_value_margin": "0.2",
                "state_vector": json.dumps([10, 3, 3, 3, 3, 3, 2.5, 1, 5]),
            }
        )

    summary = summarize(build_timelines(rows, radius=1))["offset_summary"]

    assert summary["0"]["action_4_fraction"] == 1.0
    assert summary["0"]["high_buffer_fraction"] == 1.0
