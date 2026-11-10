"""Performance benchmarking script for StreamIQ.

Measures:
1. Replay buffer throughput (add and sample ops/sec, memory).
2. ABR environment step throughput (steps/sec, memory allocation).
3. DQN training throughput (transitions/sec, learn steps/sec, total time).
4. Simulation runner batch execution throughput (steps/sec).
"""

import time
import tracemalloc
import numpy as np
import torch

from network_lab.traces.generator import generate_trace
from network_lab.traces.models import TraceConfig
from rl_engine.agents.dqn import DQNAgent
from rl_engine.agents.replay_buffer import ReplayBuffer
from rl_engine.environment.abr_env import ABREnvironment
from rl_engine.simulation.runner import SimulationRunner
from rl_engine.training.config import TrainingConfig
from rl_engine.training.trainer import DQNTrainer


def benchmark_replay_buffer(capacity=50_000, num_adds=50_000, num_samples=5_000, batch_size=64, seed=42):
    tracemalloc.start()
    buf = ReplayBuffer(capacity, seed=seed)
    state = np.ones(9, dtype=np.float32)
    next_state = np.ones(9, dtype=np.float32) * 1.5

    # Measure add
    t0 = time.perf_counter()
    for i in range(num_adds):
        buf.add(state, i % 5, 1.0, next_state, False)
    t_add = time.perf_counter() - t0
    add_rate = num_adds / t_add

    # Measure sample
    t0 = time.perf_counter()
    for _ in range(num_samples):
        batch = buf.sample(batch_size)
    t_sample = time.perf_counter() - t0
    sample_rate = num_samples / t_sample

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "add_rate_ops_per_sec": add_rate,
        "sample_rate_ops_per_sec": sample_rate,
        "add_time_s": t_add,
        "sample_time_s": t_sample,
        "peak_memory_mb": peak / (1024 * 1024),
    }


def benchmark_abr_env(num_steps=20_000, seed=42):
    config = TraceConfig(duration_seconds=120, interval_seconds=1.0, seed=seed)
    trace = generate_trace("oscillating", "bench_trace", config)
    env = ABREnvironment(trace)

    tracemalloc.start()
    t0 = time.perf_counter()
    env.reset(seed=seed)
    steps_done = 0
    while steps_done < num_steps:
        state, _, done, _, _ = env.step(steps_done % 5)
        steps_done += 1
        if done:
            env.reset(seed=seed + steps_done)
    t_env = time.perf_counter() - t0
    step_rate = num_steps / t_env

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "step_rate_steps_per_sec": step_rate,
        "total_time_s": t_env,
        "peak_memory_mb": peak / (1024 * 1024),
    }


def benchmark_dqn_training(episodes=50, seed=42):
    config = TraceConfig(duration_seconds=60, interval_seconds=1.0, seed=seed)
    trace = generate_trace("sudden_drop", "bench_train_trace", config)
    env = ABREnvironment(trace)

    agent = DQNAgent(state_size=9, action_size=5, batch_size=32, replay_capacity=10_000, seed=seed)
    train_config = TrainingConfig(episodes=episodes, seed=seed, validation_every=100)
    trainer = DQNTrainer(agent, train_config)

    tracemalloc.start()
    t0 = time.perf_counter()
    history = trainer.train([env])
    t_train = time.perf_counter() - t0

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_transitions = episodes * env.video.segment_count
    return {
        "episodes": episodes,
        "total_transitions": total_transitions,
        "throughput_transitions_per_sec": total_transitions / t_train,
        "total_time_s": t_train,
        "peak_memory_mb": peak / (1024 * 1024),
    }


def benchmark_simulation_runner(num_runs=100, seed=42):
    config = TraceConfig(duration_seconds=60, interval_seconds=1.0, seed=seed)
    trace = generate_trace("congestion", "bench_sim_trace", config)
    runner = SimulationRunner()

    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        res = runner.run_batch(trace, policy_names=["dqn", "buffer_based", "throughput_based", "fixed_240p"])
    t_sim = time.perf_counter() - t0

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_sim_steps = num_runs * res.segment_count * 4  # 4 policies
    return {
        "num_runs": num_runs,
        "total_sim_steps": total_sim_steps,
        "throughput_steps_per_sec": total_sim_steps / t_sim,
        "total_time_s": t_sim,
        "peak_memory_mb": peak / (1024 * 1024),
    }


if __name__ == "__main__":
    print("=== STREAMIQ PERFORMANCE BENCHMARK (BASELINE) ===")
    
    print("\n1. Replay Buffer Benchmark...")
    rb_res = benchmark_replay_buffer()
    print(f"   Add Rate:    {rb_res['add_rate_ops_per_sec']:,.0f} ops/sec ({rb_res['add_time_s']:.3f}s)")
    print(f"   Sample Rate: {rb_res['sample_rate_ops_per_sec']:,.0f} ops/sec ({rb_res['sample_time_s']:.3f}s)")
    print(f"   Peak Memory: {rb_res['peak_memory_mb']:.2f} MB")

    print("\n2. ABR Environment Step Benchmark...")
    env_res = benchmark_abr_env()
    print(f"   Step Rate:   {env_res['step_rate_steps_per_sec']:,.0f} steps/sec ({env_res['total_time_s']:.3f}s)")
    print(f"   Peak Memory: {env_res['peak_memory_mb']:.2f} MB")

    print("\n3. DQN Training Benchmark (50 episodes)...")
    train_res = benchmark_dqn_training(episodes=50)
    print(f"   Throughput:  {train_res['throughput_transitions_per_sec']:,.0f} transitions/sec")
    print(f"   Total Time:  {train_res['total_time_s']:.3f}s")
    print(f"   Peak Memory: {train_res['peak_memory_mb']:.2f} MB")

    print("\n4. Simulation Runner Benchmark (100 runs x 4 policies)...")
    sim_res = benchmark_simulation_runner(num_runs=100)
    print(f"   Throughput:  {sim_res['throughput_steps_per_sec']:,.0f} policy-steps/sec")
    print(f"   Total Time:  {sim_res['total_time_s']:.3f}s")
    print(f"   Peak Memory: {sim_res['peak_memory_mb']:.2f} MB")

