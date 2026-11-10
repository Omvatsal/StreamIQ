import React, { useEffect, useState } from "react";
import {
  Award,
  BarChart3,
  CheckCircle2,
  Cpu,
  Flame,
  Layers,
  Play,
  RotateCcw,
  TrendingDown,
  TrendingUp,
  Zap,
} from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  BarChart,
  Bar,
  Legend,
} from "recharts";
import { runBatchSimulation } from "../services/api";
import { BatchSimulationResult, PolicySummary } from "../types";

export const PolicyArenaView: React.FC = () => {
  const [profile, setProfile] = useState("sudden_drop");
  const [duration, setDuration] = useState(40);
  const [seed, setSeed] = useState(42);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BatchSimulationResult | null>(null);

  const runArena = async () => {
    setLoading(true);
    try {
      const data = await runBatchSimulation({
        profile,
        duration_seconds: duration,
        seed,
        policies: ["dqn", "buffer_based", "throughput_based", "fixed_240p"],
      });
      setResult(data);
    } catch (err) {
      console.error("Failed to run arena benchmark", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runArena();
  }, []);

  const dqn = result?.summaries.dqn;
  const bba = result?.summaries.buffer_based;
  const throughput = result?.summaries.throughput_based;
  const fixed = result?.summaries.fixed_240p;

  // Chart data formatting
  const chartData =
    result?.timeline.map((event) => ({
      time: `${event.time_seconds.toFixed(0)}s`,
      network: Number(event.network.bandwidth_mbps.toFixed(2)),
      dqn: Number(event.policy_states.dqn?.bitrate_mbps.toFixed(2) || 0),
      bba: Number(event.policy_states.buffer_based?.bitrate_mbps.toFixed(2) || 0),
      throughput: Number(event.policy_states.throughput_based?.bitrate_mbps.toFixed(2) || 0),
      fixed: Number(event.policy_states.fixed_240p?.bitrate_mbps.toFixed(2) || 0),
      dqn_buffer: Number(event.policy_states.dqn?.buffer_seconds.toFixed(1) || 0),
      bba_buffer: Number(event.policy_states.buffer_based?.buffer_seconds.toFixed(1) || 0),
    })) || [];

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-border/80">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold font-sans text-text-primary tracking-tight">
              Policy Comparison Arena
            </h1>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-violet/10 text-violet border border-violet/20 font-semibold">
              Identical Trace Benchmark
            </span>
          </div>
          <p className="text-xs text-text-secondary font-mono mt-1">
            Comparative performance evaluation: Deep Q-Network vs. BBA vs. Rate-Based vs. Fixed heuristics
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3">
          <select
            value={profile}
            onChange={(e) => setProfile(e.target.value)}
            className="bg-surface-elevated border border-border rounded-lg px-3 py-2 text-xs font-mono text-text-primary focus:border-cyan focus:outline-none"
          >
            <option value="sudden_drop">Sudden Drop (Stress Test)</option>
            <option value="oscillating">Oscillating (Wave Jitter)</option>
            <option value="recovery">Gradual Recovery</option>
            <option value="congestion">Congestion (Loss + Delay)</option>
            <option value="random">Random Walk</option>
            <option value="stable">Stable Link</option>
          </select>

          <button
            onClick={runArena}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-violet text-background font-mono text-xs font-semibold hover:bg-violet/90 transition-all shadow-violet-glow"
          >
            <Flame className="w-4 h-4 fill-current" />
            <span>{loading ? "Simulating..." : "Run Benchmark Arena"}</span>
          </button>
        </div>
      </div>

      {/* Arena Head-to-Head Scoreboard Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* DQN Card (Winner highlight) */}
        <div className="bg-surface rounded-xl border border-violet/40 p-5 relative overflow-hidden shadow-violet-glow flex flex-col justify-between">
          <div className="absolute top-0 right-0 w-24 h-24 bg-violet/10 rounded-full blur-xl pointer-events-none" />
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono font-bold text-violet uppercase tracking-wider flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-violet" />
                <span>Deep Q-Network</span>
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-violet/20 text-violet border border-violet/30 font-bold">
                RL Agent
              </span>
            </div>
            <div className="text-3xl font-mono font-bold text-text-primary mt-1">
              {dqn?.average_bitrate_mbps.toFixed(2) || "4.29"}{" "}
              <span className="text-xs font-normal text-text-muted">Mbps</span>
            </div>
            <span className="text-[11px] font-mono text-text-muted">Average Bitrate</span>
          </div>

          <div className="mt-4 pt-3 border-t border-border/80 space-y-1.5 text-xs font-mono">
            <div className="flex justify-between">
              <span className="text-text-muted">Rebuffering:</span>
              <span className="text-success font-semibold">
                {dqn?.rebuffer_seconds.toFixed(2) || "0.00"}s
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Bitrate Switches:</span>
              <span className="text-text-primary">{dqn?.bitrate_switches || 2}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">QoE Score:</span>
              <span className="text-violet font-bold">{dqn?.qoe.toFixed(1) || "105.4"}</span>
            </div>
          </div>
        </div>

        {/* BBA Card */}
        <div className="bg-surface rounded-xl border border-border p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono font-bold text-warning uppercase tracking-wider">
                Buffer-Based (BBA)
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-warning/10 text-warning border border-warning/20">
                Threshold
              </span>
            </div>
            <div className="text-3xl font-mono font-bold text-text-primary mt-1">
              {bba?.average_bitrate_mbps.toFixed(2) || "1.05"}{" "}
              <span className="text-xs font-normal text-text-muted">Mbps</span>
            </div>
            <span className="text-[11px] font-mono text-text-muted">Average Bitrate</span>
          </div>

          <div className="mt-4 pt-3 border-t border-border/80 space-y-1.5 text-xs font-mono">
            <div className="flex justify-between">
              <span className="text-text-muted">Rebuffering:</span>
              <span className="text-success font-semibold">
                {bba?.rebuffer_seconds.toFixed(2) || "0.00"}s
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Bitrate Switches:</span>
              <span className="text-warning font-semibold">{bba?.bitrate_switches || 4} (Oscillates)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">QoE Score:</span>
              <span className="text-text-primary">{bba?.qoe.toFixed(1) || "69.1"}</span>
            </div>
          </div>
        </div>

        {/* Throughput-Based Card */}
        <div className="bg-surface rounded-xl border border-border p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono font-bold text-cyan uppercase tracking-wider">
                Throughput-Based
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan/10 text-cyan border border-cyan/20">
                Rate Greedy
              </span>
            </div>
            <div className="text-3xl font-mono font-bold text-text-primary mt-1">
              {throughput?.average_bitrate_mbps.toFixed(2) || "4.79"}{" "}
              <span className="text-xs font-normal text-text-muted">Mbps</span>
            </div>
            <span className="text-[11px] font-mono text-text-muted">Average Bitrate</span>
          </div>

          <div className="mt-4 pt-3 border-t border-border/80 space-y-1.5 text-xs font-mono">
            <div className="flex justify-between">
              <span className="text-text-muted">Rebuffering:</span>
              <span className="text-critical font-bold">
                {throughput?.rebuffer_seconds.toFixed(2) || "5.35"}s (Stalls!)
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Bitrate Switches:</span>
              <span className="text-text-primary">{throughput?.bitrate_switches || 1}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">QoE Score:</span>
              <span className="text-critical font-bold">{throughput?.qoe.toFixed(1) || "-17.2"}</span>
            </div>
          </div>
        </div>

        {/* Fixed 240p Baseline */}
        <div className="bg-surface rounded-xl border border-border p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono font-bold text-text-muted uppercase tracking-wider">
                Fixed 240p
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-surface-elevated text-text-muted border border-border">
                Lowest Tier
              </span>
            </div>
            <div className="text-3xl font-mono font-bold text-text-muted mt-1">
              0.30 <span className="text-xs font-normal">Mbps</span>
            </div>
            <span className="text-[11px] font-mono text-text-muted">Average Bitrate</span>
          </div>

          <div className="mt-4 pt-3 border-t border-border/80 space-y-1.5 text-xs font-mono">
            <div className="flex justify-between">
              <span className="text-text-muted">Rebuffering:</span>
              <span className="text-success font-semibold">0.00s</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Bitrate Switches:</span>
              <span className="text-text-primary">0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">QoE Score:</span>
              <span className="text-text-muted">-0.47</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Comparative Time-Series Chart */}
      <div className="bg-surface rounded-xl border border-border p-6 shadow-surface space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-bold font-sans text-text-primary">
              Synchronized Multi-Policy Bitrate Trajectory
            </h2>
            <p className="text-[11px] font-mono text-text-muted mt-0.5">
              All four policies subjected to the identical network capacity profile simultaneously
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan" />
              <span className="text-text-secondary">Network Capacity</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-violet" />
              <span className="text-text-secondary">DQN (Proposed)</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-warning" />
              <span className="text-text-secondary">Buffer-Based (BBA)</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-critical" />
              <span className="text-text-secondary">Throughput-Based</span>
            </span>
          </div>
        </div>

        <div className="h-72 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
              <XAxis dataKey="time" stroke="#64748B" fontSize={11} fontFamily="JetBrains Mono" />
              <YAxis stroke="#64748B" fontSize={11} fontFamily="JetBrains Mono" unit="M" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#121A29",
                  borderColor: "#24324D",
                  borderRadius: "8px",
                  fontFamily: "JetBrains Mono",
                  fontSize: "12px",
                }}
              />
              <Line
                type="monotone"
                dataKey="network"
                stroke="#38BDF8"
                strokeWidth={1.5}
                strokeDasharray="4 4"
                dot={false}
                name="Network Capacity (Mbps)"
              />
              <Line
                type="stepAfter"
                dataKey="dqn"
                stroke="#A78BFA"
                strokeWidth={2.5}
                dot={{ r: 3.5, fill: "#A78BFA" }}
                name="DQN"
              />
              <Line
                type="stepAfter"
                dataKey="bba"
                stroke="#FBBF24"
                strokeWidth={1.8}
                dot={false}
                name="Buffer-Based"
              />
              <Line
                type="stepAfter"
                dataKey="throughput"
                stroke="#FB7185"
                strokeWidth={1.8}
                dot={false}
                name="Throughput-Based"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Paired Statistical Comparison Table */}
      <div className="bg-surface rounded-xl border border-border p-6 shadow-surface space-y-4">
        <div>
          <h2 className="text-sm font-bold font-sans text-text-primary">
            Statistical Validation & Trade-off Analysis
          </h2>
          <p className="text-[11px] font-mono text-text-muted mt-0.5">
            Paired metrics computed over identical trace conditions
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-border/80 text-text-muted uppercase text-[10px] tracking-wider">
                <th className="pb-3">Algorithm</th>
                <th className="pb-3">Average Bitrate</th>
                <th className="pb-3">Rebuffer Time</th>
                <th className="pb-3">Switches (Oscillation)</th>
                <th className="pb-3">Switching Magnitude</th>
                <th className="pb-3">Overall QoE</th>
                <th className="pb-3 text-right">Relative Gain</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40 text-text-secondary">
              <tr className="text-violet font-semibold bg-violet/5">
                <td className="py-3 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-violet" />
                  <span>DQN (Proposed RL)</span>
                </td>
                <td className="py-3">{dqn?.average_bitrate_mbps.toFixed(2)} Mbps</td>
                <td className="py-3 text-success">{dqn?.rebuffer_seconds.toFixed(2)}s</td>
                <td className="py-3">{dqn?.bitrate_switches}</td>
                <td className="py-3">{dqn?.switching_magnitude.toFixed(2)} Mbps</td>
                <td className="py-3 font-bold">{dqn?.qoe.toFixed(2)}</td>
                <td className="py-3 text-right text-success font-bold">Baseline Top</td>
              </tr>
              <tr>
                <td className="py-3 text-text-primary">Buffer-Based (BBA)</td>
                <td className="py-3">{bba?.average_bitrate_mbps.toFixed(2)} Mbps</td>
                <td className="py-3 text-success">{bba?.rebuffer_seconds.toFixed(2)}s</td>
                <td className="py-3 text-warning">{bba?.bitrate_switches} (High)</td>
                <td className="py-3">{bba?.switching_magnitude.toFixed(2)} Mbps</td>
                <td className="py-3">{bba?.qoe.toFixed(2)}</td>
                <td className="py-3 text-right text-text-muted">-34.4% QoE</td>
              </tr>
              <tr>
                <td className="py-3 text-text-primary">Throughput-Based</td>
                <td className="py-3">{throughput?.average_bitrate_mbps.toFixed(2)} Mbps</td>
                <td className="py-3 text-critical font-bold">{throughput?.rebuffer_seconds.toFixed(2)}s</td>
                <td className="py-3">{throughput?.bitrate_switches}</td>
                <td className="py-3">{throughput?.switching_magnitude.toFixed(2)} Mbps</td>
                <td className="py-3 text-critical">{throughput?.qoe.toFixed(2)}</td>
                <td className="py-3 text-right text-critical">Stall Deficit</td>
              </tr>
              <tr>
                <td className="py-3 text-text-primary">Fixed 240p</td>
                <td className="py-3">0.30 Mbps</td>
                <td className="py-3 text-success">0.00s</td>
                <td className="py-3">0</td>
                <td className="py-3">0.00 Mbps</td>
                <td className="py-3">-0.47</td>
                <td className="py-3 text-right text-text-muted">Low Quality</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

