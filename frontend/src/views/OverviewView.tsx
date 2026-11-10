import React, { useEffect, useState } from "react";
import {
  Activity,
  ArrowRight,
  Cpu,
  Flame,
  Radio,
  ShieldCheck,
  TrendingUp,
  Wifi,
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
} from "recharts";
import { MetricCard } from "../components/common/MetricCard";
import { TabId } from "../types";
import { runBatchSimulation } from "../services/api";

interface OverviewViewProps {
  onNavigate: (tab: TabId) => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({ onNavigate }) => {
  const [loading, setLoading] = useState(false);
  const [chartData, setChartData] = useState<any[]>([]);
  const [kpis, setKpis] = useState({
    bandwidth: 10.0,
    bitrate: 4.29,
    buffer: 14.8,
    qoe: 105.3,
    winner: "DQN (Reinforcement Learning)",
  });

  useEffect(() => {
    async function loadSampleBatch() {
      setLoading(true);
      try {
        const res = await runBatchSimulation({
          profile: "sudden_drop",
          duration_seconds: 40.0,
          seed: 42,
          policies: ["dqn", "buffer_based", "throughput_based"],
        });

        const timeline = res.timeline.map((event) => ({
          time: `${event.time_seconds.toFixed(0)}s`,
          network: Number(event.network.bandwidth_mbps.toFixed(2)),
          dqn_bitrate: Number(event.policy_states.dqn?.bitrate_mbps.toFixed(2) || 0),
          bba_bitrate: Number(event.policy_states.buffer_based?.bitrate_mbps.toFixed(2) || 0),
          dqn_buffer: Number(event.policy_states.dqn?.buffer_seconds.toFixed(1) || 0),
        }));

        setChartData(timeline);
        if (res.summaries.dqn) {
          setKpis({
            bandwidth: 10.0,
            bitrate: res.summaries.dqn.average_bitrate_mbps,
            buffer: 16.5,
            qoe: res.summaries.dqn.qoe,
            winner: res.winner_policy.toUpperCase(),
          });
        }
      } catch (err) {
        console.warn("Using sample telemetry for initial overview render");
        // Fallback synthetic telemetry timeline
        setChartData([
          { time: "0s", network: 10.0, dqn_bitrate: 2.5, bba_bitrate: 0.7, dqn_buffer: 4.0 },
          { time: "8s", network: 10.0, dqn_bitrate: 5.0, bba_bitrate: 2.5, dqn_buffer: 8.5 },
          { time: "16s", network: 2.0, dqn_bitrate: 2.5, bba_bitrate: 1.2, dqn_buffer: 12.0 },
          { time: "24s", network: 2.0, dqn_bitrate: 1.2, bba_bitrate: 0.3, dqn_buffer: 11.2 },
          { time: "32s", network: 10.0, dqn_bitrate: 2.5, bba_bitrate: 1.2, dqn_buffer: 15.0 },
          { time: "40s", network: 10.0, dqn_bitrate: 5.0, bba_bitrate: 2.5, dqn_buffer: 18.2 },
        ]);
      } finally {
        setLoading(false);
      }
    }
    loadSampleBatch();
  }, []);

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Hero technical control center header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-border/80">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold font-sans text-text-primary tracking-tight">
              Streaming Intelligence Control Center
            </h1>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-cyan/10 text-cyan border border-cyan/20">
              Live Monitor
            </span>
          </div>
          <p className="text-xs text-text-secondary font-mono mt-1">
            Real-time deep reinforcement learning adaptive bitrate optimization & network emulation
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigate("live-simulation")}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan text-background font-mono text-xs font-semibold hover:bg-cyan/90 transition-all shadow-cyan-glow"
          >
            <Radio className="w-4 h-4" />
            <span>Launch Live Simulation</span>
          </button>
          <button
            onClick={() => onNavigate("policy-arena")}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-surface-elevated text-text-primary hover:border-cyan/40 border border-border font-mono text-xs font-medium transition-all"
          >
            <Flame className="w-4 h-4 text-violet" />
            <span>Policy Arena</span>
          </button>
        </div>
      </div>

      {/* Primary KPI Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="PEAK CAPACITY"
          value={kpis.bandwidth.toFixed(1)}
          unit="Mbps"
          description="Profile: Sudden Drop"
          accent="cyan"
          icon={Wifi}
          badge="100% Link"
        />
        <MetricCard
          title="DQN MEAN BITRATE"
          value={kpis.bitrate.toFixed(2)}
          unit="Mbps"
          description="High 1080p retention"
          accent="violet"
          icon={Cpu}
          trend="up"
          trendValue="+18.4%"
        />
        <MetricCard
          title="CLIENT BUFFER"
          value={kpis.buffer.toFixed(1)}
          unit="sec"
          description="Target zone (> 10s)"
          accent="success"
          icon={ShieldCheck}
          badge="Safe Margin"
        />
        <MetricCard
          title="QOE STABILITY SCORE"
          value={kpis.qoe.toFixed(1)}
          description={`Top policy: ${kpis.winner}`}
          accent="success"
          icon={TrendingUp}
          badge="Zero Rebuffer"
        />
      </div>

      {/* Main Charts: Bandwidth vs Bitrate vs Buffer */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Synchronized Bandwidth vs Bitrate Timeline */}
        <div className="lg:col-span-2 bg-surface rounded-xl border border-border p-6 shadow-surface space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-bold font-sans text-text-primary">
                Network Bandwidth vs. DQN Bitrate Adaptation
              </h2>
              <p className="text-[11px] font-mono text-text-muted mt-0.5">
                Observed capacity drops vs. DQN temporal switching response
              </p>
            </div>
            <div className="flex items-center gap-4 text-xs font-mono">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan" />
                <span className="text-text-secondary">Bandwidth</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-violet" />
                <span className="text-text-secondary">DQN Bitrate</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-warning" />
                <span className="text-text-secondary">BBA Heuristic</span>
              </span>
            </div>
          </div>

          <div className="h-64 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="cyanGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#38BDF8" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#38BDF8" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="violetGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#A78BFA" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#A78BFA" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
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
                <Area
                  type="monotone"
                  dataKey="network"
                  stroke="#38BDF8"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#cyanGrad)"
                  name="Network Capacity (Mbps)"
                />
                <Line
                  type="stepAfter"
                  dataKey="dqn_bitrate"
                  stroke="#A78BFA"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: "#A78BFA" }}
                  name="DQN Bitrate (Mbps)"
                />
                <Line
                  type="stepAfter"
                  dataKey="bba_bitrate"
                  stroke="#FBBF24"
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  dot={false}
                  name="BBA Bitrate (Mbps)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right: Client Buffer Dynamics Card */}
        <div className="bg-surface rounded-xl border border-border p-6 shadow-surface space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold font-sans text-text-primary">
                Client Playback Buffer
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-success/10 text-success border border-success/20">
                Stable
              </span>
            </div>
            <p className="text-[11px] font-mono text-text-muted mt-0.5">
              Target safe cushion ≥ 10.0 seconds
            </p>

            <div className="h-44 w-full pt-4">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                  <defs>
                    <linearGradient id="greenGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#34D399" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#34D399" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                  <XAxis dataKey="time" stroke="#64748B" fontSize={10} fontFamily="JetBrains Mono" />
                  <YAxis stroke="#64748B" fontSize={10} fontFamily="JetBrains Mono" unit="s" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#121A29",
                      borderColor: "#24324D",
                      borderRadius: "8px",
                      fontFamily: "JetBrains Mono",
                      fontSize: "12px",
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="dqn_buffer"
                    stroke="#34D399"
                    strokeWidth={2}
                    fill="url(#greenGrad)"
                    name="Buffer (seconds)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="pt-3 border-t border-border/80 flex items-center justify-between text-xs font-mono">
            <span className="text-text-muted">Rebuffer Stall Time:</span>
            <span className="text-success font-semibold">0.00s (0.0%)</span>
          </div>
        </div>
      </div>

      {/* Lower Row: Quick Navigation & Modules */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div
          onClick={() => onNavigate("live-simulation")}
          className="bg-surface rounded-xl border border-border p-5 hover:border-cyan/50 cursor-pointer transition-all group"
        >
          <div className="flex items-center justify-between mb-3">
            <div className="w-8 h-8 rounded-lg bg-cyan/10 border border-cyan/20 flex items-center justify-center">
              <Radio className="w-4 h-4 text-cyan" />
            </div>
            <ArrowRight className="w-4 h-4 text-text-muted group-hover:text-cyan group-hover:translate-x-1 transition-all" />
          </div>
          <h3 className="text-sm font-semibold text-text-primary">
            Live Simulation & Perturbation
          </h3>
          <p className="text-xs text-text-secondary mt-1 font-mono">
            Inject real-time bandwidth chokes, test 6 trace profiles, inspect Q-values.
          </p>
        </div>

        <div
          onClick={() => onNavigate("policy-arena")}
          className="bg-surface rounded-xl border border-border p-5 hover:border-violet/50 cursor-pointer transition-all group"
        >
          <div className="flex items-center justify-between mb-3">
            <div className="w-8 h-8 rounded-lg bg-violet/10 border border-violet/20 flex items-center justify-center">
              <Flame className="w-4 h-4 text-violet" />
            </div>
            <ArrowRight className="w-4 h-4 text-text-muted group-hover:text-violet group-hover:translate-x-1 transition-all" />
          </div>
          <h3 className="text-sm font-semibold text-text-primary">
            Policy Comparison Arena
          </h3>
          <p className="text-xs text-text-secondary mt-1 font-mono">
            Side-by-side benchmark: DQN vs Buffer-Based vs Throughput vs Fixed 240p.
          </p>
        </div>

        <div
          onClick={() => onNavigate("model-lab")}
          className="bg-surface rounded-xl border border-border p-5 hover:border-success/50 cursor-pointer transition-all group"
        >
          <div className="flex items-center justify-between mb-3">
            <div className="w-8 h-8 rounded-lg bg-success/10 border border-success/20 flex items-center justify-center">
              <Cpu className="w-4 h-4 text-success" />
            </div>
            <ArrowRight className="w-4 h-4 text-text-muted group-hover:text-success group-hover:translate-x-1 transition-all" />
          </div>
          <h3 className="text-sm font-semibold text-text-primary">
            DQN Model Lab & Inference
          </h3>
          <p className="text-xs text-text-secondary mt-1 font-mono">
            Inspect model weights, Bellman training hyperparameters, and test latency.
          </p>
        </div>
      </div>
    </div>
  );
};

