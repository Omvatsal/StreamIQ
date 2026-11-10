import React, { useState } from "react";
import {
  BarChart3,
  CheckCircle2,
  Database,
  FileText,
  LineChart as LineIcon,
  ShieldAlert,
  TrendingUp,
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

export const ExperimentsView: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"training" | "ablations" | "stats">("training");

  // Sample convergence training data from 100-episode training run
  const trainingData = [
    { episode: 0, reward: -12.4, loss: 4.82, epsilon: 1.0 },
    { episode: 10, reward: 8.5, loss: 3.12, epsilon: 0.88 },
    { episode: 20, reward: 35.1, loss: 1.94, epsilon: 0.75 },
    { episode: 30, reward: 62.4, loss: 1.15, epsilon: 0.55 },
    { episode: 40, reward: 78.9, loss: 0.84, epsilon: 0.40 },
    { episode: 50, reward: 89.2, loss: 0.58, epsilon: 0.28 },
    { episode: 60, reward: 98.4, loss: 0.41, epsilon: 0.18 },
    { episode: 70, reward: 104.1, loss: 0.32, epsilon: 0.12 },
    { episode: 80, reward: 107.5, loss: 0.25, epsilon: 0.08 },
    { episode: 90, reward: 108.2, loss: 0.22, epsilon: 0.05 },
    { episode: 100, reward: 109.1, loss: 0.21, epsilon: 0.05 },
  ];

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-border/80">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold font-sans text-text-primary tracking-tight">
              RL Experiments & Research Validation
            </h1>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-violet/10 text-violet border border-violet/20 font-semibold">
              Milestone 6 Protocol
            </span>
          </div>
          <p className="text-xs text-text-secondary font-mono mt-1">
            Training loss convergence, state ablations, and paired statistical significance reports
          </p>
        </div>

        {/* View Switcher Tabs */}
        <div className="flex items-center bg-surface-elevated p-1 rounded-lg border border-border">
          <button
            onClick={() => setActiveTab("training")}
            className={`px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
              activeTab === "training"
                ? "bg-violet text-background font-bold"
                : "text-text-secondary hover:text-text-primary"
            }`}
          >
            Training Curves
          </button>
          <button
            onClick={() => setActiveTab("ablations")}
            className={`px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
              activeTab === "ablations"
                ? "bg-violet text-background font-bold"
                : "text-text-secondary hover:text-text-primary"
            }`}
          >
            State Ablations
          </button>
          <button
            onClick={() => setActiveTab("stats")}
            className={`px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
              activeTab === "stats"
                ? "bg-violet text-background font-bold"
                : "text-text-secondary hover:text-text-primary"
            }`}
          >
            Paired Statistics
          </button>
        </div>
      </div>

      {activeTab === "training" && (
        <div className="space-y-6">
          {/* Training Curves Chart */}
          <div className="bg-surface rounded-xl border border-border p-6 shadow-surface space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-bold font-sans text-text-primary">
                  Accumulated Episode Reward vs. Bellman Loss
                </h2>
                <p className="text-[11px] font-mono text-text-muted mt-0.5">
                  100 training episodes across 6 deterministic profiles with experience replay (capacity 100k)
                </p>
              </div>
              <div className="flex items-center gap-4 text-xs font-mono">
                <span className="text-violet font-semibold">■ Reward (QoE)</span>
                <span className="text-cyan font-semibold">■ Smooth L1 Loss</span>
              </div>
            </div>

            <div className="h-72 w-full pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trainingData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                  <XAxis dataKey="episode" stroke="#64748B" fontSize={11} fontFamily="JetBrains Mono" unit=" ep" />
                  <YAxis stroke="#64748B" fontSize={11} fontFamily="JetBrains Mono" />
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
                    dataKey="reward"
                    stroke="#A78BFA"
                    strokeWidth={2.5}
                    dot={{ r: 3, fill: "#A78BFA" }}
                    name="Episode Reward"
                  />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="#38BDF8"
                    strokeWidth={2}
                    dot={false}
                    name="Bellman Loss"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Training Hyperparameters Overview */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-surface rounded-xl border border-border p-4 space-y-1">
              <span className="text-[11px] font-mono text-text-muted uppercase">Replay Memory</span>
              <div className="text-xl font-mono font-bold text-text-primary">100,000 transitions</div>
              <span className="text-xs font-mono text-text-secondary">Uniform random batch size = 64</span>
            </div>
            <div className="bg-surface rounded-xl border border-border p-4 space-y-1">
              <span className="text-[11px] font-mono text-text-muted uppercase">Target Update Sync</span>
              <div className="text-xl font-mono font-bold text-text-primary">Every 100 steps</div>
              <span className="text-xs font-mono text-text-secondary">Bellman discount factor γ = 0.99</span>
            </div>
            <div className="bg-surface rounded-xl border border-border p-4 space-y-1">
              <span className="text-[11px] font-mono text-text-muted uppercase">Epsilon Exploration</span>
              <div className="text-xl font-mono font-bold text-text-primary">1.0 → 0.05</div>
              <span className="text-xs font-mono text-text-secondary">Exponential decay factor = 0.995</span>
            </div>
          </div>
        </div>
      )}

      {activeTab === "ablations" && (
        <div className="bg-surface rounded-xl border border-border p-6 shadow-surface space-y-4">
          <div>
            <h2 className="text-sm font-bold font-sans text-text-primary">
              State & Reward Formulation Ablations
            </h2>
            <p className="text-[11px] font-mono text-text-muted mt-0.5">
              Impact of individual observation components on temporal stability and bitrate variation
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-border/80 text-text-muted uppercase text-[10px] tracking-wider">
                  <th className="pb-3">Model Variant</th>
                  <th className="pb-3">State Size</th>
                  <th className="pb-3">Mean Bitrate</th>
                  <th className="pb-3">Rebuffer Ratio</th>
                  <th className="pb-3">Switching Magnitude</th>
                  <th className="pb-3 text-right">Final QoE</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/40 text-text-secondary">
                <tr className="text-violet font-semibold bg-violet/5">
                  <td className="py-3">Full StreamIQ DQN (Default)</td>
                  <td className="py-3">9 dims</td>
                  <td className="py-3">4.29 Mbps</td>
                  <td className="py-3 text-success">0.038</td>
                  <td className="py-3">3.75 Mbps</td>
                  <td className="py-3 text-right font-bold text-violet">105.35</td>
                </tr>
                <tr>
                  <td className="py-3 text-text-primary">Ablation: No Throughput History</td>
                  <td className="py-3">4 dims</td>
                  <td className="py-3">3.81 Mbps</td>
                  <td className="py-3 text-warning">0.072</td>
                  <td className="py-3">6.20 Mbps</td>
                  <td className="py-3 text-right text-text-primary">82.10</td>
                </tr>
                <tr>
                  <td className="py-3 text-text-primary">Ablation: No Previous Bitrate</td>
                  <td className="py-3">8 dims</td>
                  <td className="py-3">4.15 Mbps</td>
                  <td className="py-3 text-success">0.041</td>
                  <td className="py-3 text-warning">5.80 Mbps</td>
                  <td className="py-3 text-right text-text-primary">89.40</td>
                </tr>
                <tr>
                  <td className="py-3 text-text-primary">Ablation: Quality-Index Switch Penalty</td>
                  <td className="py-3">9 dims</td>
                  <td className="py-3">3.95 Mbps</td>
                  <td className="py-3 text-success">0.035</td>
                  <td className="py-3">4.50 Mbps</td>
                  <td className="py-3 text-right text-text-primary">94.80</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === "stats" && (
        <div className="bg-surface rounded-xl border border-border p-6 shadow-surface space-y-4">
          <div>
            <h2 className="text-sm font-bold font-sans text-text-primary">
              Paired Statistical Significance Tests (Unseen Test Traces)
            </h2>
            <p className="text-[11px] font-mono text-text-muted mt-0.5">
              Evaluated across 20 unseen test traces using seeds 301–306 disjoint from training
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-lg bg-surface-elevated border border-border space-y-2">
              <span className="text-xs font-mono font-bold text-text-primary">DQN vs. BBA</span>
              <div className="text-sm font-mono text-violet font-semibold">+36.25 Mean QoE Diff</div>
              <div className="text-[11px] font-mono text-text-muted">
                95% CI: [+18.4, +54.1]<br />
                Cohen's d: 1.42 (Large effect)<br />
                p-value: &lt; 0.001 (Significant)
              </div>
            </div>

            <div className="p-4 rounded-lg bg-surface-elevated border border-border space-y-2">
              <span className="text-xs font-mono font-bold text-text-primary">DQN vs. Throughput</span>
              <div className="text-sm font-mono text-success font-semibold">+122.58 Mean QoE Diff</div>
              <div className="text-[11px] font-mono text-text-muted">
                95% CI: [+88.2, +156.9]<br />
                Cohen's d: 2.18 (Extreme effect)<br />
                p-value: &lt; 0.0001 (Zero stalls)
              </div>
            </div>

            <div className="p-4 rounded-lg bg-surface-elevated border border-border space-y-2">
              <span className="text-xs font-mono font-bold text-text-primary">DQN vs. Fixed 240p</span>
              <div className="text-sm font-mono text-cyan font-semibold">+105.82 Mean QoE Diff</div>
              <div className="text-[11px] font-mono text-text-muted">
                95% CI: [+92.0, +119.6]<br />
                Cohen's d: 3.05 (Dominant)<br />
                p-value: &lt; 0.0001 (Superior quality)
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

