import React, { useEffect, useState } from "react";
import {
  Download,
  Flame,
  Globe,
  Layers,
  Radio,
  RefreshCw,
  Sliders,
  Wifi,
} from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { fetchProfiles } from "../services/api";
import { NetworkCondition, NetworkProfile, TabId } from "../types";

interface NetworkLabViewProps {
  onNavigateToSim: (profile: string) => void;
}

export const NetworkLabView: React.FC<NetworkLabViewProps> = ({ onNavigateToSim }) => {
  const [profiles, setProfiles] = useState<NetworkProfile[]>([]);
  const [activeProfile, setActiveProfile] = useState<string>("sudden_drop");
  const [duration, setDuration] = useState<number>(60);
  const [bandwidth, setBandwidth] = useState<number>(10.0);
  const [latency, setLatency] = useState<number>(5.0);
  const [packetLoss, setPacketLoss] = useState<number>(0.0);
  const [seed, setSeed] = useState<number>(42);

  const [conditions, setConditions] = useState<NetworkCondition[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    fetchProfiles().then(setProfiles);
  }, []);

  const generateTraceConditions = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/traces/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile: activeProfile,
          duration_seconds: duration,
          interval_seconds: 1.0,
          bandwidth_mbps: bandwidth,
          delay_ms: latency,
          seed,
        }),
      });
      const data = await res.json();
      setConditions(data.conditions || []);
    } catch (e) {
      console.warn("Using synthetic generator fallback");
      const synthetic: NetworkCondition[] = [];
      for (let t = 0; t <= duration; t += 2) {
        let bw = bandwidth;
        if (activeProfile === "sudden_drop" && t >= 20 && t <= 45) bw = 2.0;
        if (activeProfile === "oscillating") bw = bandwidth * (0.6 + 0.4 * Math.sin(t / 4));
        synthetic.push({
          timestamp_seconds: t,
          bandwidth_mbps: Math.max(0.5, bw),
          delay_ms: latency,
          packet_loss_percent: packetLoss,
        });
      }
      setConditions(synthetic);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    generateTraceConditions();
  }, [activeProfile, duration, bandwidth, latency, seed]);

  const handleExportJSON = () => {
    const traceJson = JSON.stringify(
      {
        profile: activeProfile,
        duration_seconds: duration,
        seed,
        conditions,
      },
      null,
      2
    );
    const blob = new Blob([traceJson], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `streamiq_trace_${activeProfile}_${seed}.json`;
    a.click();
  };

  const chartData = conditions.map((c) => ({
    time: `${c.timestamp_seconds.toFixed(0)}s`,
    bandwidth: Number(c.bandwidth_mbps.toFixed(2)),
    delay: Number(c.delay_ms.toFixed(1)),
    loss: Number(c.packet_loss_percent.toFixed(1)),
  }));

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-border/80">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold font-sans text-text-primary tracking-tight">
              Network Emulation Lab
            </h1>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-cyan/10 text-cyan border border-cyan/20 font-semibold">
              Trace Studio
            </span>
          </div>
          <p className="text-xs text-text-secondary font-mono mt-1">
            Deterministic synthetic link generator and Mininet traffic-control configuration
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleExportJSON}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-surface-elevated text-text-secondary hover:text-text-primary border border-border text-xs font-mono font-medium transition-all"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export Trace JSON</span>
          </button>
          <button
            onClick={() => onNavigateToSim(activeProfile)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan text-background font-mono text-xs font-semibold hover:bg-cyan/90 transition-all shadow-cyan-glow"
          >
            <Radio className="w-4 h-4" />
            <span>Test in Live Sim</span>
          </button>
        </div>
      </div>

      {/* Six Predefined Profiles Cards */}
      <div className="space-y-3">
        <span className="text-xs font-mono font-bold text-text-muted uppercase tracking-wider block">
          Preset Network Profiles (6 Profiles)
        </span>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {profiles.map((p) => {
            const isSelected = activeProfile === p.id;
            return (
              <div
                key={p.id}
                onClick={() => setActiveProfile(p.id)}
                className={`p-4 rounded-xl border cursor-pointer transition-all ${
                  isSelected
                    ? "bg-surface-elevated border-cyan/50 shadow-cyan-glow"
                    : "bg-surface border-border hover:border-border-highlight"
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-mono text-xs font-bold text-text-primary">
                    {p.name}
                  </span>
                  <span
                    className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${
                      p.difficulty === "Easy"
                        ? "text-success bg-success/10 border-success/20"
                        : p.difficulty === "Extreme"
                        ? "text-critical bg-critical/10 border-critical/20"
                        : "text-warning bg-warning/10 border-warning/20"
                    }`}
                  >
                    {p.difficulty || "Standard"}
                  </span>
                </div>
                <p className="text-[11px] font-mono text-text-muted line-clamp-2">
                  {p.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Trace Controls and Interactive Graph */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Parameter Sliders (4 cols) */}
        <div className="lg:col-span-4 bg-surface rounded-xl border border-border p-5 space-y-4 shadow-surface">
          <div className="flex items-center justify-between pb-2 border-b border-border/60">
            <span className="text-xs font-mono font-bold text-text-primary uppercase tracking-wider flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-cyan" />
              <span>Link Parameters</span>
            </span>
            <span className="text-[10px] font-mono text-text-muted">tc config</span>
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-text-secondary">Base Bandwidth</span>
                <span className="text-cyan font-bold">{bandwidth.toFixed(1)} Mbps</span>
              </div>
              <input
                type="range"
                min={1.0}
                max={25.0}
                step={0.5}
                value={bandwidth}
                onChange={(e) => setBandwidth(Number(e.target.value))}
                className="w-full accent-cyan"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-text-secondary">Link Delay (RTT)</span>
                <span className="text-violet font-bold">{latency.toFixed(0)} ms</span>
              </div>
              <input
                type="range"
                min={2}
                max={100}
                step={1}
                value={latency}
                onChange={(e) => setLatency(Number(e.target.value))}
                className="w-full accent-violet"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-text-secondary">Trace Duration</span>
                <span className="text-text-primary font-bold">{duration} seconds</span>
              </div>
              <input
                type="range"
                min={20}
                max={180}
                step={10}
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
                className="w-full accent-cyan"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-text-secondary">RNG Seed</span>
                <span className="text-text-primary font-bold">{seed}</span>
              </div>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="w-full bg-surface-elevated border border-border rounded-lg px-3 py-1.5 text-xs font-mono text-text-primary"
              />
            </div>

            <button
              onClick={generateTraceConditions}
              className="w-full py-2.5 rounded-lg bg-surface-elevated border border-border hover:border-cyan text-xs font-mono text-text-primary flex items-center justify-center gap-2 transition-all"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
              <span>Regenerate Trace</span>
            </button>
          </div>
        </div>

        {/* Right: Timeline Visualization (8 cols) */}
        <div className="lg:col-span-8 bg-surface rounded-xl border border-border p-6 shadow-surface space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-bold font-sans text-text-primary">
                Network Link Capacity Timeline
              </h2>
              <p className="text-[11px] font-mono text-text-muted mt-0.5">
                Deterministic condition series evaluated by ABREnvironment
              </p>
            </div>
            <span className="text-xs font-mono text-cyan">
              {conditions.length} points @ 1.0s interval
            </span>
          </div>

          <div className="h-72 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="cyanArea" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#38BDF8" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#38BDF8" stopOpacity={0.0} />
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
                  dataKey="bandwidth"
                  stroke="#38BDF8"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#cyanArea)"
                  name="Bandwidth (Mbps)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="p-3 bg-surface-elevated/50 rounded-lg border border-border/80 text-xs font-mono flex items-center justify-between">
            <span className="text-text-muted">Mininet CLI equivalent:</span>
            <code className="text-cyan text-[11px]">
              sudo python -m network_lab.experiments.run --profile {activeProfile} --duration {duration} --seed {seed}
            </code>
          </div>
        </div>
      </div>
    </div>
  );
};

