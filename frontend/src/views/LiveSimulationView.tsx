import React, { useState, useEffect, useRef } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  FastForward,
  Gauge,
  Maximize2,
  Pause,
  Play,
  RotateCcw,
  Sliders,
  Tv,
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
  CartesianGrid,
} from "recharts";
import {
  createLiveSimulationSocket,
  fetchProfiles,
  predictBitrate,
  runBatchSimulation,
} from "../services/api";
import { NetworkProfile, SimulationStepEvent } from "../types";

const BITRATES = [0.3, 0.7, 1.2, 2.5, 5.0];
const QUALITY_NAMES = ["240p", "360p", "480p", "720p", "1080p"];

export const LiveSimulationView: React.FC = () => {
  // Profiles & Controls
  const [profiles, setProfiles] = useState<NetworkProfile[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<string>("sudden_drop");
  const [duration, setDuration] = useState<number>(40);
  const [seed, setSeed] = useState<number>(42);
  const [speed, setSpeed] = useState<number>(1);
  const [playbackMode, setPlaybackMode] = useState<"synthetic" | "dash">("synthetic");

  // Manual Overrides (Perturbations)
  const [overrideBandwidth, setOverrideBandwidth] = useState<number | null>(null);
  const [overrideLatency, setOverrideLatency] = useState<number>(5);
  const [overrideLoss, setOverrideLoss] = useState<number>(0);

  // Simulation State
  const [status, setStatus] = useState<"idle" | "running" | "paused" | "completed">("idle");
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [totalSteps, setTotalSteps] = useState<number>(10);
  const [timeline, setTimeline] = useState<SimulationStepEvent[]>([]);
  const [activeEvent, setActiveEvent] = useState<SimulationStepEvent | null>(null);
  const [qValues, setQValues] = useState<number[]>([12.4, 28.5, 54.1, 88.7, 102.3]);

  // WebSocket or Batch ref
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<any>(null);

  useEffect(() => {
    fetchProfiles().then(setProfiles);
  }, []);

  // Handler to start or restart simulation
  const handleStart = async () => {
    setStatus("running");
    setCurrentStep(0);
    setTimeline([]);

    // Query backend for batch simulation to step through smoothly
    try {
      const res = await runBatchSimulation({
        profile: selectedProfile,
        duration_seconds: duration,
        seed: seed,
        policies: ["dqn", "buffer_based", "throughput_based"],
      });

      setTotalSteps(res.timeline.length);
      let stepIndex = 0;

      if (intervalRef.current) clearInterval(intervalRef.current);

      intervalRef.current = setInterval(() => {
        if (stepIndex < res.timeline.length) {
          const stepEvent = res.timeline[stepIndex];
          setActiveEvent(stepEvent);
          setTimeline((prev) => [...prev, stepEvent]);
          setCurrentStep(stepIndex + 1);

          // Update Q-values dynamically from RL predictor
          const dqnState = stepEvent.policy_states.dqn;
          if (dqnState) {
            predictBitrate({
              buffer_occupancy_seconds: dqnState.buffer_seconds,
              throughput_history_mbps: [dqnState.throughput_mbps],
              previous_bitrate_mbps: dqnState.bitrate_mbps,
            })
              .then((pred) => setQValues(pred.q_values))
              .catch(() => {});
          }

          stepIndex++;
        } else {
          clearInterval(intervalRef.current);
          setStatus("completed");
        }
      }, 1000 / speed);
    } catch (e) {
      console.error("Simulation error", e);
      setStatus("idle");
    }
  };

  const handlePauseResume = () => {
    if (status === "running") {
      clearInterval(intervalRef.current);
      setStatus("paused");
    } else if (status === "paused") {
      // Resume
      setStatus("running");
    }
  };

  const handleReset = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setStatus("idle");
    setCurrentStep(0);
    setActiveEvent(null);
    setTimeline([]);
  };

  // Derive active values
  const currentBitrate = activeEvent?.policy_states.dqn?.bitrate_mbps ?? 2.5;
  const currentQuality = activeEvent?.policy_states.dqn?.quality_name ?? "720p";
  const currentAction = activeEvent?.policy_states.dqn?.action ?? 3;
  const currentBuffer = activeEvent?.policy_states.dqn?.buffer_seconds ?? 12.0;
  const currentThroughput = activeEvent?.policy_states.dqn?.throughput_mbps ?? 5.2;
  const currentBandwidth = overrideBandwidth ?? (activeEvent?.network.bandwidth_mbps ?? 10.0);
  const currentRebuffer = activeEvent?.policy_states.dqn?.rebuffer_seconds ?? 0.0;

  // Chart telemetry array
  const chartData = timeline.map((e) => ({
    time: `${e.time_seconds.toFixed(0)}s`,
    bandwidth: Number(e.network.bandwidth_mbps.toFixed(2)),
    bitrate: Number(e.policy_states.dqn?.bitrate_mbps.toFixed(2) || 0),
    buffer: Number(e.policy_states.dqn?.buffer_seconds.toFixed(1) || 0),
  }));

  // Helper for buffer color threshold
  const getBufferColor = (buf: number) => {
    if (buf < 8.0) return "text-critical bg-critical/10 border-critical/30";
    if (buf < 16.0) return "text-warning bg-warning/10 border-warning/30";
    return "text-success bg-success/10 border-success/30";
  };

  return (
    <div className="p-6 max-w-[1700px] mx-auto space-y-6">
      {/* Top Notice: Clear Distinction Banner */}
      <div className="bg-surface-elevated/70 border border-border/80 rounded-xl px-4 py-2.5 flex items-center justify-between text-xs font-mono">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-cyan animate-pulse" />
          <span className="text-text-secondary">Execution Environment:</span>
          <span className="text-cyan font-semibold">
            {playbackMode === "synthetic"
              ? "Synthetic Link Discrete-Event Simulator (Mathematical Segment Transfers)"
              : "Live MPEG-DASH Video Delivery (Backend Video Server / HTTP ISO-BMFF Chunks)"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPlaybackMode("synthetic")}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all ${
              playbackMode === "synthetic"
                ? "bg-cyan text-background font-semibold"
                : "bg-surface text-text-muted hover:text-text-primary"
            }`}
          >
            Synthetic Simulator
          </button>
          <button
            onClick={() => setPlaybackMode("dash")}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all ${
              playbackMode === "dash"
                ? "bg-violet text-background font-semibold"
                : "bg-surface text-text-muted hover:text-text-primary"
            }`}
          >
            Real DASH Player
          </button>
        </div>
      </div>

      {/* 3-Column Technical Intelligence Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* COLUMN 1: CONTROLS & PERTURBATIONS (Width: 3 cols) */}
        <div className="lg:col-span-3 space-y-5">
          {/* Profile & Experiment Config */}
          <div className="bg-surface rounded-xl border border-border p-5 space-y-4 shadow-surface">
            <div className="flex items-center justify-between pb-2 border-b border-border/60">
              <span className="text-xs font-mono font-bold text-text-primary uppercase tracking-wider">
                Network Profile
              </span>
              <span className="text-[10px] font-mono text-cyan bg-cyan/10 px-2 py-0.5 rounded border border-cyan/20">
                Trace Engine
              </span>
            </div>

            <div className="space-y-3">
              <div>
                <label className="text-[11px] font-mono text-text-muted block mb-1.5">
                  Select Profile
                </label>
                <select
                  value={selectedProfile}
                  onChange={(e) => setSelectedProfile(e.target.value)}
                  disabled={status === "running"}
                  className="w-full bg-surface-elevated border border-border rounded-lg px-3 py-2 text-xs font-mono text-text-primary focus:border-cyan focus:outline-none"
                >
                  {profiles.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name} ({p.difficulty || "Profile"})
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-[11px] font-mono text-text-muted block mb-1">
                    Duration (s)
                  </label>
                  <input
                    type="number"
                    min={10}
                    max={120}
                    step={10}
                    value={duration}
                    onChange={(e) => setDuration(Number(e.target.value))}
                    disabled={status === "running"}
                    className="w-full bg-surface-elevated border border-border rounded-lg px-2.5 py-1.5 text-xs font-mono text-text-primary"
                  />
                </div>
                <div>
                  <label className="text-[11px] font-mono text-text-muted block mb-1">
                    RNG Seed
                  </label>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                    disabled={status === "running"}
                    className="w-full bg-surface-elevated border border-border rounded-lg px-2.5 py-1.5 text-xs font-mono text-text-primary"
                  />
                </div>
              </div>

              {/* Playback speed multiplier */}
              <div>
                <label className="text-[11px] font-mono text-text-muted block mb-1.5">
                  Simulation Clock Speed
                </label>
                <div className="grid grid-cols-3 gap-1.5">
                  {[1, 2, 5].map((s) => (
                    <button
                      key={s}
                      onClick={() => setSpeed(s)}
                      className={`py-1 rounded text-xs font-mono font-medium border transition-all ${
                        speed === s
                          ? "bg-cyan/15 text-cyan border-cyan/40"
                          : "bg-surface-elevated text-text-muted border-border hover:text-text-primary"
                      }`}
                    >
                      {s}x Speed
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Run / Pause / Reset Control Buttons */}
            <div className="pt-2 flex items-center gap-2">
              {status === "idle" || status === "completed" ? (
                <button
                  onClick={handleStart}
                  className="flex-1 py-2.5 rounded-lg bg-cyan text-background font-mono font-bold text-xs flex items-center justify-center gap-2 hover:bg-cyan/90 transition-all shadow-cyan-glow"
                >
                  <Play className="w-4 h-4 fill-current" />
                  <span>Start Simulation</span>
                </button>
              ) : (
                <button
                  onClick={handlePauseResume}
                  className="flex-1 py-2.5 rounded-lg bg-warning/20 text-warning border border-warning/40 font-mono font-bold text-xs flex items-center justify-center gap-2 hover:bg-warning/30 transition-all"
                >
                  <Pause className="w-4 h-4" />
                  <span>{status === "running" ? "Pause" : "Resume"}</span>
                </button>
              )}

              <button
                onClick={handleReset}
                title="Reset simulation"
                className="px-3 py-2.5 rounded-lg bg-surface-elevated border border-border hover:border-text-secondary text-text-secondary transition-all"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Live Perturbation Sliders (Manual Overrides) */}
          <div className="bg-surface rounded-xl border border-border p-5 space-y-4 shadow-surface">
            <div className="flex items-center justify-between pb-2 border-b border-border/60">
              <span className="text-xs font-mono font-bold text-text-primary uppercase tracking-wider flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-warning" />
                <span>Live Perturbations</span>
              </span>
              <span className="text-[10px] font-mono text-warning bg-warning/10 px-1.5 py-0.5 rounded border border-warning/20">
                Active Choke
              </span>
            </div>

            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between text-[11px] font-mono mb-1">
                  <span className="text-text-secondary">Bandwidth Override</span>
                  <span className="text-cyan font-bold">
                    {overrideBandwidth ? `${overrideBandwidth.toFixed(1)} Mbps` : "Auto (Trace)"}
                  </span>
                </div>
                <input
                  type="range"
                  min={0.5}
                  max={15.0}
                  step={0.5}
                  value={overrideBandwidth ?? 10.0}
                  onChange={(e) => setOverrideBandwidth(Number(e.target.value))}
                  className="w-full accent-cyan"
                />
                {overrideBandwidth && (
                  <button
                    onClick={() => setOverrideBandwidth(null)}
                    className="text-[10px] font-mono text-text-muted hover:text-cyan mt-1"
                  >
                    Reset to profile bandwidth
                  </button>
                )}
              </div>

              <div>
                <div className="flex items-center justify-between text-[11px] font-mono mb-1">
                  <span className="text-text-secondary">Latency (RTT delay)</span>
                  <span className="text-violet font-bold">{overrideLatency} ms</span>
                </div>
                <input
                  type="range"
                  min={5}
                  max={150}
                  step={5}
                  value={overrideLatency}
                  onChange={(e) => setOverrideLatency(Number(e.target.value))}
                  className="w-full accent-violet"
                />
              </div>

              <div>
                <div className="flex items-center justify-between text-[11px] font-mono mb-1">
                  <span className="text-text-secondary">Packet Loss Rate</span>
                  <span className={overrideLoss > 0 ? "text-critical font-bold" : "text-text-muted"}>
                    {overrideLoss}%
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={25}
                  step={1}
                  value={overrideLoss}
                  onChange={(e) => setOverrideLoss(Number(e.target.value))}
                  className="w-full accent-critical"
                />
              </div>
            </div>
          </div>
        </div>

        {/* COLUMN 2: VIDEO CANVAS & BUFFER DYNAMICS (Width: 6 cols) */}
        <div className="lg:col-span-6 space-y-5">
          {/* Main Visual Display */}
          <div className="bg-surface rounded-xl border border-border overflow-hidden shadow-surface flex flex-col">
            {/* Canvas Header */}
            <div className="h-12 px-4 bg-surface-elevated/80 border-b border-border flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Tv className="w-4 h-4 text-cyan" />
                <span className="text-xs font-mono font-medium text-text-primary">
                  {playbackMode === "synthetic"
                    ? "Interactive Simulation Canvas"
                    : "MPEG-DASH HTML5 Media Stream"}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-violet/10 text-violet border border-violet/20 font-semibold">
                  {currentQuality} · {currentBitrate.toFixed(1)} Mbps
                </span>
                <span className="text-[10px] font-mono text-text-muted">
                  Segment {currentStep} / {totalSteps}
                </span>
              </div>
            </div>

            {/* Video Canvas Visual */}
            <div className="relative aspect-video bg-[#070A10] flex items-center justify-center p-6 overflow-hidden">
              {playbackMode === "synthetic" ? (
                <div className="w-full h-full flex flex-col items-center justify-center relative border border-border/40 rounded-lg bg-surface/30 p-6">
                  {/* Watermark grid */}
                  <div className="absolute inset-0 bg-[radial-gradient(#1E293B_1px,transparent_1px)] [background-size:16px_16px] opacity-30" />

                  {/* Active segment download visualizer */}
                  <div className="relative z-10 flex flex-col items-center gap-3 text-center">
                    <div className="w-16 h-16 rounded-2xl bg-surface-elevated border border-border flex items-center justify-center shadow-cyan-glow">
                      <Cpu className="w-8 h-8 text-cyan animate-pulse" />
                    </div>

                    <div>
                      <div className="text-xl font-mono font-bold text-text-primary">
                        {currentQuality} Video Segment
                      </div>
                      <div className="text-xs font-mono text-text-muted mt-1">
                        DQN selected bitrate: <strong className="text-cyan">{currentBitrate} Mbps</strong> · 4.0s segment
                      </div>
                    </div>

                    {/* Animated Progress Bar */}
                    <div className="w-64 h-2 bg-surface-elevated rounded-full overflow-hidden border border-border/80 mt-2">
                      <div
                        className="h-full bg-cyan transition-all duration-300"
                        style={{
                          width: `${(currentStep / Math.max(1, totalSteps)) * 100}%`,
                        }}
                      />
                    </div>
                  </div>

                  {/* Rebuffer Alert Overlay */}
                  {currentRebuffer > 0 && (
                    <div className="absolute inset-0 bg-critical/20 backdrop-blur-xs flex items-center justify-center z-20">
                      <div className="bg-surface border border-critical text-critical px-4 py-2 rounded-lg font-mono text-xs font-bold flex items-center gap-2 glow-critical">
                        <AlertTriangle className="w-4 h-4" />
                        <span>REBUFFERING STALL: {currentRebuffer.toFixed(2)}s</span>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <video
                  className="w-full h-full rounded-lg"
                  controls
                  poster="/favicon.svg"
                  src="/api/video/manifest.mpd"
                >
                  Your browser does not support MPEG-DASH.
                </video>
              )}
            </div>

            {/* Buffer Gauge Ribbon */}
            <div className="p-4 bg-surface-elevated/40 border-t border-border space-y-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-text-secondary flex items-center gap-1.5">
                  <Gauge className="w-3.5 h-3.5 text-cyan" />
                  <span>Playback Buffer Occupancy</span>
                </span>
                <span className={`px-2 py-0.5 rounded border text-[11px] font-bold ${getBufferColor(currentBuffer)}`}>
                  {currentBuffer.toFixed(1)}s / 60.0s Capacity
                </span>
              </div>

              {/* Multi-segment Buffer Bar */}
              <div className="w-full h-3 bg-surface-elevated rounded-full overflow-hidden border border-border relative">
                {/* Danger threshold line (8s) */}
                <div className="absolute left-[13.3%] top-0 bottom-0 w-0.5 bg-critical/60 z-10" />
                {/* Warning threshold line (16s) */}
                <div className="absolute left-[26.6%] top-0 bottom-0 w-0.5 bg-warning/60 z-10" />

                <div
                  className="h-full bg-gradient-to-r from-warning via-success to-success transition-all duration-500"
                  style={{ width: `${Math.min(100, (currentBuffer / 60.0) * 100)}%` }}
                />
              </div>

              <div className="flex justify-between text-[10px] font-mono text-text-muted">
                <span>0s (Stall)</span>
                <span>8s (Low Alert)</span>
                <span>16s (Comfortable)</span>
                <span>60s (Max Buffer)</span>
              </div>
            </div>
          </div>

          {/* Synchronized Real-time Line Graph */}
          <div className="bg-surface rounded-xl border border-border p-4 shadow-surface space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-semibold text-text-primary">
                Telemetry Timeline
              </span>
              <div className="flex items-center gap-3 text-[11px] font-mono">
                <span className="text-cyan">■ Bandwidth ({currentBandwidth.toFixed(1)}M)</span>
                <span className="text-violet">■ Selected ({currentBitrate}M)</span>
              </div>
            </div>

            <div className="h-36 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="2 2" stroke="#1E293B" vertical={false} />
                  <XAxis dataKey="time" stroke="#64748B" fontSize={10} fontFamily="JetBrains Mono" />
                  <YAxis stroke="#64748B" fontSize={10} fontFamily="JetBrains Mono" unit="M" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#121A29",
                      borderColor: "#24324D",
                      borderRadius: "6px",
                      fontFamily: "JetBrains Mono",
                      fontSize: "11px",
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="bandwidth"
                    stroke="#38BDF8"
                    strokeWidth={1.5}
                    fillOpacity={0.15}
                    fill="#38BDF8"
                  />
                  <Area
                    type="stepAfter"
                    dataKey="bitrate"
                    stroke="#A78BFA"
                    strokeWidth={2}
                    fillOpacity={0.15}
                    fill="#A78BFA"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* COLUMN 3: DQN INTELLIGENCE & Q-VALUES (Width: 3 cols) */}
        <div className="lg:col-span-3 space-y-5">
          {/* Action & Quality Card */}
          <div className="bg-surface rounded-xl border border-border p-5 space-y-4 shadow-surface">
            <div className="flex items-center justify-between pb-2 border-b border-border/60">
              <span className="text-xs font-mono font-bold text-text-primary uppercase tracking-wider flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-violet" />
                <span>DQN Decision Engine</span>
              </span>
              <span className="text-[10px] font-mono text-violet bg-violet/10 px-1.5 py-0.5 rounded border border-violet/20 font-semibold">
                Action {currentAction}
              </span>
            </div>

            {/* Large Highlight Pill */}
            <div className="bg-surface-elevated p-3.5 rounded-lg border border-border-highlight text-center space-y-1">
              <span className="text-[11px] font-mono text-text-muted uppercase">Recommended Quality</span>
              <div className="text-3xl font-mono font-black text-violet tracking-tight">
                {currentQuality}
              </div>
              <span className="text-xs font-mono text-cyan">{currentBitrate} Mbps Target Bitrate</span>
            </div>

            {/* Q-Value Distribution Bars */}
            <div className="space-y-2.5">
              <div className="flex items-center justify-between text-[11px] font-mono text-text-muted">
                <span>Action Space (Q-Values)</span>
                <span>Expected Reward</span>
              </div>

              {QUALITY_NAMES.map((name, idx) => {
                const q = qValues[idx] ?? 0;
                const isSelected = idx === currentAction;
                const maxQ = Math.max(...qValues);
                const minQ = Math.min(...qValues);
                const normalized = maxQ === minQ ? 50 : Math.max(5, ((q - minQ) / (maxQ - minQ)) * 100);

                return (
                  <div key={name} className="space-y-1">
                    <div className="flex items-center justify-between text-[11px] font-mono">
                      <span className={isSelected ? "text-violet font-bold" : "text-text-secondary"}>
                        [{idx}] {name} ({BITRATES[idx]}M)
                      </span>
                      <span className={isSelected ? "text-violet font-bold" : "text-text-muted"}>
                        {q.toFixed(1)}
                      </span>
                    </div>
                    <div className="w-full h-1.5 bg-surface-elevated rounded-full overflow-hidden border border-border/50">
                      <div
                        className={`h-full transition-all duration-300 ${
                          isSelected ? "bg-violet" : "bg-text-muted/40"
                        }`}
                        style={{ width: `${normalized}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Telemetry Observation State Vector */}
          <div className="bg-surface rounded-xl border border-border p-5 space-y-3 shadow-surface">
            <span className="text-xs font-mono font-bold text-text-primary uppercase tracking-wider block pb-2 border-b border-border/60">
              State Vector (9 Features)
            </span>

            <div className="space-y-2 text-xs font-mono">
              <div className="flex justify-between py-1 border-b border-border/30">
                <span className="text-text-muted">Buffer Occupancy:</span>
                <span className="text-text-primary">{currentBuffer.toFixed(2)}s</span>
              </div>
              <div className="flex justify-between py-1 border-b border-border/30">
                <span className="text-text-muted">Measured Throughput:</span>
                <span className="text-cyan">{currentThroughput.toFixed(2)} Mbps</span>
              </div>
              <div className="flex justify-between py-1 border-b border-border/30">
                <span className="text-text-muted">Previous Bitrate:</span>
                <span className="text-text-primary">{currentBitrate} Mbps</span>
              </div>
              <div className="flex justify-between py-1 border-b border-border/30">
                <span className="text-text-muted">Segments Remaining:</span>
                <span className="text-text-primary">{Math.max(0, totalSteps - currentStep)}</span>
              </div>
            </div>

            {/* Decision Rationale */}
            <div className="mt-3 p-2.5 rounded-lg bg-surface-elevated border border-border text-[11px] font-mono text-text-secondary">
              <strong className="text-violet">Agent Rationale: </strong>
              {currentBuffer > 15.0
                ? "Sufficient buffer reservoir (>15s) provides safety margin to absorb link fluctuation; selecting highest stable tier."
                : currentBuffer < 8.0
                ? "Buffer in critical depletion zone (<8s); stepping down quality immediately to avoid playback freeze."
                : "Balanced buffer occupancy; maintaining quality consistency to minimize switching penalty."}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

