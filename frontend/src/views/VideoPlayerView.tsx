import React, { useState } from "react";
import {
  CheckCircle2,
  Cpu,
  Layers,
  PlaySquare,
  Radio,
  Settings as SettingsIcon,
  ShieldCheck,
  Tv,
  Wifi,
} from "lucide-react";

export const VideoPlayerView: React.FC = () => {
  const [abrMode, setAbrMode] = useState<"auto" | "manual">("auto");
  const [manualQuality, setManualQuality] = useState<number>(3); // 720p default

  const bitrates = [0.3, 0.7, 1.2, 2.5, 5.0];
  const qualityNames = ["240p", "360p", "480p", "720p", "1080p"];

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-border/80">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold font-sans text-text-primary tracking-tight">
              MPEG-DASH Video Delivery Player
            </h1>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-violet/10 text-violet border border-violet/20 font-semibold">
              ISO/IEC 23009-1
            </span>
          </div>
          <p className="text-xs text-text-secondary font-mono mt-1">
            Standard HTML5 media delivery backed by StreamIQ FastAPI dynamic segment provider
          </p>
        </div>

        {/* Mode Toggle */}
        <div className="flex items-center bg-surface-elevated p-1 rounded-lg border border-border">
          <button
            onClick={() => setAbrMode("auto")}
            className={`px-3 py-1.5 rounded-md text-xs font-mono font-medium transition-all ${
              abrMode === "auto"
                ? "bg-violet text-background font-bold shadow-sm"
                : "text-text-secondary hover:text-text-primary"
            }`}
          >
            Auto (DQN Driven)
          </button>
          <button
            onClick={() => setAbrMode("manual")}
            className={`px-3 py-1.5 rounded-md text-xs font-mono font-medium transition-all ${
              abrMode === "manual"
                ? "bg-violet text-background font-bold shadow-sm"
                : "text-text-secondary hover:text-text-primary"
            }`}
          >
            Manual Tier
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left: Video Player Display (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          <div className="bg-surface rounded-xl border border-border overflow-hidden shadow-surface">
            {/* Player Topbar */}
            <div className="h-11 px-4 bg-surface-elevated/70 border-b border-border flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Tv className="w-4 h-4 text-cyan" />
                <span className="text-xs font-mono text-text-primary font-medium">
                  HTTP Live Stream · 4.0s Chunks
                </span>
              </div>
              <span className="text-[10px] font-mono text-cyan bg-cyan/10 px-2 py-0.5 rounded border border-cyan/20">
                /api/video/manifest.mpd
              </span>
            </div>

            {/* Video Viewport */}
            <div className="relative aspect-video bg-[#05070D] flex items-center justify-center p-4">
              <video
                className="w-full h-full rounded-lg shadow-lg"
                controls
                src="/api/video/manifest.mpd"
              >
                Your browser does not natively support MPEG-DASH MPD manifests.
              </video>
            </div>

            {/* Live Playback Telemetry Bar */}
            <div className="p-4 bg-surface-elevated/30 border-t border-border grid grid-cols-4 gap-4 text-xs font-mono">
              <div>
                <span className="text-[10px] text-text-muted uppercase block">Active Bitrate</span>
                <span className="text-violet font-bold text-sm">
                  {qualityNames[manualQuality]} ({bitrates[manualQuality]} Mbps)
                </span>
              </div>
              <div>
                <span className="text-[10px] text-text-muted uppercase block">Buffer Cush</span>
                <span className="text-success font-bold text-sm">18.4s Safe</span>
              </div>
              <div>
                <span className="text-[10px] text-text-muted uppercase block">Last Download</span>
                <span className="text-cyan font-bold text-sm">1.82s</span>
              </div>
              <div>
                <span className="text-[10px] text-text-muted uppercase block">QoE Penalty</span>
                <span className="text-text-primary font-bold text-sm">0.00 (Zero)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Stream Manifest Info & Manual Controls (4 cols) */}
        <div className="lg:col-span-4 space-y-5">
          <div className="bg-surface rounded-xl border border-border p-5 space-y-4 shadow-surface">
            <span className="text-xs font-mono font-bold text-text-primary uppercase tracking-wider block pb-2 border-b border-border/60">
              Adaptation Set Representations
            </span>

            <div className="space-y-2">
              {qualityNames.map((name, idx) => {
                const isSelected = manualQuality === idx;
                return (
                  <div
                    key={name}
                    onClick={() => {
                      if (abrMode === "manual") setManualQuality(idx);
                    }}
                    className={`p-3 rounded-lg border flex items-center justify-between text-xs font-mono transition-all ${
                      isSelected
                        ? "bg-violet/10 border-violet/40 text-violet font-bold shadow-sm"
                        : "bg-surface-elevated border-border text-text-secondary hover:text-text-primary"
                    } ${abrMode === "manual" ? "cursor-pointer" : "cursor-default"}`}
                  >
                    <div className="flex items-center gap-2.5">
                      <span className="w-5 text-center text-text-muted">{idx}</span>
                      <span>{name}</span>
                    </div>
                    <span className="text-text-muted">{bitrates[idx]} Mbps</span>
                  </div>
                );
              })}
            </div>

            {abrMode === "auto" && (
              <div className="p-3 bg-surface-elevated rounded-lg border border-border text-[11px] font-mono text-text-secondary">
                <strong className="text-violet">Auto Mode Active: </strong>
                StreamIQ's trained Deep Q-Network automatically intercepts segment requests and selects the optimal representation tier.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

