import React, { useState } from "react";
import {
  Check,
  CheckCircle2,
  Database,
  Globe,
  Palette,
  Server,
  Settings as SettingsIcon,
  Shield,
  Sliders,
} from "lucide-react";

export const SettingsView: React.FC = () => {
  const [apiUrl, setApiUrl] = useState("http://localhost:8000");
  const [wsUrl, setWsUrl] = useState("ws://localhost:8000/api/simulation/live");
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="p-8 space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-border/80">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold font-sans text-text-primary tracking-tight">
              Platform Settings & Parameters
            </h1>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-cyan/10 text-cyan border border-cyan/20 font-semibold">
              System Config
            </span>
          </div>
          <p className="text-xs text-text-secondary font-mono mt-1">
            FastAPI backend connection, simulation hyperparameters, and design tokens
          </p>
        </div>

        <button
          onClick={handleSave}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan text-background font-mono text-xs font-semibold hover:bg-cyan/90 transition-all shadow-cyan-glow"
        >
          {saved ? <Check className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />}
          <span>{saved ? "Saved" : "Save Changes"}</span>
        </button>
      </div>

      <div className="space-y-6">
        {/* Backend Endpoints */}
        <div className="bg-surface rounded-xl border border-border p-6 shadow-surface space-y-4">
          <div className="flex items-center gap-2 pb-2 border-b border-border/60">
            <Server className="w-4 h-4 text-cyan" />
            <h2 className="text-xs font-mono font-bold text-text-primary uppercase tracking-wider">
              Service Gateways
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-mono text-text-secondary block mb-1.5">
                FastAPI REST Base URL
              </label>
              <input
                type="text"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                className="w-full bg-surface-elevated border border-border rounded-lg px-3 py-2 text-xs font-mono text-text-primary focus:border-cyan focus:outline-none"
              />
            </div>
            <div>
              <label className="text-xs font-mono text-text-secondary block mb-1.5">
                WebSocket Live Stream URL
              </label>
              <input
                type="text"
                value={wsUrl}
                onChange={(e) => setWsUrl(e.target.value)}
                className="w-full bg-surface-elevated border border-border rounded-lg px-3 py-2 text-xs font-mono text-text-primary focus:border-cyan focus:outline-none"
              />
            </div>
          </div>
        </div>

        {/* Video & Environment Defaults */}
        <div className="bg-surface rounded-xl border border-border p-6 shadow-surface space-y-4">
          <div className="flex items-center gap-2 pb-2 border-b border-border/60">
            <Sliders className="w-4 h-4 text-violet" />
            <h2 className="text-xs font-mono font-bold text-text-primary uppercase tracking-wider">
              ABR Environment Defaults
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
            <div className="p-3.5 bg-surface-elevated rounded-lg border border-border space-y-1">
              <span className="text-text-muted uppercase text-[10px]">Segment Duration</span>
              <div className="text-base font-bold text-text-primary">4.0 seconds</div>
              <span className="text-[11px] text-text-secondary">Fixed MPEG-DASH chunk length</span>
            </div>
            <div className="p-3.5 bg-surface-elevated rounded-lg border border-border space-y-1">
              <span className="text-text-muted uppercase text-[10px]">Buffer Capacity</span>
              <div className="text-base font-bold text-text-primary">60.0 seconds</div>
              <span className="text-[11px] text-text-secondary">Maximum client player buffer</span>
            </div>
            <div className="p-3.5 bg-surface-elevated rounded-lg border border-border space-y-1">
              <span className="text-text-muted uppercase text-[10px]">Bitrate Tiers</span>
              <div className="text-base font-bold text-violet">5 Levels (240p - 1080p)</div>
              <span className="text-[11px] text-text-secondary">0.3, 0.7, 1.2, 2.5, 5.0 Mbps</span>
            </div>
          </div>
        </div>

        {/* UI Design Tokens Reference */}
        <div className="bg-surface rounded-xl border border-border p-6 shadow-surface space-y-4">
          <div className="flex items-center gap-2 pb-2 border-b border-border/60">
            <Palette className="w-4 h-4 text-success" />
            <h2 className="text-xs font-mono font-bold text-text-primary uppercase tracking-wider">
              Technical Design Tokens
            </h2>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono">
            <div className="p-3 rounded-lg bg-surface-elevated border border-border flex items-center gap-2.5">
              <div className="w-4 h-4 rounded bg-[#0B0F19] border border-border" />
              <div>
                <span className="block font-bold">Background</span>
                <span className="text-text-muted text-[10px]">#0B0F19</span>
              </div>
            </div>
            <div className="p-3 rounded-lg bg-surface-elevated border border-border flex items-center gap-2.5">
              <div className="w-4 h-4 rounded bg-[#121A29] border border-border" />
              <div>
                <span className="block font-bold">Surface</span>
                <span className="text-text-muted text-[10px]">#121A29</span>
              </div>
            </div>
            <div className="p-3 rounded-lg bg-surface-elevated border border-border flex items-center gap-2.5">
              <div className="w-4 h-4 rounded bg-[#38BDF8]" />
              <div>
                <span className="block font-bold">Primary Cyan</span>
                <span className="text-text-muted text-[10px]">#38BDF8</span>
              </div>
            </div>
            <div className="p-3 rounded-lg bg-surface-elevated border border-border flex items-center gap-2.5">
              <div className="w-4 h-4 rounded bg-[#A78BFA]" />
              <div>
                <span className="block font-bold">AI Violet</span>
                <span className="text-text-muted text-[10px]">#A78BFA</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

