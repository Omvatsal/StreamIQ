import React from "react";
import {
  Activity,
  Layers,
  RefreshCw,
  Server,
  Zap,
} from "lucide-react";

interface TopBarProps {
  simulationStatus?: "idle" | "running" | "paused" | "completed";
  onQuickRun?: () => void;
  activeProfileName?: string;
}

export const TopBar: React.FC<TopBarProps> = ({
  simulationStatus = "idle",
  onQuickRun,
  activeProfileName = "Sudden Drop",
}) => {
  return (
    <header className="h-16 bg-surface/80 backdrop-blur-md border-b border-border px-6 flex items-center justify-between z-10 sticky top-0">
      {/* Left environment status info */}
      <div className="flex items-center gap-4 text-xs font-mono">
        <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-surface-elevated border border-border">
          <Layers className="w-3.5 h-3.5 text-cyan" />
          <span className="text-text-secondary">Env:</span>
          <span className="text-text-primary font-medium">Gymnasium ABR v1.2</span>
        </div>

        <div className="hidden md:flex items-center gap-2 px-2.5 py-1 rounded-md bg-surface-elevated border border-border">
          <Server className="w-3.5 h-3.5 text-violet" />
          <span className="text-text-secondary">Target:</span>
          <span className="text-text-primary font-medium">5-Action DQN (240p-1080p)</span>
        </div>

        <div className="hidden lg:flex items-center gap-2 px-2.5 py-1 rounded-md bg-surface-elevated border border-border">
          <Activity className="w-3.5 h-3.5 text-success" />
          <span className="text-text-secondary">Trace:</span>
          <span className="text-cyan font-medium">{activeProfileName}</span>
        </div>
      </div>

      {/* Right controls & simulation state indicator */}
      <div className="flex items-center gap-3">
        {/* Status pill */}
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-surface-elevated border border-border text-xs font-mono">
          <span
            className={`w-2 h-2 rounded-full ${
              simulationStatus === "running"
                ? "bg-cyan animate-ping"
                : simulationStatus === "paused"
                ? "bg-warning"
                : simulationStatus === "completed"
                ? "bg-success"
                : "bg-text-muted"
            }`}
          />
          <span className="capitalize text-text-secondary">
            Sim: <strong className="text-text-primary">{simulationStatus}</strong>
          </span>
        </div>

        {/* Quick Batch Action */}
        {onQuickRun && (
          <button
            onClick={onQuickRun}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-cyan/15 text-cyan hover:bg-cyan/25 border border-cyan/30 text-xs font-mono font-medium transition-all shadow-sm active:scale-95"
          >
            <Zap className="w-3.5 h-3.5" />
            <span>Benchmark Arena</span>
          </button>
        )}
      </div>
    </header>
  );
};

