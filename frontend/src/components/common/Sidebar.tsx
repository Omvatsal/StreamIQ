import React from "react";
import {
  Activity,
  BarChart3,
  Cpu,
  Flame,
  Globe,
  LayoutDashboard,
  PlaySquare,
  Settings as SettingsIcon,
  Radio,
} from "lucide-react";
import { TabId } from "../../types";

interface SidebarProps {
  activeTab: TabId;
  onSelectTab: (tab: TabId) => void;
  simulationRunning?: boolean;
}

interface NavItem {
  id: TabId;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
  highlight?: boolean;
}

const NAV_ITEMS: NavItem[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "live-simulation", label: "Live Simulation", icon: Radio, highlight: true },
  { id: "policy-arena", label: "Policy Arena", icon: Flame, badge: "DQN vs 3" },
  { id: "network-lab", label: "Network Lab", icon: Globe },
  { id: "experiments", label: "Experiments", icon: BarChart3 },
  { id: "model-lab", label: "Model Lab", icon: Cpu },
  { id: "video-player", label: "Video Player", icon: PlaySquare, badge: "DASH" },
  { id: "settings", label: "Settings", icon: SettingsIcon },
];

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  simulationRunning = false,
}) => {
  return (
    <aside className="w-64 bg-surface border-r border-border flex flex-col h-screen shrink-0 select-none">
      {/* Brand header */}
      <div className="h-16 flex items-center px-5 border-b border-border/80 gap-3">
        <div className="w-9 h-9 rounded-lg bg-surface-elevated border border-border flex items-center justify-center relative overflow-hidden shadow-sm">
          <Activity className="w-5 h-5 text-cyan" />
          <div className="absolute -bottom-1 -right-1 w-2.5 h-2.5 rounded-full bg-violet" />
        </div>
        <div>
          <div className="flex items-center gap-1.5">
            <span className="font-semibold text-text-primary text-base tracking-tight font-sans">
              StreamIQ
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-violet/10 text-violet border border-violet/20 font-medium">
              v0.2
            </span>
          </div>
          <p className="text-[11px] text-text-muted font-mono tracking-tight">
            RL Streaming Lab
          </p>
        </div>
      </div>

      {/* Navigation list */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <div className="px-2 pb-2 text-[10px] font-mono uppercase tracking-wider text-text-muted font-semibold">
          Platform Navigation
        </div>
        {NAV_ITEMS.map((item) => {
          const isActive = activeTab === item.id;
          const Icon = item.icon;

          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all group relative ${
                isActive
                  ? "bg-surface-elevated text-cyan border border-border-highlight shadow-sm"
                  : "text-text-secondary hover:text-text-primary hover:bg-surface-subtle"
              }`}
            >
              <div className="flex items-center gap-3">
                <Icon
                  className={`w-4 h-4 transition-colors ${
                    isActive
                      ? "text-cyan"
                      : "text-text-muted group-hover:text-text-secondary"
                  }`}
                />
                <span className="tracking-tight">{item.label}</span>
              </div>

              {/* Status / badges */}
              <div className="flex items-center gap-1.5">
                {item.id === "live-simulation" && simulationRunning && (
                  <span className="flex h-2 w-2 relative">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan"></span>
                  </span>
                )}
                {item.badge && (
                  <span
                    className={`text-[10px] font-mono px-1.5 py-0.2 rounded border font-normal ${
                      isActive
                        ? "bg-cyan/10 text-cyan border-cyan/30"
                        : "bg-surface-elevated text-text-muted border-border"
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </div>
            </button>
          );
        })}
      </nav>

      {/* Footer system status */}
      <div className="p-3 border-t border-border/80">
        <div className="bg-surface-elevated/70 border border-border/70 rounded-lg p-2.5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
            <span className="text-[11px] font-mono text-text-secondary">
              DQN Engine Online
            </span>
          </div>
          <span className="text-[10px] font-mono text-cyan bg-cyan/10 px-1.5 py-0.5 rounded border border-cyan/20">
            PyTorch
          </span>
        </div>
      </div>
    </aside>
  );
};

