import React from "react";
import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  description?: string;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  badge?: string;
  accent?: "cyan" | "violet" | "success" | "warning" | "critical" | "neutral";
  icon?: React.ComponentType<{ className?: string }>;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  description,
  trend,
  trendValue,
  badge,
  accent = "neutral",
  icon: Icon,
}) => {
  const accentClasses = {
    cyan: "border-cyan/30 bg-cyan/5 text-cyan",
    violet: "border-violet/30 bg-violet/5 text-violet",
    success: "border-success/30 bg-success/5 text-success",
    warning: "border-warning/30 bg-warning/5 text-warning",
    critical: "border-critical/30 bg-critical/5 text-critical",
    neutral: "border-border text-text-secondary",
  };

  const accentColor = {
    cyan: "text-cyan",
    violet: "text-violet",
    success: "text-success",
    warning: "text-warning",
    critical: "text-critical",
    neutral: "text-text-primary",
  };

  return (
    <div className="bg-surface rounded-xl border border-border p-4.5 transition-all hover:border-border-highlight flex flex-col justify-between group relative overflow-hidden shadow-surface">
      {/* Subtle hover gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-white/[0.02] to-transparent pointer-events-none" />

      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-mono font-medium text-text-secondary tracking-tight">
            {title}
          </span>
          <div className="flex items-center gap-1.5">
            {badge && (
              <span
                className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${accentClasses[accent]}`}
              >
                {badge}
              </span>
            )}
            {Icon && <Icon className={`w-4 h-4 ${accentColor[accent]} opacity-70 group-hover:opacity-100 transition-opacity`} />}
          </div>
        </div>

        <div className="flex items-baseline gap-1.5 mt-1">
          <span className={`text-2xl font-mono font-bold tracking-tight ${accentColor[accent]}`}>
            {value}
          </span>
          {unit && (
            <span className="text-xs font-mono text-text-muted">{unit}</span>
          )}
        </div>
      </div>

      {(description || trendValue) && (
        <div className="mt-3 pt-2.5 border-t border-border/50 flex items-center justify-between text-[11px] font-mono">
          {description && (
            <span className="text-text-muted truncate max-w-[180px]">
              {description}
            </span>
          )}
          {trendValue && (
            <div className="flex items-center gap-0.5 text-text-secondary ml-auto shrink-0">
              {trend === "up" && <ArrowUpRight className="w-3 h-3 text-success" />}
              {trend === "down" && <ArrowDownRight className="w-3 h-3 text-critical" />}
              {trend === "neutral" && <Minus className="w-3 h-3 text-text-muted" />}
              <span
                className={
                  trend === "up"
                    ? "text-success"
                    : trend === "down"
                    ? "text-critical"
                    : "text-text-muted"
                }
              >
                {trendValue}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

