export type TabId =
  | "overview"
  | "live-simulation"
  | "policy-arena"
  | "network-lab"
  | "experiments"
  | "model-lab"
  | "video-player"
  | "settings";

export interface NetworkProfile {
  id: string;
  name: string;
  description: string;
  default_bandwidth_mbps?: number;
  difficulty?: string;
}

export interface NetworkCondition {
  timestamp_seconds: number;
  bandwidth_mbps: number;
  delay_ms: number;
  packet_loss_percent: number;
}

export interface PolicyStepTelemetry {
  policy_name: string;
  action: number;
  quality_name: string;
  bitrate_mbps: number;
  buffer_seconds: number;
  download_duration_seconds: number;
  rebuffer_seconds: number;
  throughput_mbps: number;
  step_reward: number;
  accumulated_qoe: number;
}

export interface SimulationStepEvent {
  step_index: number;
  segment_index: number;
  time_seconds: number;
  network: NetworkCondition;
  policy_states: Record<string, PolicyStepTelemetry>;
}

export interface PolicySummary {
  policy_name: string;
  average_bitrate_mbps: number;
  rebuffer_seconds: number;
  rebuffer_ratio: number;
  bitrate_switches: number;
  average_bitrate_change_mbps: number;
  total_bitrate_variation_mbps: number;
  upward_switches: number;
  downward_switches: number;
  switching_magnitude: number;
  average_download_duration_seconds: number;
  qoe: number;
}

export interface BatchSimulationResult {
  trace_id: string;
  profile_name: string;
  duration_seconds: number;
  segment_count: number;
  timeline: SimulationStepEvent[];
  summaries: Record<string, PolicySummary>;
  winner_policy: string;
}

export interface ABRPrediction {
  selected_action: number;
  quality_name: string;
  bitrate_mbps: number;
  q_values: number[];
  observation: number[];
}

