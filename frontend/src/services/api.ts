import {
  NetworkProfile,
  BatchSimulationResult,
  ABRPrediction,
} from "../types";

const API_BASE = "/api";

export async function fetchProfiles(): Promise<NetworkProfile[]> {
  try {
    const res = await fetch(`${API_BASE}/traces/profiles`);
    if (!res.ok) throw new Error("Failed to fetch profiles");
    const data = await res.json();
    return data.profiles;
  } catch {
    // Graceful fallback profiles
    return [
      { id: "stable", name: "Stable Link", description: "Minimal jitter, reliable capacity", difficulty: "Easy" },
      { id: "random", name: "Random Walk", description: "Stochastic cellular network shifts", difficulty: "Medium" },
      { id: "oscillating", name: "Oscillating", description: "Periodic sinusoidal capacity changes", difficulty: "Hard" },
      { id: "sudden_drop", name: "Sudden Drop", description: "Severe mid-stream bandwidth drop", difficulty: "Hard" },
      { id: "recovery", name: "Gradual Recovery", description: "Drop followed by slow linear restoration", difficulty: "Medium" },
      { id: "congestion", name: "Congestion", description: "Degraded capacity with 35ms latency & 5% loss", difficulty: "Extreme" },
    ];
  }
}

export async function runBatchSimulation(params: {
  profile: string;
  duration_seconds: number;
  interval_seconds?: number;
  seed?: number;
  policies?: string[];
}): Promise<BatchSimulationResult> {
  const res = await fetch(`${API_BASE}/simulation/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("Batch simulation failed");
  return res.json();
}

export async function predictBitrate(params: {
  buffer_occupancy_seconds: number;
  throughput_history_mbps: number[];
  previous_bitrate_mbps?: number;
  previous_download_duration_seconds?: number;
  remaining_segments?: number;
}): Promise<ABRPrediction> {
  const res = await fetch(`${API_BASE}/abr/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("Inference failed");
  return res.json();
}

export function createLiveSimulationSocket(
  params: {
    profile: string;
    duration_seconds: number;
    interval_seconds: number;
    seed: number;
    policies: string[];
    step_delay: number;
  },
  onMessage: (data: any) => void,
  onError?: (err: any) => void,
  onClose?: () => void
): WebSocket {
  const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${wsProtocol}//${window.location.host}/api/simulation/live`);
  ws.onopen = () => {
    ws.send(JSON.stringify(params));
  };
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error("WS Parse error", e);
    }
  };
  if (onError) ws.onerror = onError;
  if (onClose) ws.onclose = onClose;
  return ws;
}

