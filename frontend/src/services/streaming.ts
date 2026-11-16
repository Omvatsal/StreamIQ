export type Video = { id: string; manifest: string };

export type DecisionState = {
  buffer_seconds: number;
  throughput_history_mbps: number[];
  previous_bitrate_mbps: number;
  previous_download_duration_seconds: number;
  remaining_segments: number;
};

export type Decision = {
  action: number;
  bitrate_mbps: number;
  representation: string;
  fallback: boolean;
  error: string | null;
};

export async function listVideos(): Promise<Video[]> {
  const response = await fetch("/api/videos");
  if (!response.ok) throw new Error("Unable to list videos");
  return (await response.json()).videos as Video[];
}

export async function requestDecision(state: DecisionState): Promise<Decision> {
  const response = await fetch("/api/abr/decision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state),
  });
  if (!response.ok) throw new Error("ABR decision request failed");
  return (await response.json()) as Decision;
}
