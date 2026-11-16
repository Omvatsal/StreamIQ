import type { PlaybackMetrics } from "./VideoPlayer";

type Props = { metrics: PlaybackMetrics };

export function StreamingMetrics({ metrics }: Props) {
  return (
    <dl>
      <dt>Mode</dt><dd>{metrics.mode}</dd>
      <dt>Bitrate</dt><dd>{metrics.bitrate.toFixed(2)} Mbps</dd>
      <dt>Buffer</dt><dd>{metrics.buffer.toFixed(1)} s</dd>
      <dt>Throughput</dt><dd>{metrics.throughput.toFixed(2)} Mbps</dd>
      <dt>Rebuffers</dt><dd>{metrics.rebuffers}</dd>
    </dl>
  );
}
