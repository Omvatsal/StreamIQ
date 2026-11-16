import { useEffect, useRef, useState } from "react";
import dashjs from "dashjs";
import { requestDecision, type DecisionState } from "../services/streaming";

export type PlaybackMetrics = {
  mode: "dqn" | "manual" | "dashjs";
  bitrate: number;
  buffer: number;
  throughput: number;
  rebuffers: number;
};

type Props = {
  manifestUrl: string;
  mode: PlaybackMetrics["mode"];
  manualAction: number;
  onMetrics: (metrics: PlaybackMetrics) => void;
};
const BITRATES = [0.3, 0.7, 1.2, 2.5, 5.0];

export function VideoPlayer({ manifestUrl, mode, manualAction, onMetrics }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const playerRef = useRef<dashjs.MediaPlayerClass | null>(null);
  const throughputRef = useRef<number[]>([0, 0, 0, 0, 0]);
  const previousBitrateRef = useRef(0.3);
  const previousDownloadRef = useRef(0);
  const remainingRef = useRef(10);
  const [metrics, setMetrics] = useState<PlaybackMetrics>({
    mode, bitrate: 0.3, buffer: 0, throughput: 0, rebuffers: 0,
  });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!videoRef.current) return;
    const player = dashjs.MediaPlayer().create();
    playerRef.current = player;
    player.initialize(videoRef.current, manifestUrl, false);
    player.on(dashjs.MediaPlayer.events.ERROR, (event) => {
      setError(event.error?.message || "Playback error");
    });
    player.updateSettings({
      streaming: { abr: { autoSwitchBitrate: { video: mode === "dashjs" } } },
    });

    const onFragment = async () => {
      const video = videoRef.current;
      if (!video) return;
      const buffer = Math.max(0, video.buffered.length ? video.buffered.end(video.buffered.length - 1) - video.currentTime : 0);
      const throughput = player.getAverageThroughput("video") || 0;
      throughputRef.current = [...throughputRef.current, throughput / 1000].slice(-5);
      const duration = previousDownloadRef.current;
      let action = manualAction;
      if (mode === "dqn") {
        const state: DecisionState = {
          buffer_seconds: buffer,
          throughput_history_mbps: throughputRef.current,
          previous_bitrate_mbps: previousBitrateRef.current,
          previous_download_duration_seconds: duration,
          remaining_segments: remainingRef.current,
        };
        try {
          action = (await requestDecision(state)).action;
        } catch {
          action = 0;
        }
      }
      if (mode !== "dashjs") {
        player.setQualityFor("video", action);
      }
      previousBitrateRef.current = BITRATES[action];
      remainingRef.current = Math.max(0, remainingRef.current - 1);
      const nextMetrics = {
        mode,
        bitrate: BITRATES[action],
        buffer,
        throughput: throughput / 1000,
        rebuffers: 0,
      };
      setMetrics((current) => ({
        ...current,
        ...nextMetrics,
      }));
      onMetrics(nextMetrics);
    };
    player.on(dashjs.MediaPlayer.events.FRAGMENT_LOADING_COMPLETED, onFragment);
    return () => {
      player.off(dashjs.MediaPlayer.events.FRAGMENT_LOADING_COMPLETED, onFragment);
      player.reset();
      playerRef.current = null;
    };
  }, [manifestUrl, mode, manualAction, onMetrics]);

  return (
    <div>
      {error && <p className="error">{error}</p>}
      <video ref={videoRef} controls width="800" />
      <pre>{JSON.stringify(metrics, null, 2)}</pre>
    </div>
  );
}
