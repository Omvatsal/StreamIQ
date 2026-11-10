"""MPEG-DASH asset generator for StreamIQ.

Supports two operational modes:
1. Native FFmpeg mode: Generates/transcodes actual H.264/AAC media streams.
2. Synthetic fallback mode: Generates structurally valid ISO-BMFF chunk files
   with exact mathematical byte sizes when FFmpeg is not available.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rl_engine.environment.video import VideoModel

DEFAULT_OUTPUT_DIR = ROOT_DIR / "backend" / "assets" / "dash"


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available in the system PATH."""
    return shutil.which("ffmpeg") is not None


def generate_synthetic_assets(output_dir: Path, duration_seconds: float = 60.0) -> None:
    """Generate synthetic segment files matching exact bitrates and segment durations."""
    video = VideoModel()
    total_segments = int(duration_seconds // video.segment_duration_seconds)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic DASH assets in {output_dir}")
    print(f"Duration: {duration_seconds}s ({total_segments} segments of {video.segment_duration_seconds}s each)")

    for quality_idx, (bitrate, name) in enumerate(zip(video.bitrates_mbps, video.quality_names)):
        quality_dir = output_dir / str(quality_idx)
        quality_dir.mkdir(parents=True, exist_ok=True)

        # Write init.mp4
        init_file = quality_dir / "init.mp4"
        init_payload = (
            b"\x00\x00\x00\x20ftypiso5\x00\x00\x02\x00iso5avc1dash\x00\x00\x00\x08free"
        )
        init_file.write_bytes(init_payload)

        # Target size per segment in bytes
        target_bytes = int(video.segment_size_megabytes(quality_idx) * 1_000_000)
        box_header = b"\x00\x00\x00\x18stypmsdh\x00\x00\x00\x00msdhmsix"
        payload = box_header + (b"\x00" * max(0, target_bytes - len(box_header)))

        for seg_idx in range(total_segments):
            seg_file = quality_dir / f"segment_{seg_idx}.m4s"
            seg_file.write_bytes(payload)

        print(f"  [{quality_idx}] {name} ({bitrate} Mbps): created {total_segments} segments (~{target_bytes / 1024:.1f} KB each)")

    print("Synthetic asset generation complete.")


def generate_ffmpeg_assets(output_dir: Path, duration_seconds: float = 60.0) -> None:
    """Use FFmpeg to encode a synthetic video pattern into multi-bitrate DASH streams."""
    video = VideoModel()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_file = output_dir / "synthetic_source.mp4"

    print("Generating synthetic video source with FFmpeg testsrc...")
    # Generate testsrc with countdown timer
    cmd_gen = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", f"testsrc=size=1920x1080:rate=30:duration={int(duration_seconds)}",
        "-f", "lavfi",
        "-i", f"sine=frequency=1000:duration={int(duration_seconds)}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        str(source_file),
    ]
    subprocess.run(cmd_gen, check=True)

    # Encode each representation
    for quality_idx, (bitrate, name) in enumerate(zip(video.bitrates_mbps, video.quality_names)):
        quality_dir = output_dir / str(quality_idx)
        quality_dir.mkdir(parents=True, exist_ok=True)
        res_map = {
            "240p": "426x240",
            "360p": "640x360",
            "480p": "854x480",
            "720p": "1280x720",
            "1080p": "1920x1080",
        }
        scale = res_map.get(name, "640x360")
        target_kbps = int(bitrate * 1000)

        cmd_encode = [
            "ffmpeg",
            "-y",
            "-i", str(source_file),
            "-vf", f"scale={scale}",
            "-b:v", f"{target_kbps}k",
            "-maxrate", f"{int(target_kbps * 1.2)}k",
            "-bufsize", f"{target_kbps * 2}k",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-g", "120",  # 4 seconds at 30 fps
            "-keyint_min", "120",
            "-sc_threshold", "0",
            "-f", "segment",
            "-segment_time", "4",
            "-segment_format", "mp4",
            str(quality_dir / "segment_%d.m4s"),
        ]
        subprocess.run(cmd_encode, check=True)

    if source_file.exists():
        source_file.unlink()
    print("FFmpeg DASH transcoding complete.")


def main():
    parser = argparse.ArgumentParser(description="StreamIQ DASH asset generator")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Destination directory")
    parser.add_argument("--duration", type=float, default=60.0, help="Total media duration in seconds")
    parser.add_argument("--force-synthetic", action="store_true", help="Force synthetic asset creation")
    args = parser.parse_args()

    if not args.force_synthetic and check_ffmpeg():
        print("FFmpeg detected. Using native media transcoding.")
        generate_ffmpeg_assets(args.output, args.duration)
    else:
        if not check_ffmpeg():
            print("FFmpeg not found in PATH.")
        print("Using synthetic ISO-BMFF chunk generator.")
        generate_synthetic_assets(args.output, args.duration)


if __name__ == "__main__":
    main()
