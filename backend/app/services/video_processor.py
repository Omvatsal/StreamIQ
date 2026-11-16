"""FFmpeg utilities for producing aligned MPEG-DASH representations."""

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Optional, Sequence


@dataclass(frozen=True)
class Representation:
    name: str
    height: int
    bitrate_kbps: int


DEFAULT_REPRESENTATIONS = (
    Representation("240p", 240, 400),
    Representation("360p", 360, 800),
    Representation("480p", 480, 1400),
    Representation("720p", 720, 2800),
    Representation("1080p", 1080, 5000),
)


class VideoProcessor:
    """Build and execute a deterministic FFmpeg DASH command."""

    def __init__(
        self,
        processed_root: str | Path = "video_assets/processed",
        ffmpeg_binary: str = "ffmpeg",
    ):
        root = Path(processed_root)
        if not root.is_absolute() and str(processed_root) == "video_assets/processed":
            root = Path(__file__).resolve().parents[3] / root
        self.processed_root = root
        self.ffmpeg_binary = ffmpeg_binary

    def build_command(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        segment_duration: float = 4.0,
        representations: Sequence[Representation] = DEFAULT_REPRESENTATIONS,
    ) -> list[str]:
        if segment_duration <= 0:
            raise ValueError("segment_duration must be positive")
        if not representations:
            raise ValueError("at least one representation is required")
        source = Path(input_path)
        destination = Path(output_dir)
        split_labels = "".join(f"[v{index}]" for index in range(len(representations)))
        filters = [f"[0:v]split={len(representations)}{split_labels}"]
        filters.extend(
            f"[v{index}]fps=24,scale=-2:{representation.height}:flags=lanczos[v{index}out]"
            for index, representation in enumerate(representations)
        )
        command = [
            self.ffmpeg_binary,
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            ";".join(filters),
        ]
        for index, representation in enumerate(representations):
            command.extend(
                [
                    "-map",
                    f"[v{index}out]",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-b:v",
                    f"{representation.bitrate_kbps}k",
                    "-maxrate",
                    f"{representation.bitrate_kbps}k",
                    "-bufsize",
                    f"{representation.bitrate_kbps * 2}k",
                    "-g",
                    "96",
                    "-keyint_min",
                    "96",
                    "-sc_threshold",
                    "0",
                    "-an",
                ]
            )
        adaptation_sets = " ".join(
            f"id={index},streams={index}" for index in range(len(representations))
        )
        command.extend(
            [
                "-f",
                "dash",
                "-seg_duration",
                str(segment_duration),
                "-use_template",
                "1",
                "-use_timeline",
                "1",
                "-init_seg_name",
                "init-$RepresentationID$.m4s",
                "-media_seg_name",
                "seg-$RepresentationID$-$Number%05d$.m4s",
                "-adaptation_sets",
                adaptation_sets,
                str(destination / "manifest.mpd"),
            ]
        )
        return command

    def process(
        self,
        input_path: str | Path,
        video_id: Optional[str] = None,
        segment_duration: float = 4.0,
    ) -> Path:
        source = Path(input_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        identifier = video_id or source.stem
        output_dir = self.processed_root / identifier
        output_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which(self.ffmpeg_binary) is None:
            raise RuntimeError(f"FFmpeg executable not found: {self.ffmpeg_binary}")
        command = self.build_command(source, output_dir, segment_duration)
        subprocess.run(command, check=True)
        manifest = output_dir / "manifest.mpd"
        if not manifest.is_file():
            raise RuntimeError(f"FFmpeg completed without creating {manifest}")
        return manifest
