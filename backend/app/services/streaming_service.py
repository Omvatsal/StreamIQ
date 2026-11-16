"""Discovery and safe serving of processed DASH assets."""

from pathlib import Path


class StreamingService:
    def __init__(self, processed_root: str | Path = "video_assets/processed"):
        root = Path(processed_root)
        if not root.is_absolute() and str(processed_root) == "video_assets/processed":
            root = Path(__file__).resolve().parents[3] / root
        self.root = root.resolve()

    def list_videos(self) -> list[dict[str, str]]:
        if not self.root.exists():
            return []
        videos = []
        for directory in sorted(path for path in self.root.iterdir() if path.is_dir()):
            manifest = directory / "manifest.mpd"
            if manifest.is_file():
                videos.append({"id": directory.name, "manifest": f"/media/{directory.name}/manifest.mpd"})
        return videos

    def manifest_path(self, video_id: str) -> Path:
        if not video_id or Path(video_id).name != video_id:
            raise ValueError("invalid video id")
        manifest = (self.root / video_id / "manifest.mpd").resolve()
        if self.root not in manifest.parents or not manifest.is_file():
            raise FileNotFoundError(video_id)
        return manifest

    def directory(self, video_id: str) -> Path:
        manifest = self.manifest_path(video_id)
        return manifest.parent
