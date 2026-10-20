"""Output layout: everything for a video lives in ``out/<video-stem>/``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
DEFAULT_OUT_DIR = Path("out")


@dataclass(frozen=True)
class VideoOutputs:
    video: Path
    dir: Path

    @property
    def stem(self) -> str:
        return self.video.stem

    @property
    def transcript(self) -> Path:
        return self.dir / "transcript.json"

    @property
    def subtitles(self) -> Path:
        return self.dir / "subtitles.srt"

    @property
    def muted(self) -> Path:
        return self.dir / f"{self.stem}_muted.mp4"

    @property
    def filter_script(self) -> Path:
        return self.dir / "mute_filter.txt"


def outputs_for(video: str | Path, out_root: str | Path = DEFAULT_OUT_DIR) -> VideoOutputs:
    video = Path(video)
    return VideoOutputs(video=video, dir=Path(out_root) / video.stem)


def collect_videos(path: str | Path) -> list[Path]:
    """Return the video(s) referenced by ``path`` (a file or a folder)."""
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        videos = [
            p
            for p in sorted(path.iterdir())
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if not videos:
            raise FileNotFoundError(
                f"no videos ({', '.join(sorted(VIDEO_EXTENSIONS))}) found in {path}"
            )
        return videos
    raise FileNotFoundError(f"{path} does not exist")
