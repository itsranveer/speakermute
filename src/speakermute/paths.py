"""Output layout: everything for a video lives in ``out/<video-stem>/``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
DEFAULT_OUT_DIR = Path("out")

# Containers the muted video may be written to. The video stream is copied as is,
# so the output container must accept whatever codec the input holds: MP4-family
# inputs stay MP4-family, Matroska stays Matroska, and everything else (WebM,
# whose VP8/VP9 + Opus/Vorbis cannot be mixed with the AAC audio we write) goes
# to Matroska, which accepts any codec.
MUTED_CONTAINERS = {".mp4", ".mov", ".m4v", ".mkv"}
FALLBACK_MUTED_SUFFIX = ".mkv"


def muted_suffix(video: str | Path) -> str:
    """Return the suffix for the muted copy of ``video`` (see :data:`MUTED_CONTAINERS`)."""
    suffix = Path(video).suffix.lower()
    return suffix if suffix in MUTED_CONTAINERS else FALLBACK_MUTED_SUFFIX


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
        return self.dir / f"{self.stem}_muted{muted_suffix(self.video)}"

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
