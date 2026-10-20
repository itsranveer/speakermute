"""Thin wrappers around ``ffmpeg`` / ``ffprobe`` invoked via subprocess."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from speakermute.errors import SpeakermuteError
from speakermute.log import console, logger

# Environment overrides (also picked up from .env) so a build with extra
# features, e.g. Homebrew's keg-only ``ffmpeg-full``, can be used without
# changing PATH.
FFMPEG_ENV = "FFMPEG"
FFPROBE_ENV = "FFPROBE"

# Output containers that use the MP4/MOV muxer (and understand ``-movflags``).
MP4_FAMILY = {".mp4", ".mov", ".m4v"}

INSTALL_HINT = (
    "Install ffmpeg and make sure it is on your PATH:\n"
    "  macOS:          brew install ffmpeg\n"
    "  Debian/Ubuntu:  sudo apt install ffmpeg\n"
    "  Windows:        winget install ffmpeg  (or https://ffmpeg.org/download.html)\n"
    f"or point the {FFMPEG_ENV} / {FFPROBE_ENV} environment variables at the executables."
)


class FFmpegError(SpeakermuteError):
    """Raised when ffmpeg/ffprobe is missing or a command fails."""


def ffmpeg_executable() -> str:
    return os.environ.get(FFMPEG_ENV) or "ffmpeg"


def ffprobe_executable() -> str:
    explicit = os.environ.get(FFPROBE_ENV)
    if explicit:
        return explicit
    # If FFMPEG points at a specific build, prefer the ffprobe that sits next to it.
    ffmpeg = os.environ.get(FFMPEG_ENV)
    if ffmpeg:
        sibling = Path(ffmpeg).with_name("ffprobe" + Path(ffmpeg).suffix)
        if sibling.exists():
            return str(sibling)
    return "ffprobe"


def require_ffmpeg() -> None:
    """Fail fast with a clear message if ffmpeg or ffprobe cannot be found."""
    missing = [
        name for name in (ffmpeg_executable(), ffprobe_executable()) if shutil.which(name) is None
    ]
    if missing:
        raise FFmpegError(f"could not find {', '.join(missing)}.\n{INSTALL_HINT}")


def available_filters() -> set[str]:
    """Return the names of all filters the ffmpeg build supports."""
    proc = subprocess.run(
        [ffmpeg_executable(), "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        check=False,
    )
    names: set[str] = set()
    for line in proc.stdout.splitlines():
        # lines look like: " T.. volume            A->A       Change input volume."
        match = re.match(r"^\s*[A-Z.]+\s+(\S+)\s+\S+->\S+", line)
        if match:
            names.add(match.group(1))
    return names


def require_subtitles_filter() -> None:
    """Burning subtitles needs an ffmpeg built with libass (``subtitles`` filter)."""
    if "subtitles" not in available_filters():
        raise FFmpegError(
            f"your ffmpeg ({shutil.which(ffmpeg_executable())}) was built without libass, "
            "so it has no 'subtitles' filter and cannot burn subtitles.\n"
            "  macOS: brew install ffmpeg-full, then set\n"
            "         FFMPEG=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg in your .env\n"
            "  Linux: install a distro ffmpeg package built with libass (most are)."
        )


_VERSION_RE = re.compile(r"ffmpeg version\s+[nv]?(\d+)\.(\d+)")


def ffmpeg_version() -> tuple[int, int] | None:
    """Return ``(major, minor)`` of the ffmpeg build, or ``None`` for git snapshots."""
    proc = subprocess.run(
        [ffmpeg_executable(), "-version"], capture_output=True, text=True, check=False
    )
    match = _VERSION_RE.search(proc.stdout)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def filter_script_args(stream: str, script: str | Path) -> list[str]:
    """Arguments that load a filtergraph for ``stream`` (``a``/``v``) from a file.

    ffmpeg 7.0 introduced the generic ``-/OPTION FILE`` syntax and deprecated
    ``-filter_script``; ffmpeg 9 removed the old option entirely. Unversioned
    (git) builds are assumed to be recent.
    """
    version = ffmpeg_version()
    if version is None or version >= (7, 0):
        return [f"-/filter:{stream}", str(script)]
    return [f"-filter_script:{stream}", str(script)]


def probe_duration(path: str | Path) -> float:
    """Return the container duration of a media file in seconds."""
    cmd = [
        ffprobe_executable(),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed for {path}:\n{proc.stderr.strip()}")
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise FFmpegError(f"could not parse duration for {path}: {proc.stdout!r}") from exc


_PROGRESS_RE = re.compile(r"^(?P<key>[a-z_]+)=(?P<value>.*)$")


def parse_progress_line(line: str) -> float | None:
    """Return the output time in seconds from an ``-progress`` line, or ``None``.

    ffmpeg prints ``out_time_us`` and ``out_time_ms`` (both are microseconds
    due to a long-standing quirk) plus ``out_time=HH:MM:SS.micro``.
    """
    match = _PROGRESS_RE.match(line.strip())
    if not match:
        return None
    key, value = match.group("key"), match.group("value").strip()
    if key in ("out_time_us", "out_time_ms"):
        try:
            return max(0.0, int(value) / 1_000_000)
        except ValueError:
            return None
    return None


def is_progress_line(line: str) -> bool:
    return bool(_PROGRESS_RE.match(line.strip()))


def run_ffmpeg(
    args: Sequence[str],
    *,
    description: str = "ffmpeg",
    total_duration: float | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> None:
    """Run ``ffmpeg`` with ``args`` and render a progress bar.

    Progress is read from ``-progress pipe:1``. Any non-progress output (which,
    at ``-loglevel error``, means errors) is collected and included in the
    raised :class:`FFmpegError` when ffmpeg exits non-zero.
    """
    cmd = [
        ffmpeg_executable(),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-nostats",
        "-progress",
        "pipe:1",
        *args,
    ]
    logger.debug("running: %s", " ".join(cmd))

    error_lines: list[str] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=total_duration)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            seconds = parse_progress_line(line)
            if seconds is not None:
                if total_duration:
                    progress.update(task, completed=min(seconds, total_duration))
                if on_progress:
                    on_progress(seconds)
            elif not is_progress_line(line) and line.strip():
                error_lines.append(line.rstrip())
        returncode = proc.wait()
        if returncode == 0 and total_duration:
            progress.update(task, completed=total_duration)

    if returncode != 0:
        detail = "\n".join(error_lines[-20:]) or "(no error output)"
        raise FFmpegError(f"ffmpeg exited with code {returncode} ({description}):\n{detail}")


def extract_audio(
    video: str | Path,
    wav_path: str | Path,
    *,
    sample_rate: int = 16000,
    total_duration: float | None = None,
) -> Path:
    """Extract mono 16-bit PCM audio (default 16 kHz, what Whisper expects)."""
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ],
        description="Extracting audio",
        total_duration=total_duration,
    )
    return wav_path


def write_filter_script(path: str | Path, filtergraph: str) -> Path:
    """Write a filtergraph to a file for ``-filter_script``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(filtergraph, encoding="utf-8")
    return path


def mute_with_filter_script(
    video: str | Path,
    output: str | Path,
    filter_script: str | Path,
    *,
    subtitles: str | Path | None = None,
    crf: int = 20,
    preset: str = "medium",
    audio_bitrate: str = "192k",
    total_duration: float | None = None,
) -> Path:
    """Apply an audio filtergraph from a script file, optionally burning subtitles.

    Without ``subtitles`` the video stream is copied bit-for-bit (``-c:v copy``)
    so the output has the same length and quality as the input; only the audio
    is re-encoded (AAC). With ``subtitles`` (an SRT path) the video is
    re-encoded with x264 and the ``subtitles`` filter in the same pass, so there
    is never an intermediate file. The filtergraph is read from
    ``filter_script`` because the mute expression can be very long.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    args = ["-i", str(video), *filter_script_args("a", filter_script)]
    if subtitles is None:
        args += ["-c:v", "copy"]
        description = "Muting audio"
    else:
        srt = Path(subtitles).resolve()
        args += [
            "-vf",
            f"subtitles=filename={escape_filter_path(srt)}",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
        ]
        description = "Muting audio + burning subtitles"
    args += ["-c:a", "aac", "-b:a", audio_bitrate]
    if output.suffix.lower() in MP4_FAMILY:
        # Move the index to the front so the file starts playing before it is fully
        # downloaded. This is an MP4/MOV muxer option; Matroska has no equivalent.
        args += ["-movflags", "+faststart"]
    run_ffmpeg([*args, str(output)], description=description, total_duration=total_duration)
    return output


def escape_filter_path(path: str | Path) -> str:
    """Escape a path for use as a filter option value, e.g. ``subtitles=filename=...``.

    ffmpeg parses a filter option value twice: once as part of the filtergraph
    (where ``\\``, ``'``, ``[``, ``]``, ``,`` and ``;`` are special) and once as
    an option string (where ``\\``, ``'``, ``:`` and ``=`` are special). Both
    levels use backslash escaping, so we escape for the inner level first and
    then for the outer level. This handles Windows drive letters (``C:``),
    quotes, spaces and brackets without relying on shell quoting.
    """
    text = str(path)
    inner = re.sub(r"([\\':=,;\[\]\s])", r"\\\1", text)
    outer = re.sub(r"([\\'\[\],;])", r"\\\1", inner)
    return outer


def check_same_duration(
    source: str | Path, result: str | Path, tolerance: float = 0.1
) -> tuple[float, float, bool]:
    """Compare container durations; returns ``(source_s, result_s, ok)``."""
    src = probe_duration(source)
    out = probe_duration(result)
    return src, out, abs(src - out) <= tolerance
