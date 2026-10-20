"""High-level steps used by the CLI: transcribe a video, mute one or more speakers."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speakermute import ffmpeg
from speakermute.errors import SpeakermuteError, SpeakerNotFoundError
from speakermute.log import logger
from speakermute.paths import VideoOutputs
from speakermute.ranges import (
    Range,
    build_mute_filtergraph,
    speaker_ranges,
    total_duration,
)
from speakermute.srt import write_srt
from speakermute.transcribe import (
    Transcriber,
    build_transcript,
    load_transcript,
    load_wav,
    save_transcript,
)


def transcribe_video(
    video: Path,
    outputs: VideoOutputs,
    transcriber: Transcriber,
    *,
    rename: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Extract audio, run WhisperX, write ``transcript.json`` and ``subtitles.srt``."""
    duration = ffmpeg.probe_duration(video)
    with tempfile.TemporaryDirectory(prefix="speakermute-audio-") as tmp:
        wav = ffmpeg.extract_audio(video, Path(tmp) / "audio.wav", total_duration=duration)
        audio = load_wav(wav)
    result = transcriber.transcribe_audio(audio)
    del audio
    transcript = build_transcript(
        result, video=video, duration=duration, settings=transcriber.settings()
    )
    save_transcript(outputs.transcript, transcript)
    write_srt(outputs.subtitles, transcript["segments"], rename)
    return transcript


@dataclass
class MuteResult:
    speakers: list[str]
    output: Path
    burned_subtitles: bool
    ranges: list[Range]
    muted_seconds: float
    source_duration: float
    output_duration: float
    duration_ok: bool


def mute_video(
    video: Path,
    outputs: VideoOutputs,
    speakers: Sequence[str],
    *,
    pad: float = 0.15,
    burn_subs: bool = False,
    rename: Mapping[str, str] | None = None,
    transcript_path: Path | None = None,
    crf: int = 20,
    preset: str = "medium",
    output: Path | None = None,
) -> MuteResult:
    """Mute every segment of ``speakers`` using the (possibly hand-edited) transcript.

    Writes ``<stem>_muted.mp4`` next to the transcript, or ``output`` if given.
    Without ``burn_subs`` the video stream is copied; with it the subtitles are
    burned in during the same ffmpeg pass. Either way the output is exactly as
    long as the input. ``subtitles.srt`` is regenerated from the transcript so
    edits show up there too.
    """
    if burn_subs:
        ffmpeg.require_subtitles_filter()
    target = output or outputs.muted
    if target.resolve() == Path(video).resolve():
        # ffmpeg runs with -y and would truncate the file it is reading from.
        raise SpeakermuteError(
            f"output {target} is the input video itself; choose another --output path"
        )

    transcript = load_transcript(transcript_path or outputs.transcript)
    segments = transcript["segments"]
    speakers = list(dict.fromkeys(speakers))  # de-duplicate, keep order
    if not speakers:
        raise SpeakerNotFoundError("no speaker label given")
    labels = sorted({seg["speaker"] for seg in segments if seg.get("speaker")})
    missing = [s for s in speakers if s not in labels]
    if missing:
        available = ", ".join(labels) or "(none - was diarization enabled?)"
        raise SpeakerNotFoundError(
            f"speaker(s) {', '.join(missing)} do not occur in the transcript. "
            f"Available labels: {available}.\n"
            "Run `speakermute speakers <video>` to see who is who."
        )
    if set(speakers) == set(labels):
        logger.warning("muting every speaker (%s); the output will be silent", ", ".join(speakers))

    duration = ffmpeg.probe_duration(video)
    ranges = speaker_ranges(segments, speakers, pad=pad, duration=duration)
    if not ranges:
        raise SpeakerNotFoundError(
            f"speaker(s) {', '.join(speakers)} have no non-empty segments to mute"
        )

    script = ffmpeg.write_filter_script(outputs.filter_script, build_mute_filtergraph(ranges))
    write_srt(outputs.subtitles, segments, rename)
    logger.info(
        "Muting %s: %d range(s), %.1f s of speech (pad %.2f s)",
        ", ".join(speakers),
        len(ranges),
        total_duration(ranges),
        pad,
    )
    ffmpeg.mute_with_filter_script(
        video,
        target,
        script,
        subtitles=outputs.subtitles if burn_subs else None,
        crf=crf,
        preset=preset,
        total_duration=duration,
    )
    src_duration, out_duration, ok = ffmpeg.check_same_duration(video, target)
    if not ok:
        logger.warning(
            "duration changed: input %.3f s, output %.3f s (%s)",
            src_duration,
            out_duration,
            target,
        )

    return MuteResult(
        speakers=speakers,
        output=target,
        burned_subtitles=burn_subs,
        ranges=ranges,
        muted_seconds=total_duration(ranges),
        source_duration=src_duration,
        output_duration=out_duration,
        duration_ok=ok,
    )
