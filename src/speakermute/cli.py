"""Command-line interface (``speakermute``)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from dotenv import find_dotenv, load_dotenv

from speakermute import __version__, ffmpeg
from speakermute.errors import SpeakermuteError
from speakermute.log import console, logger, setup_logging
from speakermute.paths import (
    DEFAULT_OUT_DIR,
    VideoOutputs,
    collect_videos,
    outputs_for,
)
from speakermute.pipeline import MuteResult, mute_video, transcribe_video
from speakermute.speakers import (
    format_clock,
    render_speaker_table,
    summarize_speakers,
)
from speakermute.srt import parse_rename
from speakermute.transcribe import (
    DEFAULT_DIARIZATION_MODEL,
    LEGACY_DIARIZATION_MODEL,
    TranscribeOptions,
    Transcriber,
    load_transcript,
)

app = typer.Typer(
    help=(
        "Transcribe and diarize a video, then mute one or more speakers while the picture, "
        "the length and everyone else's audio stay untouched.\n\n"
        "Typical workflow: [bold]speakers[/] (see who has which label) -> "
        "[bold]run --mute-speaker SPEAKER_XX[/]."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


class Device(StrEnum):
    auto = "auto"
    cuda = "cuda"
    cpu = "cpu"


# --------------------------------------------------------------------------- #
# Shared parameter definitions
# --------------------------------------------------------------------------- #

PathArg = Annotated[
    Path,
    typer.Argument(help="A video file or a folder of videos.", exists=True, readable=True),
]
VideoArg = Annotated[
    Path,
    typer.Argument(help="A video file.", exists=True, dir_okay=False, readable=True),
]
OutDirOpt = Annotated[
    Path,
    typer.Option("--out-dir", help="Output root; results go to <out-dir>/<video-stem>/."),
]
RenameOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--rename",
        metavar="OLD=NEW",
        help="Rename a speaker label in subtitles.srt, e.g. SPEAKER_00=Host. Repeatable.",
    ),
]
ModelOpt = Annotated[
    str,
    typer.Option(
        "--model",
        help="Whisper model (tiny, base, small, medium, large-v2, large-v3, ...). "
        "'auto' = large-v3 on CUDA, medium on CPU.",
    ),
]
DeviceOpt = Annotated[
    Device,
    typer.Option("--device", case_sensitive=False, help="'auto' picks cuda when available."),
]
ComputeTypeOpt = Annotated[
    str,
    typer.Option(
        "--compute-type",
        help="ctranslate2 compute type (int8, float16, float32, int8_float16, ...). "
        "'auto' = float16 on CUDA, int8 on CPU.",
    ),
]
MinSpeakersOpt = Annotated[
    int | None,
    typer.Option(
        "--min-speakers",
        min=1,
        help="Lower bound on the number of speakers for diarization (default: auto).",
    ),
]
MaxSpeakersOpt = Annotated[
    int | None,
    typer.Option(
        "--max-speakers",
        min=1,
        help="Upper bound on the number of speakers (default: auto). Setting both bounds to "
        "the real count, when you know it, makes diarization more accurate.",
    ),
]
LanguageOpt = Annotated[
    str,
    typer.Option("--language", help="Language code (en, de, ...), or 'auto' to detect."),
]
BatchSizeOpt = Annotated[
    int | None,
    typer.Option("--batch-size", min=1, help="Whisper batch size (default: 16 CUDA / 8 CPU)."),
]
DiarizationModelOpt = Annotated[
    str,
    typer.Option(
        "--diarization-model",
        help=f"pyannote pipeline to use. Default {DEFAULT_DIARIZATION_MODEL}; "
        f"the older {LEGACY_DIARIZATION_MODEL} also works.",
    ),
]
FillNearestOpt = Annotated[
    bool,
    typer.Option(
        "--fill-nearest/--no-fill-nearest",
        help="Give segments that overlap no diarization turn the nearest speaker "
        "instead of leaving them unlabelled (UNKNOWN is never muted).",
    ),
]
ForceOpt = Annotated[
    bool, typer.Option("--force", help="Re-transcribe even if transcript.json exists.")
]
DiarizeOpt = Annotated[
    bool,
    typer.Option(
        "--diarize/--no-diarize",
        help="Run speaker diarization (needs HF_TOKEN). --no-diarize gives a transcript "
        "without speaker labels, which cannot be used for muting.",
    ),
]
PadOpt = Annotated[
    float,
    typer.Option("--pad", min=0.0, help="Extra seconds muted before and after each segment."),
]
BurnSubsOpt = Annotated[
    bool,
    typer.Option("--burn-subs", help="Burn subtitles.srt into the muted video (re-encodes it)."),
]
OutputOpt = Annotated[
    Path | None,
    typer.Option(
        "--output",
        "-o",
        dir_okay=False,
        help="Where to write the muted video (default: <out-dir>/<stem>/<stem>_muted.mp4). "
        "transcript.json, subtitles.srt and mute_filter.txt still go to <out-dir>/<stem>/.",
    ),
]
CrfOpt = Annotated[
    int, typer.Option("--crf", min=0, max=51, help="x264 quality for --burn-subs (lower = better).")
]
PresetOpt = Annotated[str, typer.Option("--preset", help="x264 preset for --burn-subs.")]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@contextmanager
def error_boundary() -> Iterator[None]:
    """Turn our user-facing exceptions into a red message and exit code 1."""
    try:
        yield
    except SpeakermuteError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc


def _renames(items: list[str] | None) -> dict[str, str]:
    try:
        return parse_rename(items)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--rename") from exc


def _build_options(
    *,
    model: str,
    device: Device,
    compute_type: str,
    language: str,
    batch_size: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    diarization_model: str,
    fill_nearest: bool,
    diarize: bool = True,
) -> TranscribeOptions:
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise typer.BadParameter(
            "--min-speakers cannot be larger than --max-speakers", param_hint="--min-speakers"
        )
    lang = None if language.strip().lower() in ("", "auto") else language.strip()
    return TranscribeOptions(
        model=model,
        device=device.value,
        compute_type=compute_type,
        language=lang,
        batch_size=batch_size,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        diarization_model=diarization_model,
        fill_nearest=fill_nearest,
        diarize=diarize,
    )


def _transcribe_all(
    videos: list[Path],
    options: TranscribeOptions,
    out_dir: Path,
    *,
    force: bool,
    rename: Mapping[str, str],
) -> list[VideoOutputs]:
    """Transcribe every video that has no transcript yet (or all, with ``force``)."""
    pending: list[tuple[Path, VideoOutputs]] = []
    done: list[VideoOutputs] = []
    for video in videos:
        outputs = outputs_for(video, out_dir)
        if outputs.transcript.exists() and not force:
            logger.info(
                "Skipping %s: %s exists (use --force to redo)", video.name, outputs.transcript
            )
        else:
            pending.append((video, outputs))
        done.append(outputs)
    if not pending:
        return done

    transcriber = Transcriber(options)
    logger.info("WhisperX settings: %s", transcriber.describe())
    try:
        for index, (video, outputs) in enumerate(pending, start=1):
            logger.info("[%d/%d] Transcribing %s", index, len(pending), video.name)
            transcript = transcribe_video(video, outputs, transcriber, rename=rename)
            logger.info(
                "Wrote %s (%d segments, speakers: %s) and %s",
                outputs.transcript,
                len(transcript["segments"]),
                ", ".join(transcript["speakers"]) or "none",
                outputs.subtitles,
            )
    finally:
        transcriber.release()
    return done


def _report_mute(result: MuteResult) -> None:
    logger.info(
        "Muted %s for %s across %d range(s)%s -> %s",
        ", ".join(result.speakers),
        format_clock(result.muted_seconds),
        len(result.ranges),
        " with burned-in subtitles" if result.burned_subtitles else "",
        result.output,
    )
    if not result.duration_ok:
        logger.warning(
            "Output duration (%.3f s) differs from input (%.3f s)",
            result.output_duration,
            result.source_duration,
        )


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"speakermute {__version__}")
        raise typer.Exit()


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    load_dotenv(find_dotenv(usecwd=True))
    setup_logging(verbose)


@app.command()
def transcribe(
    path: PathArg,
    model: ModelOpt = "auto",
    device: DeviceOpt = Device.auto,
    compute_type: ComputeTypeOpt = "auto",
    min_speakers: MinSpeakersOpt = None,
    max_speakers: MaxSpeakersOpt = None,
    language: LanguageOpt = "en",
    batch_size: BatchSizeOpt = None,
    diarization_model: DiarizationModelOpt = DEFAULT_DIARIZATION_MODEL,
    fill_nearest: FillNearestOpt = True,
    diarize: DiarizeOpt = True,
    force: ForceOpt = False,
    rename: RenameOpt = None,
    out_dir: OutDirOpt = DEFAULT_OUT_DIR,
) -> None:
    """Transcribe, align and diarize video(s): writes transcript.json + subtitles.srt."""
    with error_boundary():
        ffmpeg.require_ffmpeg()
        renames = _renames(rename)
        videos = collect_videos(path)
        options = _build_options(
            model=model,
            device=device,
            compute_type=compute_type,
            language=language,
            batch_size=batch_size,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            diarization_model=diarization_model,
            fill_nearest=fill_nearest,
            diarize=diarize,
        )
        _transcribe_all(videos, options, out_dir, force=force, rename=renames)


@app.command()
def mute(
    video: VideoArg,
    speaker: Annotated[
        list[str],
        typer.Option(
            "--speaker",
            help="Label to mute, e.g. SPEAKER_01 (see `speakers`). Repeat to mute several.",
        ),
    ],
    pad: PadOpt = 0.15,
    burn_subs: BurnSubsOpt = False,
    transcript: Annotated[
        Path | None,
        typer.Option(
            "--transcript",
            exists=True,
            dir_okay=False,
            help="transcript.json to use (default: <out-dir>/<stem>/transcript.json). "
            "Hand-edit 'speaker' fields there and re-run without re-transcribing.",
        ),
    ] = None,
    rename: RenameOpt = None,
    crf: CrfOpt = 20,
    preset: PresetOpt = "medium",
    output: OutputOpt = None,
    out_dir: OutDirOpt = DEFAULT_OUT_DIR,
) -> None:
    """Mute speaker(s) using an existing transcript.json; writes <stem>_muted.mp4 (or --output)."""
    with error_boundary():
        ffmpeg.require_ffmpeg()
        outputs = outputs_for(video, out_dir)
        result = mute_video(
            video,
            outputs,
            speaker,
            pad=pad,
            burn_subs=burn_subs,
            rename=_renames(rename),
            transcript_path=transcript,
            crf=crf,
            preset=preset,
            output=output,
        )
        _report_mute(result)


@app.command()
def run(
    path: PathArg,
    mute_speaker: Annotated[
        list[str],
        typer.Option(
            "--mute-speaker",
            help="Label to mute, e.g. SPEAKER_01. Repeat to mute several speakers.",
        ),
    ],
    burn_subs: BurnSubsOpt = False,
    pad: PadOpt = 0.15,
    model: ModelOpt = "auto",
    device: DeviceOpt = Device.auto,
    compute_type: ComputeTypeOpt = "auto",
    min_speakers: MinSpeakersOpt = None,
    max_speakers: MaxSpeakersOpt = None,
    language: LanguageOpt = "en",
    batch_size: BatchSizeOpt = None,
    diarization_model: DiarizationModelOpt = DEFAULT_DIARIZATION_MODEL,
    fill_nearest: FillNearestOpt = True,
    force: ForceOpt = False,
    rename: RenameOpt = None,
    crf: CrfOpt = 20,
    preset: PresetOpt = "medium",
    output: OutputOpt = None,
    out_dir: OutDirOpt = DEFAULT_OUT_DIR,
) -> None:
    """Transcribe (if needed) and mute the chosen speaker(s) in one go."""
    with error_boundary():
        ffmpeg.require_ffmpeg()
        if burn_subs:
            ffmpeg.require_subtitles_filter()
        renames = _renames(rename)
        videos = collect_videos(path)
        if output is not None and len(videos) > 1:
            raise typer.BadParameter(
                f"names a single file, but {path} contains {len(videos)} videos",
                param_hint="--output",
            )
        options = _build_options(
            model=model,
            device=device,
            compute_type=compute_type,
            language=language,
            batch_size=batch_size,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            diarization_model=diarization_model,
            fill_nearest=fill_nearest,
        )
        all_outputs = _transcribe_all(videos, options, out_dir, force=force, rename=renames)
        for video, outputs in zip(videos, all_outputs, strict=True):
            logger.info("Muting %s", video.name)
            result = mute_video(
                video,
                outputs,
                mute_speaker,
                pad=pad,
                burn_subs=burn_subs,
                rename=renames,
                crf=crf,
                preset=preset,
                output=output,
            )
            _report_mute(result)


@app.command()
def speakers(
    video: VideoArg,
    lines: Annotated[
        int, typer.Option("--lines", min=1, help="How many of each speaker's first lines to show.")
    ] = 3,
    model: ModelOpt = "auto",
    device: DeviceOpt = Device.auto,
    compute_type: ComputeTypeOpt = "auto",
    min_speakers: MinSpeakersOpt = None,
    max_speakers: MaxSpeakersOpt = None,
    language: LanguageOpt = "en",
    batch_size: BatchSizeOpt = None,
    diarization_model: DiarizationModelOpt = DEFAULT_DIARIZATION_MODEL,
    fill_nearest: FillNearestOpt = True,
    force: ForceOpt = False,
    out_dir: OutDirOpt = DEFAULT_OUT_DIR,
) -> None:
    """Show each speaker label with talk time and first lines (transcribes first if needed)."""
    with error_boundary():
        ffmpeg.require_ffmpeg()
        outputs = outputs_for(video, out_dir)
        if force or not outputs.transcript.exists():
            options = _build_options(
                model=model,
                device=device,
                compute_type=compute_type,
                language=language,
                batch_size=batch_size,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                diarization_model=diarization_model,
                fill_nearest=fill_nearest,
            )
            _transcribe_all([video], options, out_dir, force=force, rename={})
        transcript = load_transcript(outputs.transcript)
        summaries = summarize_speakers(transcript["segments"], lines=lines)
        console.print(render_speaker_table(summaries, title=f"Speakers in {video.name}"))
        console.print(
            "[dim]Labels are assigned by order of first speech, not by role. "
            "Once you know which label(s) to silence (repeat --mute-speaker for several):[/]\n"
            f"  speakermute run {_quote(video)} --mute-speaker <LABEL> --burn-subs"
        )


def _quote(path: Path) -> str:
    text = str(path)
    return f'"{text}"' if " " in text else text


if __name__ == "__main__":  # pragma: no cover
    app()
