"""WhisperX wrapper: transcription, word-level alignment and speaker diarization.

Written against the API of whisperx 3.8.6 (checked in the installed package):

* ``whisperx.load_model(whisper_arch, device, compute_type=..., language=..., threads=...)``
* ``pipeline.transcribe(audio, batch_size=..., language=..., progress_callback=cb)``
* ``whisperx.load_align_model(language_code=..., device=...)``
* ``whisperx.align(segments, model, metadata, audio, device, return_char_alignments=False,
  progress_callback=cb)``
* ``whisperx.diarize.DiarizationPipeline(model_name=..., token=..., device=...)`` -- note:
  not re-exported from the top-level package, and the keyword is ``token`` (older
  versions used ``use_auth_token``). ``pipeline(audio, min_speakers=..., max_speakers=...,
  progress_callback=cb)`` returns a pandas DataFrame.
* ``whisperx.assign_word_speakers(diarize_df, result, fill_nearest=...)``

whisperx is imported lazily so that ``--help`` and the unit tests do not pay the
multi-second torch import cost.
"""

from __future__ import annotations

import datetime as dt
import gc
import json
import os
import warnings
import wave
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from speakermute import __version__
from speakermute.errors import MissingTokenError, TranscriptError
from speakermute.log import console

SAMPLE_RATE = 16000
TRANSCRIPT_VERSION = 1
HF_TOKEN_ENV = "HF_TOKEN"
# pyannote.audio 4 reports anonymous usage metrics unless this is "false". The
# variable is read when pyannote is first imported, so it must be set before.
PYANNOTE_TELEMETRY_ENV = "PYANNOTE_METRICS_ENABLED"

# whisperx 3.8 / pyannote.audio 4 default. The older 3.1 pipeline still works and
# can be selected with --diarization-model.
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
LEGACY_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

MISSING_TOKEN_MESSAGE = (
    "speaker diarization needs a Hugging Face token but none was found.\n"
    "  1. Create a token (read access) at https://huggingface.co/settings/tokens\n"
    "  2. Accept the model terms while logged in:\n"
    "       https://huggingface.co/{model}\n"
    "       (for the legacy pipeline also https://huggingface.co/pyannote/segmentation-3.0)\n"
    "  3. Put HF_TOKEN=... in a .env file next to where you run the tool "
    f"(or export {HF_TOKEN_ENV})"
)


@dataclass
class TranscribeOptions:
    model: str = "auto"
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = "en"
    batch_size: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    diarize: bool = True
    diarization_model: str = DEFAULT_DIARIZATION_MODEL
    fill_nearest: bool = True
    hf_token: str | None = None
    threads: int | None = None


# --------------------------------------------------------------------------- #
# Defaults / detection
# --------------------------------------------------------------------------- #


def detect_device() -> str:
    """Return ``"cuda"`` when torch can see a CUDA GPU, otherwise ``"cpu"``.

    ctranslate2 (which runs Whisper) does not support Apple MPS, so macOS is CPU.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - whisperx always brings torch
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_device(device: str | None) -> str:
    device = (device or "auto").lower()
    if device == "auto":
        return detect_device()
    if device not in ("cuda", "cpu"):
        raise ValueError(f"unsupported device {device!r}; use auto, cuda or cpu")
    if device == "cuda" and detect_device() != "cuda":
        raise ValueError("--device cuda was requested but torch cannot see a CUDA GPU")
    return device


def resolve_model(model: str | None, device: str) -> str:
    if not model or model.lower() == "auto":
        return "large-v3" if device == "cuda" else "medium"
    return model


def resolve_compute_type(compute_type: str | None, device: str) -> str:
    if not compute_type or compute_type.lower() == "auto":
        return "float16" if device == "cuda" else "int8"
    return compute_type


def resolve_batch_size(batch_size: int | None, device: str) -> int:
    if batch_size and batch_size > 0:
        return batch_size
    return 16 if device == "cuda" else 8


def disable_pyannote_telemetry() -> None:
    """Opt out of pyannote's usage telemetry unless the user set the variable themselves."""
    os.environ.setdefault(PYANNOTE_TELEMETRY_ENV, "false")


def silence_torchcodec_warning() -> None:
    """Hide pyannote.audio 4's import-time "torchcodec is not installed correctly" warning.

    pyannote only ever receives audio that was already decoded by ffmpeg (as an
    in-memory waveform), so torchcodec is never used and the warning is noise.
    """
    warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")


def resolve_hf_token(explicit: str | None = None) -> str | None:
    """Explicit value, then ``HF_TOKEN``, then the ``huggingface-cli login`` cache."""
    if explicit:
        return explicit
    from_env = os.environ.get(HF_TOKEN_ENV)
    if from_env:
        return from_env
    try:
        from huggingface_hub import get_token
    except ImportError:  # pragma: no cover
        return None
    try:
        return get_token()
    except Exception:  # pragma: no cover - defensive
        return None


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


def load_wav(path: str | Path, expected_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load a mono 16-bit PCM WAV (as written by :func:`ffmpeg.extract_audio`) as float32.

    This is the same layout ``whisperx.load_audio`` produces, but without
    spawning a second ffmpeg process.
    """
    with wave.open(str(path), "rb") as wav:
        channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        if (channels, width, rate) != (1, 2, expected_rate):
            raise ValueError(
                f"{path}: expected mono 16-bit {expected_rate} Hz WAV, "
                f"got {channels} ch / {width * 8}-bit / {rate} Hz"
            )
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


# --------------------------------------------------------------------------- #
# WhisperX pipeline
# --------------------------------------------------------------------------- #


@contextmanager
def _step(progress: Progress, description: str) -> Iterator[None]:
    """Show an indeterminate spinner row while a model loads."""
    task = progress.add_task(description, total=None)
    try:
        yield
    finally:
        progress.update(task, total=1, completed=1)


def _progress_bar() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


class Transcriber:
    """Holds the WhisperX models so a batch of videos reuses them.

    Models load lazily on first use. Call :meth:`release` when done to free
    memory (and GPU memory) between large batches.
    """

    def __init__(self, options: TranscribeOptions):
        disable_pyannote_telemetry()
        silence_torchcodec_warning()
        self.options = options
        self.device = resolve_device(options.device)
        self.model_name = resolve_model(options.model, self.device)
        self.compute_type = resolve_compute_type(options.compute_type, self.device)
        self.batch_size = resolve_batch_size(options.batch_size, self.device)
        self.threads = options.threads or max(1, os.cpu_count() or 4)
        self.hf_token: str | None = None
        if options.diarize:
            # Resolve up front so a missing token fails before any heavy work.
            self.hf_token = resolve_hf_token(options.hf_token)
            if not self.hf_token:
                raise MissingTokenError(
                    MISSING_TOKEN_MESSAGE.format(model=options.diarization_model)
                )
        self._asr: Any = None
        self._align: dict[str, tuple[Any, dict]] = {}
        self._diarizer: Any = None

    # -- settings -------------------------------------------------------- #

    def describe(self) -> str:
        parts = [
            f"model={self.model_name}",
            f"device={self.device}",
            f"compute_type={self.compute_type}",
            f"batch_size={self.batch_size}",
            f"language={self.options.language or 'auto'}",
        ]
        if self.options.diarize:
            lo, hi = self.options.min_speakers, self.options.max_speakers
            if lo is None and hi is None:
                parts.append("speakers=auto")
            else:
                parts.append(f"speakers={lo or '?'}-{hi or '?'}")
            parts.append(f"diarization={self.options.diarization_model}")
        else:
            parts.append("diarization=off")
        return " ".join(parts)

    def settings(self) -> dict[str, Any]:
        data = asdict(self.options)
        data.pop("hf_token", None)
        data.update(
            model=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            batch_size=self.batch_size,
            whisperx_version=_whisperx_version(),
        )
        return data

    # -- lazy loading ---------------------------------------------------- #

    def _load_asr(self) -> None:
        import whisperx

        self._asr = whisperx.load_model(
            self.model_name,
            self.device,
            compute_type=self.compute_type,
            language=self.options.language,
            threads=self.threads,
        )

    def _load_align(self, language: str) -> None:
        import whisperx

        model, metadata = whisperx.load_align_model(language_code=language, device=self.device)
        self._align[language] = (model, metadata)

    def _load_diarizer(self) -> None:
        from whisperx.diarize import DiarizationPipeline

        self._diarizer = DiarizationPipeline(
            model_name=self.options.diarization_model,
            token=self.hf_token,
            device=self.device,
        )

    def release(self) -> None:
        """Drop model references and free (GPU) memory."""
        self._asr = None
        self._align.clear()
        self._diarizer = None
        gc.collect()
        if self.device == "cuda":
            import torch

            torch.cuda.empty_cache()

    # -- the pipeline ---------------------------------------------------- #

    def transcribe_audio(self, audio: np.ndarray) -> dict[str, Any]:
        """Run transcribe -> align -> (diarize + assign speakers) on 16 kHz mono audio.

        Returns the raw WhisperX result: ``{"segments": [...], "language": "en", ...}``
        where each segment has ``start``, ``end``, ``text``, ``words`` and (when
        diarization is on) ``speaker``.
        """
        import whisperx

        opts = self.options
        with _progress_bar() as progress:
            if self._asr is None:
                with _step(
                    progress,
                    f"Loading Whisper model {self.model_name} ({self.device}, {self.compute_type})",
                ):
                    self._load_asr()

            task = progress.add_task("Transcribing", total=100)
            result = self._asr.transcribe(
                audio,
                batch_size=self.batch_size,
                language=opts.language,
                progress_callback=lambda pct: progress.update(task, completed=pct),
            )
            progress.update(task, completed=100)
            language = result.get("language") or opts.language or "en"

            if language not in self._align:
                with _step(progress, f"Loading alignment model ({language})"):
                    self._load_align(language)
            align_model, metadata = self._align[language]
            task = progress.add_task("Aligning words", total=100)
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
                progress_callback=lambda pct: progress.update(task, completed=pct),
            )
            progress.update(task, completed=100)
            result["language"] = language

            if opts.diarize:
                if self._diarizer is None:
                    with _step(progress, "Loading diarization model"):
                        self._load_diarizer()
                task = progress.add_task("Diarizing speakers", total=100)
                diarize_df = self._diarizer(
                    audio,
                    min_speakers=opts.min_speakers,
                    max_speakers=opts.max_speakers,
                    progress_callback=lambda pct: progress.update(task, completed=pct),
                )
                progress.update(task, completed=100)
                result = whisperx.assign_word_speakers(
                    diarize_df, result, fill_nearest=opts.fill_nearest
                )
        return result


def _whisperx_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("whisperx")
    except Exception:  # pragma: no cover
        return None


# --------------------------------------------------------------------------- #
# transcript.json
# --------------------------------------------------------------------------- #


def _f(value: Any) -> float:
    return round(float(value), 3)


def normalize_segments(raw_segments: list[Mapping]) -> list[dict[str, Any]]:
    """Reduce WhisperX segments to plain JSON-friendly dicts.

    Key order is chosen for hand editing: ``start``, ``end``, ``speaker``,
    ``text``, ``words``.
    """
    segments: list[dict[str, Any]] = []
    for seg in raw_segments:
        entry: dict[str, Any] = {"start": _f(seg["start"]), "end": _f(seg["end"])}
        if seg.get("speaker"):
            entry["speaker"] = str(seg["speaker"])
        entry["text"] = " ".join(str(seg.get("text") or "").split())
        words: list[dict[str, Any]] = []
        for word in seg.get("words") or []:
            item: dict[str, Any] = {"word": str(word.get("word", "")).strip()}
            for key in ("start", "end", "score"):
                if word.get(key) is not None:
                    item[key] = _f(word[key])
            if word.get("speaker"):
                item["speaker"] = str(word["speaker"])
            words.append(item)
        entry["words"] = words
        segments.append(entry)
    return segments


def build_transcript(
    result: Mapping[str, Any],
    *,
    video: Path,
    duration: float | None,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    segments = normalize_segments(list(result.get("segments", [])))
    speakers = sorted({seg["speaker"] for seg in segments if "speaker" in seg})
    return {
        "version": TRANSCRIPT_VERSION,
        "tool": f"speakermute {__version__}",
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "video": video.name,
        "video_path": str(video.resolve()),
        "duration": _f(duration) if duration is not None else None,
        "language": result.get("language"),
        "settings": dict(settings or {}),
        "speakers": speakers,
        "segments": segments,
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def save_transcript(path: str | Path, transcript: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(transcript, fh, indent=2, ensure_ascii=False, default=_json_default)
        fh.write("\n")
    return path


def load_transcript(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise TranscriptError(f"{path} does not exist. Run `speakermute transcribe <video>` first.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TranscriptError(f"{path} is not valid JSON (line {exc.lineno}): {exc.msg}") from exc
    segments = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(segments, list):
        raise TranscriptError(f"{path} has no 'segments' list")
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict) or "start" not in seg or "end" not in seg:
            raise TranscriptError(f"{path}: segment #{i} must be an object with 'start' and 'end'")
    return data
