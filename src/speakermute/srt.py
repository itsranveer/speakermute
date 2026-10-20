"""SubRip (.srt) writing with speaker labels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

UNKNOWN_SPEAKER = "UNKNOWN"


def format_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp ``HH:MM:SS,mmm``."""
    total_ms = max(0, round(float(seconds) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_rename(items: Iterable[str] | None) -> dict[str, str]:
    """Parse ``OLD=NEW`` strings (e.g. from repeated ``--rename`` flags) into a dict."""
    rename: dict[str, str] = {}
    for item in items or []:
        old, sep, new = item.partition("=")
        old, new = old.strip(), new.strip()
        if not sep or not old or not new:
            raise ValueError(
                f"invalid rename {item!r}: expected the form OLD=NEW, e.g. SPEAKER_00=Host"
            )
        rename[old] = new
    return rename


def speaker_label(
    segment: Mapping,
    rename: Mapping[str, str] | None = None,
    unknown: str = UNKNOWN_SPEAKER,
) -> str:
    """Return the display label for a segment's speaker, applying renames."""
    raw = segment.get("speaker") or unknown
    if rename and raw in rename:
        return rename[raw]
    return raw


def _clean_text(text: object) -> str:
    return " ".join(str(text or "").split())


def segments_to_srt(
    segments: Iterable[Mapping],
    rename: Mapping[str, str] | None = None,
    unknown: str = UNKNOWN_SPEAKER,
) -> str:
    """Render segments as SRT text. Each cue reads ``[LABEL]: text``.

    Segments with empty text are skipped; cue numbers stay contiguous.
    """
    blocks: list[str] = []
    index = 1
    for seg in segments:
        text = _clean_text(seg.get("text"))
        if not text:
            continue
        start = float(seg["start"])
        end = max(float(seg["end"]), start)
        label = speaker_label(seg, rename, unknown)
        blocks.append(
            f"{index}\n{format_timestamp(start)} --> {format_timestamp(end)}\n[{label}]: {text}"
        )
        index += 1
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def write_srt(
    path: str | Path,
    segments: Iterable[Mapping],
    rename: Mapping[str, str] | None = None,
) -> Path:
    """Write ``segments`` to ``path`` as UTF-8 SRT and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(segments_to_srt(segments, rename), encoding="utf-8")
    return path
