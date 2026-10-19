"""Pure helpers for time ranges and the ffmpeg ``volume`` mute filter.

Everything in here is side-effect free so it can be unit-tested without
ffmpeg or WhisperX.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

Range = tuple[float, float]

# Keep each ``volume`` filter's enable expression to a bounded number of terms.
# ffmpeg's expression parser is recursive with a fixed depth budget: a
# ``between(...)+between(...)+...`` chain of more than 100 terms fails with
# "Error when evaluating the expression ... Cannot allocate memory" (measured on
# ffmpeg 9.0: 100 terms work, 101 fail). Chaining several ``volume`` filters,
# each with a short expression, has no such limit.
DEFAULT_MAX_TERMS_PER_FILTER = 50


def pad_ranges(ranges: Iterable[Range], pad: float, duration: float | None = None) -> list[Range]:
    """Widen every range by ``pad`` seconds on each side.

    Results are clamped to ``[0, duration]`` (``duration`` may be ``None`` for
    "no upper bound"). Ranges that collapse to zero or negative length are
    dropped. The output is *not* merged; see :func:`merge_ranges`.
    """
    if pad < 0:
        raise ValueError("pad must be >= 0")
    out: list[Range] = []
    for start, end in ranges:
        a = max(0.0, float(start) - pad)
        b = float(end) + pad
        if duration is not None:
            b = min(b, float(duration))
        if b > a:
            out.append((a, b))
    return out


def merge_ranges(ranges: Iterable[Range]) -> list[Range]:
    """Sort ranges by start and merge any that overlap or touch.

    Zero-length or inverted ranges are discarded.
    """
    merged: list[Range] = []
    for start, end in sorted((float(s), float(e)) for s, e in ranges):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def speaker_ranges(
    segments: Iterable[Mapping],
    speakers: str | Iterable[str],
    pad: float = 0.0,
    duration: float | None = None,
) -> list[Range]:
    """Collect, pad and merge the time ranges of every segment spoken by ``speakers``.

    ``speakers`` is one label or any iterable of labels; ranges of several
    speakers are merged together. ``segments`` are WhisperX-style dicts with
    ``start``, ``end`` and an optional ``speaker`` key. Segments without a
    ``speaker`` never match.
    """
    wanted = {speakers} if isinstance(speakers, str) else set(speakers)
    raw = [
        (float(seg["start"]), float(seg["end"])) for seg in segments if seg.get("speaker") in wanted
    ]
    return merge_ranges(pad_ranges(raw, pad, duration))


def total_duration(ranges: Iterable[Range]) -> float:
    """Sum of the lengths of all ranges (assumes they are already merged)."""
    return sum(max(0.0, end - start) for start, end in ranges)


def build_volume_filter(ranges: Sequence[Range], volume: float = 0.0) -> str:
    """Build a single ffmpeg ``volume`` filter that is enabled only inside ``ranges``.

    Example output::

        volume=enable='between(t,1.000,2.500)+between(t,4.000,5.250)':volume=0
    """
    if not ranges:
        raise ValueError("at least one range is required to build a volume filter")
    terms = "+".join(f"between(t,{start:.3f},{end:.3f})" for start, end in ranges)
    return f"volume=enable='{terms}':volume={volume:g}"


def build_mute_filtergraph(
    ranges: Sequence[Range],
    volume: float = 0.0,
    max_terms_per_filter: int = DEFAULT_MAX_TERMS_PER_FILTER,
) -> str:
    """Build the full audio filtergraph that mutes ``ranges``.

    For a small number of ranges this is exactly one ``volume`` filter. For very
    long transcripts the ranges are split across several chained ``volume``
    filters (separated by commas) so no single expression becomes unwieldy.
    """
    if max_terms_per_filter < 1:
        raise ValueError("max_terms_per_filter must be >= 1")
    if not ranges:
        raise ValueError("at least one range is required to build a mute filtergraph")
    chunks = [
        ranges[i : i + max_terms_per_filter] for i in range(0, len(ranges), max_terms_per_filter)
    ]
    return ",".join(build_volume_filter(chunk, volume=volume) for chunk in chunks)
