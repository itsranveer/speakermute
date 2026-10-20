"""Per-speaker summaries used by the ``speakers`` command."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from rich.table import Table

from speakermute.srt import UNKNOWN_SPEAKER

# A line needs this many words to count as a "first line" preview. Shorter ones
# ("Okay.", "Right.") are only used when a speaker has nothing longer to show.
MIN_PREVIEW_WORDS = 4


@dataclass
class SpeakerSummary:
    label: str
    talk_time: float = 0.0
    segment_count: int = 0
    first_start: float = 0.0
    first_lines: list[str] = field(default_factory=list)


def pick_preview_lines(texts: Sequence[str], lines: int, min_words: int) -> list[str]:
    """Choose up to ``lines`` representative lines, in chronological order.

    Lines with at least ``min_words`` words are preferred so that a speaker whose
    first utterances are "Okay." and "Right." is still recognisable; shorter
    lines fill the remaining slots only when there are not enough long ones.
    """
    long_idx = [i for i, t in enumerate(texts) if len(t.split()) >= min_words]
    chosen = long_idx[:lines]
    if len(chosen) < lines:
        taken = set(chosen)
        chosen += [i for i in range(len(texts)) if i not in taken][: lines - len(chosen)]
    return [texts[i] for i in sorted(chosen)]


def summarize_speakers(
    segments: Iterable[Mapping],
    lines: int = 3,
    unknown: str = UNKNOWN_SPEAKER,
    min_words: int = MIN_PREVIEW_WORDS,
) -> list[SpeakerSummary]:
    """Aggregate talk time, segment count and a few representative lines per speaker.

    Speakers are returned in order of first appearance. Segments without a
    ``speaker`` key are grouped under ``unknown``. See :func:`pick_preview_lines`
    for how the lines are chosen.
    """
    summaries: dict[str, SpeakerSummary] = {}
    texts: dict[str, list[str]] = {}
    ordered = sorted(segments, key=lambda s: float(s["start"]))
    for seg in ordered:
        label = seg.get("speaker") or unknown
        start, end = float(seg["start"]), float(seg["end"])
        summary = summaries.get(label)
        if summary is None:
            summary = SpeakerSummary(label=label, first_start=start)
            summaries[label] = summary
            texts[label] = []
        summary.talk_time += max(0.0, end - start)
        summary.segment_count += 1
        text = " ".join(str(seg.get("text") or "").split())
        if text:
            texts[label].append(text)
    for label, summary in summaries.items():
        summary.first_lines = pick_preview_lines(texts[label], lines, min_words)
    return list(summaries.values())


def format_clock(seconds: float) -> str:
    """Format seconds as ``M:SS`` or ``H:MM:SS``."""
    total = int(round(max(0.0, seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def render_speaker_table(summaries: Iterable[SpeakerSummary], title: str | None = None) -> Table:
    """Build a rich table for the speaker summaries."""
    table = Table(title=title, show_lines=True)
    table.add_column("Speaker", style="bold cyan", no_wrap=True)
    table.add_column("Talk time", justify="right")
    table.add_column("Segments", justify="right")
    table.add_column("First speaks at", justify="right")
    table.add_column("First lines")
    for s in summaries:
        lines = "\n".join(f"• {line}" for line in s.first_lines) or "[dim](no text)[/dim]"
        table.add_row(
            s.label,
            format_clock(s.talk_time),
            str(s.segment_count),
            format_clock(s.first_start),
            lines,
        )
    return table
