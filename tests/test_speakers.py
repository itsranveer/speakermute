from rich.console import Console

from speakermute.speakers import (
    SpeakerSummary,
    format_clock,
    render_speaker_table,
    summarize_speakers,
)

SEGMENTS = [
    {"start": 10.0, "end": 12.0, "speaker": "SPEAKER_01", "text": "Thanks for having me."},
    {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "Welcome to the show."},
    {"start": 3.5, "end": 9.0, "speaker": "SPEAKER_00", "text": "Tell us what you work on."},
    {"start": 12.5, "end": 20.0, "speaker": "SPEAKER_01", "text": "I'm a backend engineer."},
    {"start": 20.0, "end": 25.0, "speaker": "SPEAKER_01", "text": "I work mostly in Go."},
    {"start": 25.0, "end": 27.0, "speaker": "SPEAKER_01", "text": "And some Python."},
    {"start": 27.5, "end": 28.0, "text": "(crosstalk)"},
]


def test_summarize_orders_by_first_appearance_and_aggregates():
    summaries = summarize_speakers(SEGMENTS)
    assert [s.label for s in summaries] == ["SPEAKER_00", "SPEAKER_01", "UNKNOWN"]

    s00, s01, unknown = summaries
    assert s00.talk_time == 8.5
    assert s00.segment_count == 2
    assert s00.first_start == 0.0
    assert s00.first_lines == ["Welcome to the show.", "Tell us what you work on."]

    assert s01.talk_time == 16.5
    assert s01.segment_count == 4
    assert s01.first_start == 10.0
    # only the first 3 lines, in chronological order
    assert s01.first_lines == [
        "Thanks for having me.",
        "I'm a backend engineer.",
        "I work mostly in Go.",
    ]

    assert unknown.talk_time == 0.5
    assert unknown.first_lines == ["(crosstalk)"]


def test_summarize_respects_lines_argument():
    summaries = summarize_speakers(SEGMENTS, lines=1)
    assert summaries[1].first_lines == ["Thanks for having me."]


def test_summarize_empty():
    assert summarize_speakers([]) == []


def test_format_clock():
    assert format_clock(0) == "0:00"
    assert format_clock(65) == "1:05"
    assert format_clock(3725.4) == "1:02:05"


def test_render_speaker_table_contains_labels_and_lines():
    summaries = [
        SpeakerSummary("SPEAKER_00", 8.5, 2, 0.0, ["Welcome."]),
        SpeakerSummary("SPEAKER_01", 125.0, 4, 10.0, []),
    ]
    console = Console(record=True, width=120)
    console.print(render_speaker_table(summaries))
    text = console.export_text()
    assert "SPEAKER_00" in text and "SPEAKER_01" in text
    assert "Welcome." in text
    assert "2:05" in text
    assert "(no text)" in text
