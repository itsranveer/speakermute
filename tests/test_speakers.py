from rich.console import Console

from speakermute.speakers import (
    SpeakerSummary,
    format_clock,
    pick_preview_lines,
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


def test_preview_skips_one_word_acknowledgements_when_longer_lines_exist():
    texts = [
        "Okay.",
        "Okay.",
        "All right.",
        "Let me repeat the question back.",
        "Sure.",
        "So I need a class with three operations.",
        "Insert, delete and get random.",
    ]
    assert pick_preview_lines(texts, lines=3, min_words=4) == [
        "Let me repeat the question back.",
        "So I need a class with three operations.",
        "Insert, delete and get random.",
    ]


def test_preview_falls_back_to_short_lines_in_order():
    # one long line, the rest short: keep the long one and fill from the start
    texts = ["Okay.", "Right.", "Let me think about that one.", "Hmm."]
    assert pick_preview_lines(texts, lines=3, min_words=4) == [
        "Okay.",
        "Right.",
        "Let me think about that one.",
    ]
    assert pick_preview_lines(["Yes.", "No."], lines=3, min_words=4) == ["Yes.", "No."]
    assert pick_preview_lines([], lines=3, min_words=4) == []


def test_summarize_prefers_substantive_lines():
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "S", "text": "Okay."},
        {"start": 1.0, "end": 2.0, "speaker": "S", "text": "Right."},
        {
            "start": 2.0,
            "end": 5.0,
            "speaker": "S",
            "text": "From what I understand, I need a class.",
        },
        {"start": 5.0, "end": 6.0, "speaker": "S", "text": "One is to insert an element."},
    ]
    (summary,) = summarize_speakers(segments, lines=2)
    assert summary.first_lines == [
        "From what I understand, I need a class.",
        "One is to insert an element.",
    ]
    assert summary.segment_count == 4


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
