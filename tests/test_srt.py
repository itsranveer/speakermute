import pytest

from speakermute.srt import (
    format_timestamp,
    parse_rename,
    segments_to_srt,
    speaker_label,
    write_srt,
)

SEGMENTS = [
    {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00", "text": "  Hi, thanks for coming. "},
    {"start": 1.75, "end": 3.2, "speaker": "SPEAKER_01", "text": "Thanks for\nhaving me."},
    {"start": 3.5, "end": 4.0, "text": "unlabelled"},
]


class TestFormatTimestamp:
    @pytest.mark.parametrize(
        "seconds, expected",
        [
            (0, "00:00:00,000"),
            (1.5, "00:00:01,500"),
            (61.001, "00:01:01,001"),
            (3661.9999, "01:01:02,000"),
            (-2, "00:00:00,000"),
        ],
    )
    def test_formats(self, seconds, expected):
        assert format_timestamp(seconds) == expected


class TestParseRename:
    def test_parses_pairs(self):
        assert parse_rename(["SPEAKER_00=Host", " SPEAKER_01 = Guest "]) == {
            "SPEAKER_00": "Host",
            "SPEAKER_01": "Guest",
        }

    def test_none_and_empty(self):
        assert parse_rename(None) == {}
        assert parse_rename([]) == {}

    @pytest.mark.parametrize("bad", ["SPEAKER_00", "=Guest", "SPEAKER_00=", "="])
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError):
            parse_rename([bad])


class TestSpeakerLabel:
    def test_raw_label(self):
        assert speaker_label({"speaker": "SPEAKER_00"}) == "SPEAKER_00"

    def test_renamed(self):
        assert speaker_label({"speaker": "SPEAKER_00"}, {"SPEAKER_00": "Host"}) == "Host"

    def test_missing_speaker_is_unknown(self):
        assert speaker_label({}) == "UNKNOWN"
        assert speaker_label({"speaker": None}) == "UNKNOWN"


class TestSegmentsToSrt:
    def test_basic_output(self):
        srt = segments_to_srt(SEGMENTS)
        assert srt == (
            "1\n00:00:00,000 --> 00:00:01,500\n[SPEAKER_00]: Hi, thanks for coming.\n"
            "\n"
            "2\n00:00:01,750 --> 00:00:03,200\n[SPEAKER_01]: Thanks for having me.\n"
            "\n"
            "3\n00:00:03,500 --> 00:00:04,000\n[UNKNOWN]: unlabelled\n"
        )

    def test_renaming(self):
        srt = segments_to_srt(SEGMENTS, rename={"SPEAKER_00": "Host", "SPEAKER_01": "Guest"})
        assert "[Host]: Hi, thanks for coming." in srt
        assert "[Guest]: Thanks for having me." in srt
        assert "[UNKNOWN]: unlabelled" in srt
        assert "SPEAKER_0" not in srt

    def test_skips_empty_text_and_keeps_indices_contiguous(self):
        segs = [
            {"start": 0, "end": 1, "speaker": "A", "text": "   "},
            {"start": 1, "end": 2, "speaker": "A", "text": "one"},
            {"start": 2, "end": 3, "speaker": "A"},
            {"start": 3, "end": 4, "speaker": "A", "text": "two"},
        ]
        srt = segments_to_srt(segs)
        assert srt.startswith("1\n")
        assert "\n\n2\n" in srt
        assert "\n3\n" not in srt

    def test_empty_input(self):
        assert segments_to_srt([]) == ""

    def test_end_never_before_start(self):
        srt = segments_to_srt([{"start": 5.0, "end": 4.0, "speaker": "A", "text": "x"}])
        assert "00:00:05,000 --> 00:00:05,000" in srt


def test_write_srt_creates_parent_dirs(tmp_path):
    target = tmp_path / "out" / "video" / "subtitles.srt"
    result = write_srt(target, SEGMENTS, rename={"SPEAKER_01": "Guest"})
    assert result == target
    content = target.read_text(encoding="utf-8")
    assert "[Guest]: Thanks for having me." in content
    assert content.endswith("\n")
