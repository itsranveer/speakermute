import pytest

from speakermute.ranges import (
    build_mute_filtergraph,
    build_volume_filter,
    merge_ranges,
    pad_ranges,
    speaker_ranges,
    total_duration,
)


class TestPadRanges:
    def test_pads_both_sides(self):
        assert pad_ranges([(1.0, 2.0)], 0.15) == [(0.85, 2.15)]

    def test_clamps_to_zero_and_duration(self):
        assert pad_ranges([(0.05, 9.95)], 0.2, duration=10.0) == [(0.0, 10.0)]

    def test_no_duration_means_no_upper_clamp(self):
        assert pad_ranges([(9.9, 10.0)], 0.5) == [(9.4, 10.5)]

    def test_zero_pad_is_identity(self):
        assert pad_ranges([(1.0, 2.0), (3.0, 4.0)], 0.0) == [(1.0, 2.0), (3.0, 4.0)]

    def test_drops_empty_ranges(self):
        assert pad_ranges([(5.0, 5.0)], 0.0) == []

    def test_negative_pad_rejected(self):
        with pytest.raises(ValueError):
            pad_ranges([(1.0, 2.0)], -0.1)


class TestMergeRanges:
    def test_empty(self):
        assert merge_ranges([]) == []

    def test_disjoint_sorted(self):
        assert merge_ranges([(3.0, 4.0), (1.0, 2.0)]) == [(1.0, 2.0), (3.0, 4.0)]

    def test_overlapping(self):
        assert merge_ranges([(1.0, 3.0), (2.0, 4.0)]) == [(1.0, 4.0)]

    def test_touching(self):
        assert merge_ranges([(1.0, 2.0), (2.0, 3.0)]) == [(1.0, 3.0)]

    def test_contained(self):
        assert merge_ranges([(1.0, 10.0), (2.0, 3.0)]) == [(1.0, 10.0)]

    def test_drops_inverted_and_zero_length(self):
        assert merge_ranges([(5.0, 4.0), (6.0, 6.0), (1.0, 2.0)]) == [(1.0, 2.0)]

    def test_accepts_ints(self):
        assert merge_ranges([(1, 2), (2, 3)]) == [(1.0, 3.0)]


class TestSpeakerRanges:
    SEGMENTS = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "Hi"},
        {"start": 2.5, "end": 4.0, "speaker": "SPEAKER_01", "text": "Hello"},
        {"start": 4.1, "end": 6.0, "speaker": "SPEAKER_01", "text": "I'm Sam"},
        {"start": 6.5, "end": 8.0, "speaker": "SPEAKER_00", "text": "Tell me"},
        {"start": 8.0, "end": 9.0, "text": "no speaker key"},
    ]

    def test_collects_only_target_speaker(self):
        assert speaker_ranges(self.SEGMENTS, "SPEAKER_00") == [(0.0, 2.0), (6.5, 8.0)]

    def test_pad_merges_adjacent_segments(self):
        # 2.5-4.0 and 4.1-6.0 padded by 0.15 overlap -> one range
        assert speaker_ranges(self.SEGMENTS, "SPEAKER_01", pad=0.15) == [(2.35, 6.15)]

    def test_pad_clamped_to_duration(self):
        assert speaker_ranges(self.SEGMENTS, "SPEAKER_00", pad=0.5, duration=8.2) == [
            (0.0, 2.5),
            (6.0, 8.2),
        ]

    def test_unknown_speaker_gives_empty(self):
        assert speaker_ranges(self.SEGMENTS, "SPEAKER_99") == []

    def test_several_speakers_are_merged_together(self):
        assert speaker_ranges(self.SEGMENTS, ["SPEAKER_00", "SPEAKER_01"]) == [
            (0.0, 2.0),
            (2.5, 4.0),
            (4.1, 6.0),
            (6.5, 8.0),
        ]
        # touching ranges across speakers merge once padded
        assert speaker_ranges(self.SEGMENTS, {"SPEAKER_01", "SPEAKER_00"}, pad=0.3) == [(0.0, 8.3)]

    def test_empty_speaker_list(self):
        assert speaker_ranges(self.SEGMENTS, []) == []


def test_total_duration():
    assert total_duration([(0.0, 1.5), (3.0, 4.0)]) == pytest.approx(2.5)


class TestVolumeFilter:
    def test_single_range(self):
        assert (
            build_volume_filter([(1.0, 2.5)]) == "volume=enable='between(t,1.000,2.500)':volume=0"
        )

    def test_multiple_ranges_joined_with_plus(self):
        expr = build_volume_filter([(0.0, 1.0), (4.0, 5.25), (10.123456, 11.0)])
        assert expr == (
            "volume=enable='between(t,0.000,1.000)+between(t,4.000,5.250)"
            "+between(t,10.123,11.000)':volume=0"
        )

    def test_custom_volume(self):
        assert build_volume_filter([(0.0, 1.0)], volume=0.5).endswith(":volume=0.5")

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            build_volume_filter([])


class TestMuteFiltergraph:
    def test_small_input_is_single_filter(self):
        ranges = [(0.0, 1.0), (2.0, 3.0)]
        assert build_mute_filtergraph(ranges) == build_volume_filter(ranges)

    def test_chunks_into_chained_filters(self):
        ranges = [(float(i), float(i) + 0.5) for i in range(5)]
        graph = build_mute_filtergraph(ranges, max_terms_per_filter=2)
        filters = graph.split(",volume=")
        assert len(filters) == 3
        assert graph.count("between(") == 5
        assert graph.startswith(
            "volume=enable='between(t,0.000,0.500)+between(t,1.000,1.500)':volume=0"
        )
        assert graph.endswith("volume=enable='between(t,4.000,4.500)':volume=0")

    def test_default_chunk_size_stays_under_ffmpeg_parser_limit(self):
        # 148 ranges broke ffmpeg 9's expression parser in a single filter
        ranges = [(float(i), float(i) + 0.5) for i in range(148)]
        graph = build_mute_filtergraph(ranges)
        filters = graph.split(",volume=")
        assert len(filters) == 3
        assert all(f.count("between(") <= 50 for f in filters)
        assert graph.count("between(") == 148

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            build_mute_filtergraph([])

    def test_bad_chunk_size_rejected(self):
        with pytest.raises(ValueError):
            build_mute_filtergraph([(0.0, 1.0)], max_terms_per_filter=0)
