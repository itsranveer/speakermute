import pytest

from speakermute.paths import collect_videos, muted_suffix, outputs_for


def test_outputs_layout(tmp_path):
    out = outputs_for("videos/My Podcast.mp4", tmp_path / "out")
    assert out.dir == tmp_path / "out" / "My Podcast"
    assert out.transcript == out.dir / "transcript.json"
    assert out.subtitles == out.dir / "subtitles.srt"
    assert out.muted == out.dir / "My Podcast_muted.mp4"
    assert out.filter_script == out.dir / "mute_filter.txt"


@pytest.mark.parametrize(
    ("video", "suffix"),
    [
        ("a.mp4", ".mp4"),
        ("a.MP4", ".mp4"),
        ("a.mov", ".mov"),
        ("a.m4v", ".m4v"),
        ("a.mkv", ".mkv"),
        ("a.webm", ".mkv"),  # WebM cannot hold AAC, Matroska takes anything
        ("a.avi", ".mkv"),
    ],
)
def test_muted_suffix_follows_input_container(video, suffix):
    assert muted_suffix(video) == suffix


def test_muted_output_keeps_matroska(tmp_path):
    out = outputs_for("clips/talk.webm", tmp_path / "out")
    assert out.muted == tmp_path / "out" / "talk" / "talk_muted.mkv"


def test_collect_videos_single_file(tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"")
    assert collect_videos(f) == [f]


def test_collect_videos_folder_filters_and_sorts(tmp_path):
    for name in ["b.MP4", "a.mp4", "notes.txt", "c.webm", "d.mkv"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub").mkdir()
    assert [p.name for p in collect_videos(tmp_path)] == ["a.mp4", "b.MP4", "c.webm", "d.mkv"]


def test_collect_videos_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_videos(tmp_path)


def test_collect_videos_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_videos(tmp_path / "nope.mp4")
