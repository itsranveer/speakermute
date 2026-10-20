import json
import shutil
from pathlib import Path

import pytest

from speakermute import pipeline
from speakermute.errors import SpeakermuteError, SpeakerNotFoundError
from speakermute.paths import outputs_for
from speakermute.transcribe import TranscribeOptions, Transcriber


@pytest.fixture
def fake_ffmpeg(monkeypatch, wav_file):
    """Replace every ffmpeg call the pipeline makes with recorders."""
    calls: dict = {}

    def extract_audio(video, wav_path, *, sample_rate=16000, total_duration=None):
        Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(wav_file, wav_path)
        calls["extract_audio"] = {"video": Path(video), "total_duration": total_duration}
        return Path(wav_path)

    def mute_with_filter_script(
        video,
        output,
        filter_script,
        *,
        subtitles=None,
        crf=20,
        preset="medium",
        audio_bitrate="192k",
        total_duration=None,
    ):
        calls["mute"] = {
            "video": Path(video),
            "output": Path(output),
            "script": Path(filter_script).read_text(encoding="utf-8"),
            "subtitles": Path(subtitles) if subtitles else None,
            "crf": crf,
            "preset": preset,
            "total_duration": total_duration,
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"muted")
        return Path(output)

    monkeypatch.setattr(
        pipeline.ffmpeg, "probe_duration", lambda p: calls.setdefault("duration", 120.0)
    )
    monkeypatch.setattr(pipeline.ffmpeg, "extract_audio", extract_audio)
    monkeypatch.setattr(pipeline.ffmpeg, "mute_with_filter_script", mute_with_filter_script)
    monkeypatch.setattr(
        pipeline.ffmpeg,
        "check_same_duration",
        lambda a, b: (
            120.0,
            calls.get("out_duration", 120.0),
            calls.get("out_duration", 120.0) == 120.0,
        ),
    )
    monkeypatch.setattr(
        pipeline.ffmpeg,
        "require_subtitles_filter",
        lambda: calls.setdefault("subtitles_checked", True),
    )
    return calls


def test_transcribe_video_writes_json_and_srt(tmp_path, fake_whisperx, fake_ffmpeg, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok")
    video = tmp_path / "Episode One.mp4"
    video.write_bytes(b"")
    outputs = outputs_for(video, tmp_path / "out")
    transcriber = Transcriber(TranscribeOptions())

    transcript = pipeline.transcribe_video(
        video, outputs, transcriber, rename={"SPEAKER_01": "Guest"}
    )

    assert fake_ffmpeg["extract_audio"]["video"] == video
    assert fake_ffmpeg["extract_audio"]["total_duration"] == 120.0
    assert fake_whisperx["transcribe"]["samples"] == 8000

    saved = json.loads(outputs.transcript.read_text(encoding="utf-8"))
    assert saved["video"] == "Episode One.mp4"
    assert saved["duration"] == 120.0
    assert saved["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert saved["segments"][0]["speaker"] == "SPEAKER_00"
    assert saved["segments"][0]["text"] == "Hello there."
    assert saved["segments"][0]["words"][0]["word"] == "Hello"
    assert saved["settings"]["model"] == "medium"
    assert transcript["segments"] == saved["segments"]

    srt = outputs.subtitles.read_text(encoding="utf-8")
    assert "[SPEAKER_00]: Hello there." in srt
    assert "[Guest]: Hi, thanks." in srt


@pytest.fixture
def transcript_dir(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")
    outputs = outputs_for(video, tmp_path / "out")
    outputs.dir.mkdir(parents=True)
    outputs.transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "First topic?"},
                    {"start": 2.5, "end": 5.0, "speaker": "SPEAKER_01", "text": "Reply one."},
                    {"start": 5.1, "end": 7.0, "speaker": "SPEAKER_01", "text": "More reply."},
                    {"start": 8.0, "end": 9.0, "speaker": "SPEAKER_00", "text": "Second topic?"},
                    {"start": 9.5, "end": 12.0, "speaker": "SPEAKER_01", "text": "Reply two."},
                ]
            }
        ),
        encoding="utf-8",
    )
    return video, outputs


class TestMuteVideo:
    def test_mutes_speaker_and_refreshes_srt(self, transcript_dir, fake_ffmpeg):
        video, outputs = transcript_dir
        result = pipeline.mute_video(
            video, outputs, ["SPEAKER_01"], pad=0.15, rename={"SPEAKER_01": "Guest"}
        )

        # 2.5-5.0 and 5.1-7.0 merge once padded; 9.5-12.0 stays separate
        assert result.ranges == [(2.35, 7.15), (9.35, 12.15)]
        assert result.speakers == ["SPEAKER_01"]
        assert result.muted_seconds == pytest.approx(7.6)
        assert result.output == outputs.muted and outputs.muted.exists()
        assert result.burned_subtitles is False
        assert result.duration_ok is True

        mute = fake_ffmpeg["mute"]
        assert mute["video"] == video
        assert mute["output"] == outputs.muted
        assert mute["subtitles"] is None
        assert mute["total_duration"] == 120.0
        assert mute["script"] == (
            "volume=enable='between(t,2.350,7.150)+between(t,9.350,12.150)':volume=0"
        )
        assert outputs.filter_script.exists()
        srt = outputs.subtitles.read_text(encoding="utf-8")
        assert "[Guest]: Reply one." in srt and "[SPEAKER_00]: First topic?" in srt
        assert "subtitles_checked" not in fake_ffmpeg

    def test_burn_subs_is_a_single_pass(self, transcript_dir, fake_ffmpeg):
        video, outputs = transcript_dir
        result = pipeline.mute_video(
            video, outputs, ["SPEAKER_01"], burn_subs=True, crf=18, preset="fast"
        )
        assert fake_ffmpeg["subtitles_checked"] is True
        mute = fake_ffmpeg["mute"]
        assert mute["video"] == video  # straight from the original, no intermediate
        assert mute["subtitles"] == outputs.subtitles
        assert mute["output"] == outputs.muted
        assert (mute["crf"], mute["preset"]) == (18, "fast")
        assert result.output == outputs.muted
        assert result.burned_subtitles is True

    def test_unknown_speaker(self, transcript_dir, fake_ffmpeg):
        video, outputs = transcript_dir
        with pytest.raises(SpeakerNotFoundError) as exc:
            pipeline.mute_video(video, outputs, ["SPEAKER_01", "SPEAKER_07"])
        assert "SPEAKER_07" in str(exc.value)
        assert "SPEAKER_00, SPEAKER_01" in str(exc.value)
        assert "mute" not in fake_ffmpeg

    def test_empty_speaker_list(self, transcript_dir, fake_ffmpeg):
        video, outputs = transcript_dir
        with pytest.raises(SpeakerNotFoundError):
            pipeline.mute_video(video, outputs, [])

    def test_mutes_several_speakers(self, transcript_dir, fake_ffmpeg, caplog):
        video, outputs = transcript_dir
        with caplog.at_level("WARNING", logger="speakermute"):
            result = pipeline.mute_video(
                video, outputs, ["SPEAKER_01", "SPEAKER_00", "SPEAKER_01"], pad=0.0
            )
        assert result.speakers == ["SPEAKER_01", "SPEAKER_00"]  # de-duplicated
        assert result.ranges == [(0.0, 2.0), (2.5, 5.0), (5.1, 7.0), (8.0, 9.0), (9.5, 12.0)]
        assert fake_ffmpeg["mute"]["script"].count("between(") == 5
        assert "muting every speaker" in caplog.text  # both labels => silent output

    def test_custom_output_path(self, transcript_dir, fake_ffmpeg, tmp_path):
        video, outputs = transcript_dir
        custom = tmp_path / "elsewhere" / "host-only.mp4"
        result = pipeline.mute_video(video, outputs, ["SPEAKER_01"], output=custom)
        assert result.output == custom and custom.exists()
        assert fake_ffmpeg["mute"]["output"] == custom
        # the side files stay in the per-video output folder
        assert outputs.subtitles.exists() and outputs.filter_script.exists()
        assert not outputs.muted.exists()

    def test_refuses_to_overwrite_the_input(self, transcript_dir, fake_ffmpeg):
        video, outputs = transcript_dir
        before = video.read_bytes()
        with pytest.raises(SpeakermuteError, match="input video itself"):
            pipeline.mute_video(video, outputs, ["SPEAKER_01"], output=video)
        assert "mute" not in fake_ffmpeg  # ffmpeg never started
        assert video.read_bytes() == before

    def test_uses_explicit_transcript(self, transcript_dir, fake_ffmpeg, tmp_path):
        video, outputs = transcript_dir
        edited = tmp_path / "edited.json"
        edited.write_text(
            json.dumps({"segments": [{"start": 1.0, "end": 2.0, "speaker": "GUEST", "text": "x"}]}),
            encoding="utf-8",
        )
        result = pipeline.mute_video(video, outputs, ["GUEST"], pad=0.0, transcript_path=edited)
        assert result.ranges == [(1.0, 2.0)]

    def test_duration_mismatch_is_reported(self, transcript_dir, fake_ffmpeg, caplog):
        video, outputs = transcript_dir
        fake_ffmpeg["out_duration"] = 118.0
        with caplog.at_level("WARNING", logger="speakermute"):
            result = pipeline.mute_video(video, outputs, ["SPEAKER_01"])
        assert result.duration_ok is False
        assert "duration changed" in caplog.text

    def test_pad_clamped_to_video_duration(self, transcript_dir, fake_ffmpeg):
        video, outputs = transcript_dir
        fake_ffmpeg["duration"] = 12.05
        result = pipeline.mute_video(video, outputs, ["SPEAKER_01"], pad=0.5)
        assert result.ranges[-1] == (9.0, 12.05)
