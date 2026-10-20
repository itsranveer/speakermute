import json
import re

import pytest
from typer.testing import CliRunner

from speakermute import cli
from speakermute.ffmpeg import FFmpegError
from speakermute.pipeline import MuteResult

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_BOXES = re.compile(r"[\u2500-\u257f]")  # rich panel borders


def plain(result) -> str:
    """CLI output as one line of plain text.

    typer renders errors through rich, and on GitHub Actions (``GITHUB_ACTIONS`` set)
    it forces colour and an 80-column panel, so option names get ANSI codes inside
    them and messages wrap. Strip both so assertions see the same text everywhere.
    """
    text = _BOXES.sub(" ", _ANSI.sub("", result.output))
    return " ".join(text.split())


@pytest.fixture(autouse=True)
def _ffmpeg_present(monkeypatch):
    monkeypatch.setattr(cli.ffmpeg, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(cli.ffmpeg, "require_subtitles_filter", lambda: None)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "episode.mp4"
    path.write_bytes(b"")
    return path


@pytest.fixture
def transcript(video, tmp_path):
    outputs = cli.outputs_for(video, tmp_path / "out")
    outputs.dir.mkdir(parents=True)
    outputs.transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "Welcome."},
                    {
                        "start": 2.5,
                        "end": 9.0,
                        "speaker": "SPEAKER_01",
                        "text": "Thanks, happy to be here.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return outputs


@pytest.fixture
def fake_transcribe(monkeypatch):
    """Record Transcriber construction and transcribe_video calls without WhisperX."""
    calls = {"videos": [], "options": None, "released": 0}

    class FakeTranscriber:
        def __init__(self, options):
            calls["options"] = options

        def describe(self):
            return "fake"

        def release(self):
            calls["released"] += 1

    def fake_transcribe_video(video, outputs, transcriber, *, rename=None):
        calls["videos"].append(video)
        calls["rename"] = rename
        outputs.dir.mkdir(parents=True, exist_ok=True)
        outputs.transcript.write_text(
            json.dumps(
                {"segments": [{"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "hi"}]}
            )
        )
        outputs.subtitles.write_text("1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: hi\n")
        return {
            "segments": [{"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "hi"}],
            "speakers": ["SPEAKER_00"],
        }

    monkeypatch.setattr(cli, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(cli, "transcribe_video", fake_transcribe_video)
    return calls


@pytest.fixture
def fake_mute(monkeypatch):
    calls = []

    def fake_mute_video(
        video,
        outputs,
        speaker,
        *,
        pad,
        burn_subs,
        rename=None,
        transcript_path=None,
        crf=20,
        preset="medium",
        output=None,
    ):
        calls.append(
            {
                "video": video,
                "speaker": speaker,
                "pad": pad,
                "burn_subs": burn_subs,
                "rename": rename,
                "transcript_path": transcript_path,
                "output": output,
            }
        )
        return MuteResult(
            speakers=list(speaker),
            output=output or outputs.muted,
            burned_subtitles=burn_subs,
            ranges=[(2.35, 9.15)],
            muted_seconds=6.8,
            source_duration=100.0,
            output_duration=100.0,
            duration_ok=True,
        )

    monkeypatch.setattr(cli, "mute_video", fake_mute_video)
    return calls


def test_help_lists_commands():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for name in ("transcribe", "mute", "run", "speakers"):
        assert name in plain(result)


def test_version():
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0 and "speakermute" in plain(result)


def test_missing_ffmpeg_is_a_clean_error(monkeypatch, video):
    def boom():
        raise FFmpegError("could not find ffmpeg")

    monkeypatch.setattr(cli.ffmpeg, "require_ffmpeg", boom)
    result = runner.invoke(cli.app, ["mute", str(video), "--speaker", "SPEAKER_01"])
    assert result.exit_code == 1
    assert "could not find ffmpeg" in plain(result)
    assert "Traceback" not in plain(result)


class TestTranscribe:
    def test_transcribes_with_options(self, video, tmp_path, fake_transcribe):
        result = runner.invoke(
            cli.app,
            [
                "transcribe",
                str(video),
                "--out-dir",
                str(tmp_path / "out"),
                "--model",
                "small",
                "--device",
                "cpu",
                "--compute-type",
                "float32",
                "--min-speakers",
                "1",
                "--max-speakers",
                "3",
                "--language",
                "auto",
                "--rename",
                "SPEAKER_01=Guest",
                "--no-fill-nearest",
            ],
        )
        assert result.exit_code == 0, result.output
        assert fake_transcribe["videos"] == [video]
        opts = fake_transcribe["options"]
        assert (opts.model, opts.device, opts.compute_type) == ("small", "cpu", "float32")
        assert (opts.min_speakers, opts.max_speakers) == (1, 3)
        assert opts.language is None
        assert opts.fill_nearest is False
        assert fake_transcribe["rename"] == {"SPEAKER_01": "Guest"}
        assert fake_transcribe["released"] == 1

    def test_speaker_bounds_default_to_auto(self, video, tmp_path, fake_transcribe):
        result = runner.invoke(cli.app, ["transcribe", str(video), "--out-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        opts = fake_transcribe["options"]
        assert (opts.min_speakers, opts.max_speakers) == (None, None)

    def test_min_above_max_rejected(self, video, fake_transcribe):
        result = runner.invoke(
            cli.app, ["transcribe", str(video), "--min-speakers", "3", "--max-speakers", "2"]
        )
        assert result.exit_code == 2
        assert "--min-speakers" in plain(result)
        assert fake_transcribe["videos"] == []

    def test_skips_existing_unless_force(self, video, tmp_path, transcript, fake_transcribe):
        out = str(tmp_path / "out")
        result = runner.invoke(cli.app, ["transcribe", str(video), "--out-dir", out])
        assert result.exit_code == 0, result.output
        assert fake_transcribe["videos"] == []
        assert "Skipping" in plain(result)
        assert fake_transcribe["options"] is None  # no Transcriber built => no HF token needed

        result = runner.invoke(cli.app, ["transcribe", str(video), "--out-dir", out, "--force"])
        assert result.exit_code == 0, result.output
        assert fake_transcribe["videos"] == [video]

    def test_folder_input(self, tmp_path, fake_transcribe):
        folder = tmp_path / "videos"
        folder.mkdir()
        for name in ("b.mp4", "a.mkv", "notes.txt"):
            (folder / name).write_bytes(b"")
        result = runner.invoke(
            cli.app, ["transcribe", str(folder), "--out-dir", str(tmp_path / "out")]
        )
        assert result.exit_code == 0, result.output
        assert [v.name for v in fake_transcribe["videos"]] == ["a.mkv", "b.mp4"]

    def test_bad_rename(self, video, fake_transcribe):
        result = runner.invoke(cli.app, ["transcribe", str(video), "--rename", "nonsense"])
        assert result.exit_code != 0
        assert "OLD=NEW" in plain(result)

    def test_min_greater_than_max(self, video, fake_transcribe):
        result = runner.invoke(
            cli.app, ["transcribe", str(video), "--min-speakers", "3", "--max-speakers", "2"]
        )
        assert result.exit_code != 0
        assert "--min-speakers" in plain(result)

    def test_missing_path(self, tmp_path):
        result = runner.invoke(cli.app, ["transcribe", str(tmp_path / "nope.mp4")])
        assert result.exit_code != 0

    def test_user_error_is_clean(self, video, monkeypatch, tmp_path):
        from speakermute.errors import MissingTokenError

        def boom(options):
            raise MissingTokenError("no token")

        monkeypatch.setattr(cli, "Transcriber", boom)
        result = runner.invoke(
            cli.app, ["transcribe", str(video), "--out-dir", str(tmp_path / "out")]
        )
        assert result.exit_code == 1
        assert "no token" in plain(result) and "Traceback" not in plain(result)


class TestMute:
    def test_mute_passes_arguments(self, video, tmp_path, fake_mute):
        out = tmp_path / "out"
        result = runner.invoke(
            cli.app,
            [
                "mute",
                str(video),
                "--speaker",
                "SPEAKER_01",
                "--pad",
                "0.3",
                "--burn-subs",
                "--rename",
                "SPEAKER_01=Guest",
                "--out-dir",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert fake_mute == [
            {
                "video": video,
                "speaker": ["SPEAKER_01"],
                "pad": 0.3,
                "burn_subs": True,
                "rename": {"SPEAKER_01": "Guest"},
                "transcript_path": None,
                "output": None,
            }
        ]
        assert "episode_muted.mp4" in plain(result)
        assert "burned-in" in plain(result)  # rich may wrap the line

    def test_mute_output_option(self, video, tmp_path, fake_mute):
        custom = tmp_path / "custom.mp4"
        result = runner.invoke(cli.app, ["mute", str(video), "--speaker", "S", "-o", str(custom)])
        assert result.exit_code == 0, result.output
        assert fake_mute[0]["output"] == custom
        assert "custom.mp4" in plain(result).replace(" ", "")

    def test_mute_with_explicit_transcript(self, video, tmp_path, fake_mute):
        edited = tmp_path / "edited.json"
        edited.write_text("{}")
        result = runner.invoke(
            cli.app, ["mute", str(video), "--speaker", "S", "--transcript", str(edited)]
        )
        assert result.exit_code == 0, result.output
        assert fake_mute[0]["transcript_path"] == edited

    def test_mute_several_speakers(self, video, fake_mute):
        result = runner.invoke(
            cli.app,
            ["mute", str(video), "--speaker", "SPEAKER_01", "--speaker", "SPEAKER_02"],
        )
        assert result.exit_code == 0, result.output
        assert fake_mute[0]["speaker"] == ["SPEAKER_01", "SPEAKER_02"]
        assert "SPEAKER_01, SPEAKER_02" in plain(result)

    def test_mute_requires_speaker(self, video):
        result = runner.invoke(cli.app, ["mute", str(video)])
        assert result.exit_code != 0
        assert "--speaker" in plain(result)

    def test_mute_unknown_speaker_real_pipeline(self, video, tmp_path, transcript):
        result = runner.invoke(
            cli.app,
            ["mute", str(video), "--speaker", "SPEAKER_09", "--out-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 1
        assert "SPEAKER_00, SPEAKER_01" in plain(result)


class TestRun:
    def test_run_transcribes_then_mutes(self, video, tmp_path, fake_transcribe, fake_mute):
        result = runner.invoke(
            cli.app,
            [
                "run",
                str(video),
                "--mute-speaker",
                "SPEAKER_01",
                "--burn-subs",
                "--out-dir",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert fake_transcribe["videos"] == [video]
        assert fake_mute[0]["speaker"] == ["SPEAKER_01"]
        assert fake_mute[0]["burn_subs"] is True

    def test_run_output_single_video(self, video, tmp_path, fake_transcribe, fake_mute):
        custom = tmp_path / "custom.mp4"
        result = runner.invoke(
            cli.app, ["run", str(video), "--mute-speaker", "S", "--output", str(custom)]
        )
        assert result.exit_code == 0, result.output
        assert fake_mute[0]["output"] == custom

    def test_run_output_rejects_folder(self, tmp_path, fake_transcribe, fake_mute):
        folder = tmp_path / "videos"
        folder.mkdir()
        (folder / "a.mp4").write_bytes(b"")
        (folder / "b.mp4").write_bytes(b"")
        result = runner.invoke(
            cli.app, ["run", str(folder), "--mute-speaker", "S", "-o", str(tmp_path / "x.mp4")]
        )
        assert result.exit_code == 2
        assert "--output" in plain(result) and "2 videos" in plain(result)
        assert fake_transcribe["videos"] == [] and fake_mute == []

    def test_run_skips_transcription_when_present(
        self, video, tmp_path, transcript, fake_transcribe, fake_mute
    ):
        result = runner.invoke(
            cli.app,
            ["run", str(video), "--mute-speaker", "SPEAKER_01", "--out-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 0, result.output
        assert fake_transcribe["videos"] == []
        assert len(fake_mute) == 1

    def test_run_checks_subtitles_filter_before_work(
        self, video, tmp_path, monkeypatch, fake_transcribe, fake_mute
    ):
        def boom():
            raise FFmpegError("no libass")

        monkeypatch.setattr(cli.ffmpeg, "require_subtitles_filter", boom)
        result = runner.invoke(cli.app, ["run", str(video), "--mute-speaker", "S", "--burn-subs"])
        assert result.exit_code == 1 and "no libass" in plain(result)
        assert fake_transcribe["videos"] == [] and fake_mute == []


class TestSpeakers:
    def test_prints_table_from_existing_transcript(
        self, video, tmp_path, transcript, fake_transcribe
    ):
        result = runner.invoke(
            cli.app, ["speakers", str(video), "--out-dir", str(tmp_path / "out")]
        )
        assert result.exit_code == 0, result.output
        assert "SPEAKER_00" in plain(result) and "SPEAKER_01" in plain(result)
        assert "Welcome." in plain(result)
        assert "0:06" in plain(result)  # SPEAKER_01 talk time 6.5 s
        assert fake_transcribe["videos"] == []
        assert "--mute-speaker" in plain(result)

    def test_transcribes_when_missing(self, video, tmp_path, fake_transcribe):
        result = runner.invoke(
            cli.app, ["speakers", str(video), "--out-dir", str(tmp_path / "out")]
        )
        assert result.exit_code == 0, result.output
        assert fake_transcribe["videos"] == [video]
        assert "SPEAKER_00" in plain(result)


def test_transcribe_no_diarize(video, tmp_path, fake_transcribe):
    result = runner.invoke(
        cli.app, ["transcribe", str(video), "--no-diarize", "--out-dir", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.output
    assert fake_transcribe["options"].diarize is False
