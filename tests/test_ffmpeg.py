import subprocess

import pytest

from speakermute import ffmpeg as ff


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(ff.FFMPEG_ENV, raising=False)
    monkeypatch.delenv(ff.FFPROBE_ENV, raising=False)


class FakeProc:
    """Stand-in for subprocess.Popen that replays canned stdout lines."""

    def __init__(self, lines, returncode=0):
        self.stdout = iter(lines)
        self.returncode = returncode

    def wait(self):
        return self.returncode


class TestExecutableResolution:
    def test_defaults(self):
        assert ff.ffmpeg_executable() == "ffmpeg"
        assert ff.ffprobe_executable() == "ffprobe"

    def test_env_override(self, monkeypatch, tmp_path):
        exe = tmp_path / "bin" / "ffmpeg"
        exe.parent.mkdir()
        exe.write_text("")
        (tmp_path / "bin" / "ffprobe").write_text("")
        monkeypatch.setenv(ff.FFMPEG_ENV, str(exe))
        assert ff.ffmpeg_executable() == str(exe)
        # ffprobe resolved next to the overridden ffmpeg
        assert ff.ffprobe_executable() == str(tmp_path / "bin" / "ffprobe")

    def test_explicit_ffprobe_env(self, monkeypatch):
        monkeypatch.setenv(ff.FFPROBE_ENV, "/x/ffprobe")
        assert ff.ffprobe_executable() == "/x/ffprobe"


class TestRequireFfmpeg:
    def test_ok(self, monkeypatch):
        monkeypatch.setattr(ff.shutil, "which", lambda name: f"/usr/bin/{name}")
        ff.require_ffmpeg()

    def test_missing(self, monkeypatch):
        monkeypatch.setattr(ff.shutil, "which", lambda name: None)
        with pytest.raises(ff.FFmpegError) as exc:
            ff.require_ffmpeg()
        assert "ffmpeg" in str(exc.value) and "ffprobe" in str(exc.value)
        assert "brew install ffmpeg" in str(exc.value)


class TestFilters:
    FILTERS_OUTPUT = """Filters:
  T.. = Timeline support
 T.. volume            A->A       Change input volume.
 ... subtitles         V->V       Render text subtitles onto input video using the libass library.
 ... scale             V->V       Scale the input video size and/or convert the image format.
"""

    def test_available_filters_parses_names(self, monkeypatch):
        monkeypatch.setattr(
            ff.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, self.FILTERS_OUTPUT, ""),
        )
        assert ff.available_filters() == {"volume", "subtitles", "scale"}

    def test_require_subtitles_filter_missing(self, monkeypatch):
        monkeypatch.setattr(ff, "available_filters", lambda: {"volume"})
        monkeypatch.setattr(ff.shutil, "which", lambda name: "/opt/homebrew/bin/ffmpeg")
        with pytest.raises(ff.FFmpegError) as exc:
            ff.require_subtitles_filter()
        assert "ffmpeg-full" in str(exc.value)

    def test_require_subtitles_filter_present(self, monkeypatch):
        monkeypatch.setattr(ff, "available_filters", lambda: {"volume", "subtitles"})
        ff.require_subtitles_filter()


class TestProbeDuration:
    def test_parses(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "123.456\n", "")

        monkeypatch.setattr(ff.subprocess, "run", fake_run)
        assert ff.probe_duration("in.mp4") == pytest.approx(123.456)
        assert captured["cmd"][0] == "ffprobe"
        assert "format=duration" in captured["cmd"]
        assert captured["cmd"][-1] == "in.mp4"

    def test_failure(self, monkeypatch):
        monkeypatch.setattr(
            ff.subprocess,
            "run",
            lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", "No such file"),
        )
        with pytest.raises(ff.FFmpegError, match="No such file"):
            ff.probe_duration("missing.mp4")

    def test_unparseable(self, monkeypatch):
        monkeypatch.setattr(
            ff.subprocess,
            "run",
            lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, "N/A\n", ""),
        )
        with pytest.raises(ff.FFmpegError, match="could not parse"):
            ff.probe_duration("weird.mp4")


class TestProgressParsing:
    def test_out_time_us(self):
        assert ff.parse_progress_line("out_time_us=1500000\n") == pytest.approx(1.5)

    def test_out_time_ms_is_also_microseconds(self):
        assert ff.parse_progress_line("out_time_ms=2000000") == pytest.approx(2.0)

    def test_other_keys_ignored(self):
        assert ff.parse_progress_line("frame=12") is None
        assert ff.parse_progress_line("progress=end") is None

    def test_non_progress_line(self):
        assert ff.parse_progress_line("Error opening input") is None
        assert ff.is_progress_line("Error opening input") is False
        assert ff.is_progress_line("speed=2.5x") is True

    def test_negative_clamped(self):
        assert ff.parse_progress_line("out_time_us=-9223372036854775808") == 0.0


class TestRunFfmpeg:
    def test_success_reports_progress(self, monkeypatch):
        lines = [
            "frame=1\n",
            "out_time_us=500000\n",
            "progress=continue\n",
            "out_time_us=1000000\n",
            "progress=end\n",
        ]
        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            return FakeProc(lines)

        monkeypatch.setattr(ff.subprocess, "Popen", fake_popen)
        seen = []
        ff.run_ffmpeg(["-i", "a.mp4", "b.mp4"], total_duration=1.0, on_progress=seen.append)
        assert seen == [0.5, 1.0]
        cmd = captured["cmd"]
        assert cmd[0] == "ffmpeg"
        assert cmd[-3:] == ["-i", "a.mp4", "b.mp4"]
        assert "-progress" in cmd and "pipe:1" in cmd and "-y" in cmd and "-nostdin" in cmd

    def test_failure_raises_with_error_output(self, monkeypatch):
        lines = ["frame=1\n", "a.mp4: No such file or directory\n"]
        monkeypatch.setattr(ff.subprocess, "Popen", lambda cmd, **k: FakeProc(lines, returncode=1))
        with pytest.raises(ff.FFmpegError) as exc:
            ff.run_ffmpeg(["-i", "a.mp4", "b.mp4"], description="Muting audio")
        msg = str(exc.value)
        assert "code 1" in msg and "Muting audio" in msg
        assert "No such file or directory" in msg
        assert "frame=1" not in msg


class TestCommandBuilders:
    @pytest.fixture
    def capture(self, monkeypatch):
        calls = []

        def fake_run_ffmpeg(args, **kwargs):
            calls.append((list(args), kwargs))

        monkeypatch.setattr(ff, "run_ffmpeg", fake_run_ffmpeg)
        return calls

    def test_extract_audio(self, capture, tmp_path):
        wav = tmp_path / "nested" / "audio.wav"
        out = ff.extract_audio("in.mp4", wav, total_duration=42.0)
        assert out == wav and wav.parent.exists()
        args, kwargs = capture[0]
        assert args[:2] == ["-i", "in.mp4"]
        assert "-vn" in args
        assert args[args.index("-ar") + 1] == "16000"
        assert args[args.index("-ac") + 1] == "1"
        assert args[args.index("-c:a") + 1] == "pcm_s16le"
        assert args[-1] == str(wav)
        assert kwargs["total_duration"] == 42.0

    def test_mute_with_filter_script_copies_video(self, capture, tmp_path, monkeypatch):
        monkeypatch.setattr(ff, "ffmpeg_version", lambda: (9, 0))
        script = tmp_path / "mute_filter.txt"
        out = tmp_path / "x_muted.mp4"
        ff.mute_with_filter_script("in.mp4", out, script, total_duration=10.0)
        args, kwargs = capture[0]
        assert args[args.index("-/filter:a") + 1] == str(script)
        assert "-filter_script:a" not in args
        assert args[args.index("-c:v") + 1] == "copy"
        assert "-vf" not in args and "libx264" not in args
        assert args[args.index("-c:a") + 1] == "aac"
        assert args[args.index("-movflags") + 1] == "+faststart"
        assert args[-1] == str(out)
        assert kwargs["description"] == "Muting audio"
        assert kwargs["total_duration"] == 10.0

    @pytest.mark.parametrize("suffix", [".mkv", ".MKV"])
    def test_mute_to_matroska_has_no_faststart(self, capture, tmp_path, monkeypatch, suffix):
        monkeypatch.setattr(ff, "ffmpeg_version", lambda: (9, 0))
        script = tmp_path / "mute_filter.txt"
        out = tmp_path / f"x_muted{suffix}"
        ff.mute_with_filter_script("in.mkv", out, script)
        args, _ = capture[0]
        assert "-movflags" not in args  # an MP4/MOV muxer option
        assert args[args.index("-c:v") + 1] == "copy"
        assert args[-1] == str(out)

    def test_mute_with_filter_script_burns_subtitles_in_one_pass(
        self, capture, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(ff, "ffmpeg_version", lambda: (9, 0))
        script = tmp_path / "mute_filter.txt"
        srt = tmp_path / "subtitles.srt"
        out = tmp_path / "x_muted.mp4"
        ff.mute_with_filter_script("in.mp4", out, script, subtitles=srt, crf=18, preset="fast")
        args, kwargs = capture[0]
        assert len(capture) == 1  # single ffmpeg invocation
        assert args[args.index("-/filter:a") + 1] == str(script)
        vf = args[args.index("-vf") + 1]
        assert vf.startswith("subtitles=filename=") and vf.endswith("subtitles.srt")
        assert args[args.index("-c:v") + 1] == "libx264"
        assert args[args.index("-crf") + 1] == "18"
        assert args[args.index("-preset") + 1] == "fast"
        assert args[args.index("-c:a") + 1] == "aac"
        assert args[-1] == str(out)
        assert kwargs["description"] == "Muting audio + burning subtitles"


def test_write_filter_script(tmp_path):
    path = tmp_path / "out" / "mute_filter.txt"
    ff.write_filter_script(path, "volume=enable='between(t,0.000,1.000)':volume=0")
    assert path.read_text(encoding="utf-8") == "volume=enable='between(t,0.000,1.000)':volume=0"


class TestEscapeFilterPath:
    """Decode the escaped string the way ffmpeg does (outer graph level, then
    inner option level) and check we get the original path back."""

    @staticmethod
    def _unescape(text: str) -> str:
        out = []
        i = 0
        while i < len(text):
            if text[i] == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
            else:
                out.append(text[i])
                i += 1
        return "".join(out)

    @pytest.mark.parametrize(
        "path",
        [
            "/home/me/out/video/subtitles.srt",
            "/Users/me/My Videos/out/Episode #1/subtitles.srt",
            "C:\\Users\\me\\out\\video\\subtitles.srt",
            "/tmp/it's [a] test, ok; yes=no:maybe.srt",
        ],
    )
    def test_round_trips_through_two_parsing_levels(self, path):
        escaped = ff.escape_filter_path(path)
        assert self._unescape(self._unescape(escaped)) == path

    def test_windows_drive_colon(self):
        assert ff.escape_filter_path("C:\\a.srt") == "C\\\\:\\\\\\\\a.srt"

    def test_plain_posix_path_untouched(self):
        assert ff.escape_filter_path("/a/b/c.srt") == "/a/b/c.srt"

    def test_no_unescaped_specials_remain(self):
        escaped = ff.escape_filter_path("/x/a:b,c;d[e]'f g.srt")
        # every special char must be preceded by a backslash at the outer level
        for ch in ",;[]'":
            idx = escaped.find(ch)
            assert idx > 0 and escaped[idx - 1] == "\\"


def test_check_same_duration(monkeypatch):
    durations = iter([100.0, 100.05])
    monkeypatch.setattr(ff, "probe_duration", lambda p: next(durations))
    assert ff.check_same_duration("a", "b") == (100.0, 100.05, True)
    durations = iter([100.0, 97.0])
    assert ff.check_same_duration("a", "b")[2] is False
