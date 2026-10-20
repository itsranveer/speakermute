import json
import wave
from pathlib import Path

import numpy as np
import pytest

from speakermute import transcribe as tr
from speakermute.errors import MissingTokenError, TranscriptError


class TestResolution:
    def test_detect_device_cpu(self, fake_whisperx):
        assert tr.detect_device() == "cpu"
        assert tr.resolve_device("auto") == "cpu"

    def test_detect_device_cuda(self, fake_whisperx):
        fake_whisperx["cuda"] = True
        assert tr.resolve_device("auto") == "cuda"
        assert tr.resolve_device("CUDA") == "cuda"

    def test_explicit_cuda_without_gpu_fails(self, fake_whisperx):
        with pytest.raises(ValueError, match="CUDA"):
            tr.resolve_device("cuda")

    def test_bad_device(self, fake_whisperx):
        with pytest.raises(ValueError):
            tr.resolve_device("mps")

    def test_model_defaults(self):
        assert tr.resolve_model("auto", "cpu") == "medium"
        assert tr.resolve_model("auto", "cuda") == "large-v3"
        assert tr.resolve_model("small", "cuda") == "small"

    def test_compute_type_defaults(self):
        assert tr.resolve_compute_type("auto", "cpu") == "int8"
        assert tr.resolve_compute_type("auto", "cuda") == "float16"
        assert tr.resolve_compute_type("float32", "cpu") == "float32"

    def test_batch_size_defaults(self):
        assert tr.resolve_batch_size(None, "cpu") == 8
        assert tr.resolve_batch_size(None, "cuda") == 16
        assert tr.resolve_batch_size(4, "cuda") == 4


class TestHfToken:
    def test_explicit_wins(self, fake_whisperx, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "env-token")
        assert tr.resolve_hf_token("explicit") == "explicit"

    def test_env(self, fake_whisperx, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "env-token")
        assert tr.resolve_hf_token() == "env-token"

    def test_cached_login(self, fake_whisperx):
        fake_whisperx["cached_token"] = "cached"
        assert tr.resolve_hf_token() == "cached"

    def test_none(self, fake_whisperx):
        assert tr.resolve_hf_token() is None


class TestTelemetry:
    def test_disabled_by_default(self, fake_whisperx, monkeypatch):
        monkeypatch.delenv(tr.PYANNOTE_TELEMETRY_ENV, raising=False)
        tr.Transcriber(tr.TranscribeOptions(diarize=False))
        assert __import__("os").environ[tr.PYANNOTE_TELEMETRY_ENV] == "false"

    def test_user_choice_respected(self, fake_whisperx, monkeypatch):
        monkeypatch.setenv(tr.PYANNOTE_TELEMETRY_ENV, "true")
        tr.Transcriber(tr.TranscribeOptions(diarize=False))
        assert __import__("os").environ[tr.PYANNOTE_TELEMETRY_ENV] == "true"


class TestLoadWav:
    def test_loads_float32(self, wav_file):
        audio = tr.load_wav(wav_file)
        assert audio.dtype == np.float32
        assert len(audio) == 8000
        assert np.abs(audio).max() <= 1.0

    def test_rejects_wrong_format(self, tmp_path):
        path = tmp_path / "stereo.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(44100)
            wav.writeframes(b"\x00" * 400)
        with pytest.raises(ValueError, match="expected mono"):
            tr.load_wav(path)


class TestTranscriber:
    def test_requires_token_when_diarizing(self, fake_whisperx):
        with pytest.raises(MissingTokenError) as exc:
            tr.Transcriber(tr.TranscribeOptions())
        assert "huggingface.co/pyannote/speaker-diarization-community-1" in str(exc.value)
        assert "HF_TOKEN" in str(exc.value)

    def test_no_token_needed_without_diarization(self, fake_whisperx):
        t = tr.Transcriber(tr.TranscribeOptions(diarize=False))
        assert t.hf_token is None

    def test_cpu_defaults(self, fake_whisperx, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "tok")
        t = tr.Transcriber(tr.TranscribeOptions())
        assert (t.device, t.model_name, t.compute_type, t.batch_size) == (
            "cpu",
            "medium",
            "int8",
            8,
        )
        assert "model=medium" in t.describe() and "device=cpu" in t.describe()
        assert "speakers=auto" in t.describe()
        settings = t.settings()
        assert "hf_token" not in settings
        assert settings["model"] == "medium"
        assert (settings["min_speakers"], settings["max_speakers"]) == (None, None)

    def test_full_pipeline(self, fake_whisperx, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "tok")
        t = tr.Transcriber(
            tr.TranscribeOptions(model="small", language="en", min_speakers=2, max_speakers=3)
        )
        audio = np.zeros(16000, dtype=np.float32)
        assert "speakers=2-3" in t.describe()

        result = t.transcribe_audio(audio)

        assert fake_whisperx["load_model"] == [
            {"arch": "small", "device": "cpu", "compute_type": "int8", "language": "en"}
        ]
        assert fake_whisperx["transcribe"] == {"batch_size": 8, "language": "en", "samples": 16000}
        assert fake_whisperx["load_align_model"] == [{"language": "en", "device": "cpu"}]
        assert fake_whisperx["align"]["model"] == "align-model"
        assert fake_whisperx["diarizer_init"] == {
            "model_name": "pyannote/speaker-diarization-community-1",
            "token": "tok",
            "device": "cpu",
        }
        assert fake_whisperx["diarize"] == {"min": 2, "max": 3, "samples": 16000}
        assert fake_whisperx["assign"] == {"df": "DIARIZE_DF", "fill_nearest": True}
        assert result["language"] == "en"
        assert [s["speaker"] for s in result["segments"]] == ["SPEAKER_00", "SPEAKER_01"]

    def test_models_are_loaded_once_across_videos(self, fake_whisperx, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "tok")
        t = tr.Transcriber(tr.TranscribeOptions())
        audio = np.zeros(1600, dtype=np.float32)
        t.transcribe_audio(audio)
        t.transcribe_audio(audio)
        assert len(fake_whisperx["load_model"]) == 1
        assert len(fake_whisperx["load_align_model"]) == 1
        t.release()
        t.transcribe_audio(audio)
        assert len(fake_whisperx["load_model"]) == 2

    def test_language_auto_uses_detected_language_for_alignment(self, fake_whisperx):
        t = tr.Transcriber(tr.TranscribeOptions(language=None, diarize=False))
        result = t.transcribe_audio(np.zeros(1600, dtype=np.float32))
        assert fake_whisperx["load_model"][0]["language"] is None
        assert fake_whisperx["load_align_model"] == [{"language": "de", "device": "cpu"}]
        assert result["language"] == "de"

    def test_diarize_off_skips_diarization(self, fake_whisperx):
        t = tr.Transcriber(tr.TranscribeOptions(diarize=False))
        result = t.transcribe_audio(np.zeros(1600, dtype=np.float32))
        assert "diarize" not in fake_whisperx
        assert all("speaker" not in s for s in result["segments"])


class TestTranscriptJson:
    RAW = {
        "language": "en",
        "segments": [
            {
                "start": np.float64(0.123456),
                "end": 1.0,
                "text": "  Hello   there. ",
                "speaker": "SPEAKER_00",
                "words": [
                    {
                        "word": "Hello",
                        "start": 0.1,
                        "end": 0.4,
                        "score": np.float32(0.95),
                        "speaker": "SPEAKER_00",
                    },
                    {"word": "there.", "start": 0.5, "end": 1.0, "score": 0.8},
                    {"word": "42"},  # numerals get no timestamps from the aligner
                ],
            },
            {"start": 2.0, "end": 3.0, "text": "unlabelled"},
        ],
    }

    def test_normalize_segments(self):
        segs = tr.normalize_segments(self.RAW["segments"])
        assert segs[0] == {
            "start": 0.123,
            "end": 1.0,
            "speaker": "SPEAKER_00",
            "text": "Hello there.",
            "words": [
                {"word": "Hello", "start": 0.1, "end": 0.4, "score": 0.95, "speaker": "SPEAKER_00"},
                {"word": "there.", "start": 0.5, "end": 1.0, "score": 0.8},
                {"word": "42"},
            ],
        }
        assert list(segs[0].keys()) == ["start", "end", "speaker", "text", "words"]
        assert "speaker" not in segs[1]
        assert isinstance(segs[0]["start"], float) and not isinstance(segs[0]["start"], np.floating)

    def test_build_save_load_round_trip(self, tmp_path):
        transcript = tr.build_transcript(
            self.RAW,
            video=Path("videos/episode.mp4"),
            duration=np.float64(3.5),
            settings={"model": "medium"},
        )
        assert transcript["video"] == "episode.mp4"
        assert transcript["duration"] == 3.5
        assert transcript["language"] == "en"
        assert transcript["speakers"] == ["SPEAKER_00"]
        assert transcript["settings"] == {"model": "medium"}

        path = tr.save_transcript(tmp_path / "out" / "transcript.json", transcript)
        text = path.read_text(encoding="utf-8")
        json.loads(text)  # valid JSON, numpy types converted
        loaded = tr.load_transcript(path)
        assert loaded["segments"] == transcript["segments"]

    def test_load_missing(self, tmp_path):
        with pytest.raises(TranscriptError, match="transcribe"):
            tr.load_transcript(tmp_path / "transcript.json")

    def test_load_invalid_json(self, tmp_path):
        p = tmp_path / "transcript.json"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(TranscriptError, match="not valid JSON"):
            tr.load_transcript(p)

    def test_load_without_segments(self, tmp_path):
        p = tmp_path / "transcript.json"
        p.write_text('{"foo": 1}', encoding="utf-8")
        with pytest.raises(TranscriptError, match="segments"):
            tr.load_transcript(p)

    def test_load_bad_segment(self, tmp_path):
        p = tmp_path / "transcript.json"
        p.write_text('{"segments": [{"start": 1}]}', encoding="utf-8")
        with pytest.raises(TranscriptError, match="segment #0"):
            tr.load_transcript(p)
