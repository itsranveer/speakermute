"""Shared fakes: an in-memory ``whisperx`` + ``torch`` so no models are ever loaded."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def fake_whisperx(monkeypatch):
    """Install fake ``whisperx``, ``whisperx.diarize``, ``torch`` and ``huggingface_hub``.

    Returns a dict that records every call so tests can assert on arguments.
    """
    calls: dict = {}

    class FakeASR:
        def transcribe(self, audio, batch_size=None, language=None, progress_callback=None):
            calls["transcribe"] = {
                "batch_size": batch_size,
                "language": language,
                "samples": int(len(audio)),
            }
            if progress_callback:
                progress_callback(50.0)
                progress_callback(100.0)
            return {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "  Hello there. "},
                    {"start": 1.5, "end": 3.0, "text": "Hi, thanks."},
                ],
                "language": language or "de",
            }

    def load_model(arch, device, compute_type=None, language=None, threads=None, **kwargs):
        calls.setdefault("load_model", []).append(
            {"arch": arch, "device": device, "compute_type": compute_type, "language": language}
        )
        return FakeASR()

    def load_align_model(language_code, device):
        calls.setdefault("load_align_model", []).append(
            {"language": language_code, "device": device}
        )
        return ("align-model", {"language": language_code})

    def align(
        segments,
        model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
        progress_callback=None,
    ):
        calls["align"] = {"model": model, "device": device, "n": len(segments)}
        if progress_callback:
            progress_callback(100.0)
        out = []
        for seg in segments:
            words = [
                {
                    "word": w,
                    "start": np.float64(seg["start"]),
                    "end": seg["end"],
                    "score": np.float32(0.9),
                }
                for w in seg["text"].split()
            ]
            out.append({**seg, "start": np.float64(seg["start"]), "words": words})
        return {"segments": out, "word_segments": []}

    class FakeDiarizer:
        def __init__(self, model_name=None, token=None, device=None):
            calls["diarizer_init"] = {"model_name": model_name, "token": token, "device": device}

        def __call__(self, audio, min_speakers=None, max_speakers=None, progress_callback=None):
            calls["diarize"] = {
                "min": min_speakers,
                "max": max_speakers,
                "samples": int(len(audio)),
            }
            if progress_callback:
                progress_callback(100.0)
            return "DIARIZE_DF"

    def assign_word_speakers(df, result, fill_nearest=False):
        calls["assign"] = {"df": df, "fill_nearest": fill_nearest}
        for i, seg in enumerate(result["segments"]):
            seg["speaker"] = f"SPEAKER_0{i}"
            for w in seg.get("words", []):
                w["speaker"] = seg["speaker"]
        return result

    whisperx = types.ModuleType("whisperx")
    whisperx.load_model = load_model
    whisperx.load_align_model = load_align_model
    whisperx.align = align
    whisperx.assign_word_speakers = assign_word_speakers
    diarize = types.ModuleType("whisperx.diarize")
    diarize.DiarizationPipeline = FakeDiarizer
    whisperx.diarize = diarize

    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: calls.get("cuda", False), empty_cache=lambda: None
    )

    hub = types.ModuleType("huggingface_hub")
    hub.get_token = lambda: calls.get("cached_token")

    monkeypatch.setitem(sys.modules, "whisperx", whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", diarize)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return calls


@pytest.fixture
def wav_file(tmp_path):
    """A 0.5 s mono 16 kHz 16-bit WAV."""
    import wave

    path = tmp_path / "audio.wav"
    samples = (np.sin(np.linspace(0, 100, 8000)) * 1000).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())
    return path
