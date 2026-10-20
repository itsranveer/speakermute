# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-20

First release: mute chosen speakers in any video (podcasts, panels, lectures, meetings,
recorded calls) without cutting it.

### Added

* `transcribe`: WhisperX transcription, word alignment and pyannote speaker diarization;
  writes `transcript.json` and `subtitles.srt` with speaker labels (`--rename OLD=NEW`).
  Videos that already have a transcript are skipped unless `--force`.
* `speakers`: per-speaker talk time, segment count and first lines, to see who has which label.
* `mute`: silences every segment of one or more speakers (`--speaker`, repeatable) from an
  existing, hand-editable `transcript.json` with an ffmpeg `volume` filter; the video stream is
  copied so the length is unchanged. `--pad` widens each range, `--burn-subs` burns the subtitles
  in during the same pass, and `--output`/`-o` picks the output path (default
  `out/<stem>/<stem>_muted.<ext>`). The output keeps the input's container (`.mp4`, `.mov`, `.m4v`,
  `.mkv`); `.webm` and anything else becomes `.mkv`, since the copied video stream may not fit in
  MP4 and WebM cannot hold AAC. `--output` pointing at the input video is refused.
* `run`: transcribe (if needed) + mute in one go; works on folders, `--mute-speaker` repeatable.
* pyannote estimates the speaker count by default; `--min-speakers` / `--max-speakers` bound it
  when the number is known.
* CPU-friendly defaults (`medium` / `int8`), CUDA auto-detection, progress bars for every stage.
* ffmpeg 7+ `-/filter:a` support (ffmpeg 9 removed `-filter_script`), `FFMPEG`/`FFPROBE`
  environment overrides, and a clear error when the build lacks libass.
* Mute expressions are split across chained filters (50 ranges each) to stay under ffmpeg's
  expression-parser limit.
* pyannote's anonymous usage metrics are switched off unless `PYANNOTE_METRICS_ENABLED` is set.
  Without `-v`, INFO chatter from whisperx, Lightning and pyannote and two known-harmless warnings
  (`torchcodec is not installed correctly`, `std(): degrees of freedom is <= 0`) are hidden.

[0.1.0]: https://github.com/itsranveer/speakermute/releases/tag/v0.1.0
