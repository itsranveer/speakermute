# Contributing

Thanks for your interest. Bug reports, fixes and small focused features are welcome.

## Development setup

```bash
git clone https://github.com/itsranveer/speakermute.git
cd speakermute
uv sync                  # installs the package (editable) plus dev tools
uv run pytest            # unit tests: whisperx and ffmpeg are mocked, nothing is downloaded
uv run ruff check .      # lint
uv run ruff format .     # format
```

`ffmpeg` is only needed to run the tool itself, not the tests. Trying a change end to end needs a
video and a Hugging Face token (see the README).

## Layout

```
src/speakermute/
├── cli.py          typer commands: transcribe, speakers, mute, run
├── pipeline.py     transcribe_video() and mute_video(), the steps the commands compose
├── transcribe.py   WhisperX wrapper and transcript.json read/write
├── ffmpeg.py       ffmpeg/ffprobe subprocess wrapper with progress bars
├── ranges.py       pure helpers: pad/merge time ranges, build the volume filtergraph
├── srt.py          SRT writer with speaker labels and --rename
├── speakers.py     per-speaker summary table
└── paths.py        out/<stem>/ layout
```

`ranges.py`, `srt.py` and `speakers.py` are pure and fully unit-tested; keep them that way.
ffmpeg and WhisperX calls go through `ffmpeg.py` and `transcribe.py` so tests can replace them.

## Pull requests

* Keep commits small with clear messages.
* Add or update tests for behaviour you change. `uv run pytest` and `uv run ruff check .` must pass
  (CI runs both).
* If you touch the WhisperX integration, note which `whisperx` version you verified against; its API
  has changed between releases.
* Update the README when you add or change a command-line option.
