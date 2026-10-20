# speakermute

[![CI](https://github.com/itsranveer/speakermute/actions/workflows/ci.yml/badge.svg)](https://github.com/itsranveer/speakermute/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)

Mute one or more speakers in a video without cutting it.

1. **Transcribe + diarize** the video with [WhisperX](https://github.com/m-bain/whisperX) (Whisper for text, wav2vec2 for word timestamps, pyannote for "who spoke when").
2. **Mute the speakers you pick** with ffmpeg. The picture, the length and everyone else's audio stay as they were; the chosen voices become silent gaps.
3. **Subtitles** (`.srt`, with speaker labels) let you check what was said in the gaps. Optionally burn them into the video.

```
out/<video-stem>/
├── transcript.json        # WhisperX segments: start, end, speaker, text, words (hand-editable)
├── subtitles.srt          # [SPEAKER_00]: text   (labels renamable with --rename)
├── mute_filter.txt        # the ffmpeg filter script that was applied
└── <stem>_muted.mp4       # chosen speakers silenced (.mkv/.mov keep their container, .webm -> .mkv)
```

## What it works on

Any recording of people talking: podcasts, panel discussions, lectures and Q&A sessions, meetings,
interviews, recorded calls. Some things people use it for:

* **Rehearse your side of a conversation**: mute one participant and speak in the gaps, with the
  subtitles as the answer key.
* **Remove a co-host or guest** from an episode while keeping the rest of the conversation intact.
* **Take the lecturer out of a Q&A** to work through the audience's questions yourself.
* **Build listening exercises**: silence one side of a dialogue and keep subtitles for the answers.

### Songs and music

Songs are supported only in a best-effort way, and you should know two limits up front:

* The tool silences the **whole mix** during the chosen singer's lines. It does not separate stems, so
  the instruments and anyone singing at the same time go silent too. Keeping the backing track
  would need source separation (for example Demucs), which this tool does not do.
* Whisper and pyannote are trained on **speech**. On singing over music the speaker labels are
  unreliable and the lyrics are often misheard, so expect to hand-fix `transcript.json` (see
  "Fixing diarization mistakes") before muting. Giving the real number of singers with
  `--min-speakers N --max-speakers N` helps.

## Setup

### 1. Tools

* **[uv](https://docs.astral.sh/uv/)** manages Python (3.11 or 3.12) and the virtualenv.
  `brew install uv` / `pipx install uv` / see the uv docs.
* **ffmpeg + ffprobe** on your `PATH`. The tool checks at startup and tells you if they are missing.
  * Linux: `sudo apt install ffmpeg` (distro builds include libass).
  * macOS: `brew install ffmpeg`. **For `--burn-subs` you need libass**, which the slim Homebrew
    `ffmpeg` formula no longer bundles. Install `brew install ffmpeg-full` (keg-only) and put
    `FFMPEG=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg` in your `.env`. Everything else works with the
    regular build.
  * Windows: `winget install ffmpeg`.
* A CUDA GPU is optional. On CPU-only machines (including Apple Silicon, where ctranslate2 cannot use
  the GPU) the defaults switch to `--model medium --compute-type int8`.

### 2. Hugging Face token (for speaker diarization)

pyannote's diarization models are gated. Once:

1. Create a read token at <https://huggingface.co/settings/tokens>.
2. While logged in, accept the terms of the pipeline you will use:
   * default: <https://huggingface.co/pyannote/speaker-diarization-community-1>
   * legacy (`--diarization-model pyannote/speaker-diarization-3.1`):
     <https://huggingface.co/pyannote/speaker-diarization-3.1> **and**
     <https://huggingface.co/pyannote/segmentation-3.0>
3. `cp .env.example .env` and paste the token as `HF_TOKEN=...`. The `.env` is read from the directory
   you run the command in (or any parent). `huggingface-cli login` works as a fallback.

pyannote.audio 4 sends anonymous usage metrics by default. This tool switches that off
(`PYANNOTE_METRICS_ENABLED=false`) unless you set the variable yourself.

### 3. Install

As a tool, straight from GitHub (no clone):

```bash
uv tool install git+https://github.com/itsranveer/speakermute   # a few GB of torch/whisperx, one time
speakermute --help
# or one-off, without installing:
uvx --from git+https://github.com/itsranveer/speakermute speakermute --help
```

Or from a clone, which is what the rest of this README assumes (`uv run speakermute ...`):

```bash
git clone https://github.com/itsranveer/speakermute.git
cd speakermute
uv sync                       # downloads torch, whisperx, pyannote... (a few GB, one time)
uv run speakermute --help
```

Whisper, alignment and diarization model weights download on first use into the Hugging Face /
torch caches (`~/.cache/huggingface`, `~/.cache/torch`): about 1.5 GB for `medium` (3 GB for
`large-v3`), 360 MB for the English alignment model and 35 MB for diarization.

## Workflow

```bash
# 1. Transcribe and see who is who. Labels are assigned by ORDER OF FIRST SPEECH, not by role,
#    so always check which label belongs to whom before muting.
uv run speakermute speakers videos/episode-12.mp4
#  ┃ Speaker    ┃ Talk time ┃ Segments ┃ First speaks at ┃ First lines                         ┃
#  │ SPEAKER_00 │     18:12 │      141 │            0:03 │ • Welcome back to the show. Today…   │
#  │ SPEAKER_01 │     31:40 │      158 │            0:21 │ • Thanks for having me. So I…        │

# 2. Mute the guest (re-uses the transcript from step 1) and burn subtitles.
uv run speakermute run videos/episode-12.mp4 --mute-speaker SPEAKER_01 --burn-subs \
    --rename SPEAKER_00=Host --rename SPEAKER_01=Guest

# 3. The result is out/episode-12/episode-12_muted.mp4, with subtitles.srt and
#    transcript.json next to it. Use -o to put the video somewhere else:
uv run speakermute run videos/episode-12.mp4 --mute-speaker SPEAKER_01 -o episode-12-host-only.mp4
```

Whole folders work too: `uv run speakermute run videos/ --mute-speaker SPEAKER_01`. The label is applied to
every video, so run `speakers` per video first if the person is not always the same label.
(`--output` only works with a single video.)

### Speaker count

By default pyannote estimates how many people are talking. When you know the number, pass it as
`--min-speakers N --max-speakers N`: it makes diarization noticeably more accurate. A two-person
call, and a panel of three where everyone but one person is muted:

```bash
uv run speakermute run "videos/Call with Sam.mp4" --min-speakers 2 --max-speakers 2 \
    --mute-speaker SPEAKER_01 --burn-subs --rename SPEAKER_00=Me --rename SPEAKER_01=Sam

uv run speakermute speakers videos/panel.mp4 --min-speakers 3 --max-speakers 3
uv run speakermute run videos/panel.mp4 --mute-speaker SPEAKER_00 --mute-speaker SPEAKER_02 --burn-subs
```

Add `--force` if a transcript made with different speaker bounds already exists; `run` and `speakers`
otherwise re-use it.

### Fixing diarization mistakes

Diarization is not perfect. Because transcription and muting are decoupled, you can hand-edit
`out/<stem>/transcript.json` (change a segment's `"speaker"`, or delete it) and re-run just the mute
step without re-transcribing:

```bash
uv run speakermute mute videos/episode-12.mp4 --speaker SPEAKER_01 --burn-subs
# or from an edited copy elsewhere:
uv run speakermute mute videos/episode-12.mp4 --speaker SPEAKER_01 --transcript my-fixed.json
```

`mute` regenerates `subtitles.srt` from the transcript, so your edits show up there as well.

## Commands

| Command | What it does |
|---|---|
| `transcribe <video-or-folder>` | WhisperX transcribe → align → diarize. Writes `transcript.json` + `subtitles.srt`. Skips videos that already have a transcript unless `--force`. |
| `speakers <video>` | Prints each speaker label with total talk time, segment count and its first ~3 lines (`--lines`). Transcribes first if needed. |
| `mute <video> --speaker LABEL` | Mutes every segment of `LABEL` (plus `--pad` seconds on each side, default 0.15) using the existing transcript and writes `<stem>_muted.<ext>` (or the path given with `--output/-o`). Repeat `--speaker` to mute several people. With `--burn-subs` the subtitles are burned in during the same pass. |
| `run <video-or-folder> --mute-speaker LABEL` | `transcribe` (if needed) + `mute` in one go. `--mute-speaker` is repeatable too; `--output` only with a single video. |

Common options for the transcription commands:

| Option | Default | Notes |
|---|---|---|
| `--model` | `auto` | `large-v3` on CUDA, `medium` on CPU. Any faster-whisper model name works (`small` is a good quick test). |
| `--device` | `auto` | `cuda` if torch sees a GPU, else `cpu`. |
| `--compute-type` | `auto` | `float16` on CUDA, `int8` on CPU. |
| `--language` | `en` | `auto` to let Whisper detect it. |
| `--min-speakers` / `--max-speakers` | auto | Diarization bounds. Unset, pyannote estimates the count; giving the real number (e.g. `--min-speakers 2 --max-speakers 2`) improves accuracy. |
| `--diarization-model` | `pyannote/speaker-diarization-community-1` | Or `pyannote/speaker-diarization-3.1`. |
| `--fill-nearest` | on | Segments overlapping no diarization turn get the nearest speaker. With `--no-fill-nearest` they stay `UNKNOWN` and are never muted. |
| `--rename OLD=NEW` | | Relabel speakers in `subtitles.srt` (repeatable). `transcript.json` keeps the raw labels. |
| `--out-dir` | `out` | Output root. |

`mute`/`run` also take `--pad`, `--burn-subs`, `--output`/`-o` (custom path for the muted video),
`--crf` and `--preset` (x264 settings for the burned version). `--verbose` / `-v` before the command turns on debug logging (shows the ffmpeg commands).

## How muting works

* All segments whose `speaker` matches are collected, padded by `--pad` seconds and merged.
* They become one ffmpeg `volume` filter, `volume=enable='between(t,a,b)+between(t,c,d)+…':volume=0`,
  written to `mute_filter.txt` and loaded from that file (`-/filter:a FILE` on ffmpeg ≥ 7, the older
  `-filter_script:a` on ffmpeg ≤ 6; the expression can get long, and very long lists are split across
  several chained `volume` filters).
* Without `--burn-subs` the video stream is copied (`-c:v copy`) and only the audio is re-encoded to AAC,
  so the muted video is pixel-identical to the input and `subtitles.srt` can be loaded as a
  toggleable sidecar in VLC/mpv/IINA.
* Because the video stream is copied unchanged, the output keeps the input's container: `.mp4`, `.mov`,
  `.m4v` and `.mkv` stay as they are, and `.webm` becomes `.mkv` (WebM cannot hold AAC audio, Matroska
  takes any codec). Pass `--output name.ext` to pick a container yourself.
* With `--burn-subs` the same single ffmpeg pass also runs the `subtitles` filter (needs libass, see
  Setup) and re-encodes the video with x264 (`--crf`, `--preset`). There is no intermediate file.
* Afterwards the input and output durations are compared with `ffprobe` and a warning is logged if they
  differ by more than 0.1 s.
* **Overlapping speech** (two people talking at once) is muted together with the chosen speaker. Accepted
  trade-off; the filter works on time ranges, not on individual voices.

## Performance notes

* Long videos are fine: audio is extracted once to a temporary 16 kHz WAV, models are loaded once per
  batch, and every stage shows a progress bar.
* Measured on an Apple M1 (CPU, `medium`/`int8`) for a 25-minute 720p recording of two people:

  | Stage | Time |
  |---|---|
  | Transcribing | 6.5 min |
  | Aligning words | 1 min |
  | Diarizing speakers | 24 min |
  | Muting (stream copy) | 15 s |
  | Muting + burning subtitles | 2.5 min |

  Diarization is the slow stage on CPU and scales with recording length. On a CUDA GPU with `large-v3`
  the whole pipeline runs several times faster than real time. `--model small` roughly halves the
  transcription time if you only need to identify the speakers.
* Running out of memory? Try `--batch-size 4` or a smaller `--model`.

## Troubleshooting

**`speaker diarization needs a Hugging Face token`** – no `HF_TOKEN` in `.env`/environment and no
`huggingface-cli login`. See Setup §2.

**`401`/`403`, "Cannot access gated repo", or `Could not download 'pyannote/...'`** – the token is valid
but you have not clicked "Agree" on the model page(s) listed in Setup §2 (both pages for the legacy
pipeline). Accepting takes effect immediately; re-run.

**CUDA / cuDNN errors with WhisperX (ctranslate2)** – e.g. `Could not load library libcudnn_ops_infer.so.8`,
`CUDNN_STATUS_NOT_SUPPORTED`, `cublas64_12.dll not found`:

* ctranslate2 4.x needs CUDA 12 + cuDNN 9; torch 2.8 wheels ship their own cuDNN 9, so on Linux the
  usual fix is to make those libraries visible:
  `export LD_LIBRARY_PATH=$(uv run python -c "import nvidia.cublas.lib, nvidia.cudnn.lib, os; print(os.path.dirname(nvidia.cublas.lib.__file__)+':'+os.path.dirname(nvidia.cudnn.lib.__file__))"):$LD_LIBRARY_PATH`
* Mismatched versions: the pinned `torch~=2.8` / `ctranslate2>=4.5` combination in `uv.lock` is known to
  work; do not mix in an older `ctranslate2` (3.x expects cuDNN 8).
* Quick workaround to keep going: `--device cpu --compute-type int8`.

**`float16 compute type not supported on CPU`** – use `--compute-type int8` (or `float32`). `auto` already
does this.

**`your ffmpeg … has no 'subtitles' filter`** – the build lacks libass. On macOS install `ffmpeg-full`
and set `FFMPEG=` in `.env` (see Setup §1).

**Apple Silicon: `--device mps`?** – not supported by ctranslate2; use CPU (the default) or a CUDA machine.

**Warnings like "Model was trained with pyannote.audio 0.0.1 / torch 1.x" or "Lightning automatically
upgraded your loaded checkpoint"** are harmless; they come from the bundled VAD checkpoint.

**`torchcodec is not installed correctly so built-in audio decoding will fail`** (pyannote.audio 4, mostly
on macOS where the torchcodec wheel cannot find ffmpeg's libraries) is also harmless: pyannote only ever
receives audio that this tool has already decoded with ffmpeg, so torchcodec is never used. The tool
silences this particular warning; `-v` does not bring it back.

**Wrong speaker muted** – labels are assigned by order of first speech. Run `speakers`, pick the right label,
or fix `transcript.json` by hand and re-run `mute`.

## Development

```bash
uv run pytest            # unit tests; whisperx and ffmpeg are mocked, nothing is downloaded or encoded
uv run ruff check .      # lint
uv run ruff format .     # format
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the code layout and pull-request expectations, and
[CHANGELOG.md](CHANGELOG.md) for release notes.

Layout (`src/speakermute/`): `ranges.py` (padding/merging/filter expression), `srt.py`,
`speakers.py`, `paths.py` (output layout), `ffmpeg.py` (subprocess wrapper with progress),
`transcribe.py` (WhisperX), `pipeline.py` (transcribe/mute steps), `cli.py` (typer), `errors.py`
(user-facing exceptions), `log.py` (rich console + logger).

## License

[MIT](LICENSE). WhisperX, faster-whisper, ctranslate2 and pyannote.audio are separate projects with
their own licenses; the pyannote models additionally require accepting their terms on Hugging Face.
