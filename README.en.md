# 🎙️ Local Audio & Video Transcription

A Streamlit service that transcribes audio and video **on your own machine**: from a URL
(YouTube, podcasts, RSS episodes) or a local file. Recognition uses local
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) models — audio never leaves
your computer. Long recordings are processed in the background with progress saved to
disk, so multi-hour files can run overnight and resume after a restart.

[Русская версия →](README.md)

## Features

- **URL or file.** YouTube and thousands of other sites via
  [yt-dlp](https://github.com/yt-dlp/yt-dlp) (podcasts, SoundCloud, RSS), file upload,
  or a path to a file on disk. Castbox is supported separately.
- **Existing YouTube subtitles.** If a video has subtitles (manual or auto-generated),
  they are used directly — no audio download, no CPU load. Rolling-caption duplicates
  are collapsed automatically, so the text stays readable.
- **Local Whisper models** `tiny` … `large-v3` (default `small`), CPU `int8`,
  VAD silence filter, automatic language detection.
- **Progress is never lost.** Every chunk is saved to `segments.jsonl`, state to
  `state.json`. A job can be cancelled and resumed later, even after a server restart.
- **Exports** TXT, SRT, VTT with timecodes, plus a full-text preview in the UI.
- **Summary (LLM).** Optional summary and key points via OpenRouter (free models
  included), in Russian or in the original language. Long recordings use map-reduce
  with protection against looping and lost key points; the summary can be generated
  automatically right after transcription.

## Requirements

- **Python 3.10+** (tested with 3.13).
- **ffmpeg** — required for audio conversion. No manual install needed: if `ffmpeg`
  is not in `PATH`, the binary bundled with the `imageio-ffmpeg` package (installed
  with the dependencies) is used.
- Disk space: Whisper models are downloaded on first use (~0.5 GB `tiny`/`base`,
  ~1 GB `small`, ~1.5 GB `medium`, ~3 GB `large-v3`), plus a WAV copy of the recording
  (16 kHz mono ≈ 115 MB per hour).

## Installation

### Windows

```bat
git clone https://github.com/lukyanenkomax/aiscribe.git
cd aiscribe
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
git clone https://github.com/lukyanenkomax/aiscribe.git
cd aiscribe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To use a system ffmpeg instead of the bundled one:

- Windows: `winget install Gyan.FFmpeg`
- macOS: `brew install ffmpeg`
- Debian/Ubuntu: `sudo apt install ffmpeg`

## Running

**Windows:** double-click `start.bat` — it locates Streamlit (in `PATH`, in a per-user
Anaconda/Miniconda install, or via `python -m streamlit`), opens the browser and starts
the service.

**Any OS:**

```bash
streamlit run app.py
```

UI: <http://localhost:8501>

## Usage

1. In the sidebar, pick a Whisper model, compute type (`int8` for CPU), language, and
   toggle VAD.
2. **📺 Use existing subtitles (YouTube)** — fetch available subtitles instead of
   transcribing. If there are none, regular transcription runs automatically.
3. **📝 Summarize after transcription** — build the summary right after a successful
   transcription. Requires an OpenRouter API key
   ([get one](https://openrouter.ai/keys)); the default model is the free
   `openrouter/free`. The key is kept in session memory only and never written to disk.
4. Add a job: the “URL”, “Local file”, or “File path” tab, then
   **▶️ Start transcription**.
5. The job card shows the channel, duration, status, progress and estimated time left.
   Download TXT/SRT/VTT, and the summary as Markdown.

## How it works

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI, job cards, auto-refresh via `st.fragment` |
| `downloader.py` | URL download (yt-dlp), Castbox, existing subtitles (VTT/SRT), conversion to 16 kHz WAV |
| `transcriber.py` | faster-whisper, chunked processing with checkpoints, TXT/SRT/VTT export |
| `jobs.py` | job queue, single background worker, state in `jobs/<id>/` |
| `summarizer.py` | summary via OpenRouter (map-reduce, retries, loop protection) |

All data stays local in `jobs/<id>/`: `state.json` (state), `audio.wav`,
`segments.jsonl` (recognized segments), `transcript.txt/.srt/.vtt`, `summary.md`.
The “Delete” button wipes the whole job folder.

## Troubleshooting

- **First run is slow** — the Whisper model is being downloaded from Hugging Face.
- **`streamlit` is not recognized** — activate the virtual environment
  (`.venv\Scripts\activate`) or use `start.bat`.
- **`DLL load failed` / KMP errors** — the service already sets
  `KMP_DUPLICATE_LIB_OK=TRUE`; do the same for manual runs.
- **Subtitles failed to download (HTTP 429)** — YouTube rate-limits frequent requests;
  the service automatically falls back to regular transcription.
- **Changes to `jobs.py`/`summarizer.py` are not applied** — Streamlit caches modules;
  restart the service.
- **No GPU** — everything targets CPU; `small` + `int8` is a reasonable balance, use
  `base`/`tiny` for multi-hour recordings.

## Limitations

- CPU-only: roughly 0.3–1× realtime depending on the model and hardware.
- Free OpenRouter models may be overloaded and have a small output limit; the service
  retries and reports an error if the model never responds.
- YouTube auto-subtitles contain recognition errors in names and terms — disable
  existing subtitles and run Whisper for critical recordings.

## License

MIT — see [LICENSE](LICENSE).
