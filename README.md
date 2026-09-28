# Voice cue sheet

Local web app that builds a timestamped voiceover with **Deepgram Flux `alexis-en`**.

Each cue is synthesized, fitted into its time window, and mixed into one WAV. Gaps stay silent.

## Run

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env   # add your Deepgram API key
./run.sh
```

Open http://127.0.0.1:8787

Requires `ffmpeg`, `ffprobe`, and `rubberband` on `PATH` (or in `/usr/local/bin`).

## Defaults

| Control | Default |
|---|---|
| Voice | `flux-alexis-en` |
| Speed | `1.5` (0.5–1.5) |
| Expressivity | `2` (−2–2) |
| Duration | 30s |

The generated file is also copied to `~/Downloads/voiceover.wav`.
