#!/usr/bin/env python3
"""Voice cue sheet — Deepgram Flux Alexis, timestamped cues."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)
STATIC = ROOT / "static"

DEEPGRAM_SPEAK = "https://api.deepgram.com/v2/speak"
VOICE = "flux-alexis-en"
SAMPLE_RATE = 24000
FFMPEG = shutil.which("ffmpeg") or "/usr/local/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/usr/local/bin/ffprobe"
RUBBERBAND = shutil.which("rubberband") or "/usr/local/bin/rubberband"

DEFAULT_CUES = [
    {
        "start": 0.0,
        "end": 4.0,
        "text": "Here is how to access the official app in just a few steps.",
    },
    {
        "start": 5.0,
        "end": 8.0,
        "text": "First, apply as a buyer or media on the website.",
    },
    {
        "start": 9.0,
        "end": 13.0,
        "text": "Check your inbox or spam folder for the confirmation email and activate your account.",
    },
    {
        "start": 14.0,
        "end": 18.0,
        "text": "After receiving the confirmation email, activate your account, download the app and sign in.",
    },
    {
        "start": 19.0,
        "end": 25.0,
        "text": "Explore the program and reserve your spot at the events.",
    },
    {
        "start": 26.0,
        "end": 30.0,
        "text": "Download the app today on the App Store or Google Play. See you there!",
    },
]


class Cue(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1)


class GenerateRequest(BaseModel):
    api_key: str = ""
    speed: float = Field(default=1.5, ge=0.5, le=1.5)
    expressivity: int = Field(default=2, ge=-2, le=2)
    cues: list[Cue]


app = FastAPI(title="Voice cue sheet")


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:] or r.stdout[-2000:] or "command failed")
    return r


def probe_duration(path: Path) -> float:
    r = run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ]
    )
    return float(r.stdout.strip())


def speak_clip(api_key: str, text: str, speed: float, dest: Path, expressivity: int = 2) -> None:
    url = (
        f"{DEEPGRAM_SPEAK}?model={VOICE}"
        f"&encoding=linear16&container=wav&sample_rate={SAMPLE_RATE}"
        f"&speed={speed}&expressivity={int(expressivity)}"
    )
    with httpx.Client(timeout=60.0) as client:
        r = client.post(
            url,
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
            },
            json={"text": text},
        )
    if r.status_code != 200:
        try:
            msg = r.json().get("err_msg") or r.text
        except Exception:
            msg = r.text
        raise HTTPException(status_code=502, detail=f"Deepgram: {msg}")
    dest.write_bytes(r.content)


def trim_silence(src: Path, dest: Path) -> None:
    run(
        [
            FFMPEG,
            "-y",
            "-i",
            str(src),
            "-af",
            "silenceremove=start_periods=1:start_silence=0.03:start_threshold=-38dB:detection=peak,"
            "areverse,silenceremove=start_periods=1:start_silence=0.05:start_threshold=-38dB:detection=peak,areverse",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(dest),
        ]
    )


def fit_to_slot(src: Path, dest: Path, slot: float) -> float:
    src_dur = probe_duration(src)
    if src_dur <= 0.05:
        shutil.copy(src, dest)
        return src_dur
    target = slot if src_dur > slot else src_dur
    r = subprocess.run(
        [RUBBERBAND, "--duration", f"{target:.4f}", "-3", str(src), str(dest)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if r.returncode != 0:
        tempo = max(src_dur / target, 0.5)
        run(
            [
                FFMPEG,
                "-y",
                "-i",
                str(src),
                "-filter:a",
                f"rubberband=tempo={tempo:.5f}",
                "-ar",
                "48000",
                str(dest),
            ]
        )
    return probe_duration(dest)


def mix_timeline(clips: list[tuple[Path, float, float]], total: float, dest: Path) -> None:
    if not clips:
        raise HTTPException(status_code=400, detail="No clips to mix")
    inputs: list[str] = []
    filters: list[str] = []
    for i, (path, start, dur) in enumerate(clips):
        inputs += ["-i", str(path)]
        delay = int(round(start * 1000))
        fade = min(0.016, max(dur / 12, 0.008))
        filters.append(
            f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
            f"afade=t=in:st=0:d={fade:.3f},afade=t=out:st={max(0, dur - fade):.3f}:d={fade:.3f},"
            f"adelay={delay}|{delay},apad=whole_dur={total:.3f}[a{i}]"
        )
    n = len(clips)
    mix_ins = "".join(f"[a{i}]" for i in range(n))
    filters.append(
        f"{mix_ins}amix=inputs={n}:duration=first:dropout_transition=0:normalize=0[mix]"
    )
    filters.append(
        "[mix]loudnorm=I=-16:TP=-1.5:LRA=8,alimiter=limit=0.96,"
        f"apad=whole_dur={total:.3f}[out]"
    )
    run(
        [
            FFMPEG,
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-t",
            f"{total:.3f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(dest),
        ]
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/api/defaults")
def defaults() -> dict:
    return {
        "voice": VOICE,
        "has_env_key": bool(os.environ.get("DEEPGRAM_API_KEY")),
        "cues": DEFAULT_CUES,
        "speed": 1.5,
        "expressivity": 2,
        "duration": 30.0,
    }


@app.post("/api/generate")
def generate(req: GenerateRequest) -> FileResponse:
    api_key = (req.api_key or os.environ.get("DEEPGRAM_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing Deepgram API key")
    if not req.cues:
        raise HTTPException(status_code=400, detail="Add at least one cue")

    cues = sorted(req.cues, key=lambda c: c.start)
    for c in cues:
        if c.end <= c.start:
            raise HTTPException(status_code=400, detail=f"Bad window {c.start}–{c.end}")

    for bin_path, label in ((FFMPEG, "ffmpeg"), (FFPROBE, "ffprobe"), (RUBBERBAND, "rubberband")):
        if not Path(bin_path).exists():
            raise HTTPException(status_code=500, detail=f"Missing {label} at {bin_path}")

    total = max(c.end for c in cues)
    job = uuid.uuid4().hex[:8]
    work = Path(tempfile.mkdtemp(prefix=f"vo_{job}_"))
    placed: list[tuple[Path, float, float]] = []
    report = []

    try:
        for i, cue in enumerate(cues, start=1):
            raw = work / f"{i:02d}_raw.wav"
            trimmed = work / f"{i:02d}_trim.wav"
            fit = work / f"{i:02d}_fit.wav"
            slot = cue.end - cue.start
            speak_clip(api_key, cue.text, req.speed, raw, req.expressivity)
            trim_silence(raw, trimmed)
            actual = fit_to_slot(trimmed, fit, slot)
            placed.append((fit, cue.start, actual))
            report.append(
                {
                    "index": i,
                    "start": cue.start,
                    "end": cue.end,
                    "slot": round(slot, 3),
                    "speech": round(actual, 3),
                }
            )

        wav = OUTPUT / f"voiceover_{job}.wav"
        mix_timeline(placed, total, wav)
        (OUTPUT / f"voiceover_{job}.json").write_text(
            json.dumps(
                {
                    "voice": VOICE,
                    "speed": req.speed,
                    "expressivity": req.expressivity,
                    "duration": total,
                    "cues": report,
                },
                indent=2,
            )
        )
        downloads = Path.home() / "Downloads" / "voiceover.wav"
        shutil.copy(wav, downloads)
        return FileResponse(
            wav,
            media_type="audio/wav",
            filename="voiceover.wav",
            headers={"X-VO-Job": job, "X-VO-Duration": f"{total:.3f}"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8787, reload=False)
