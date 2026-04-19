---
title: AlphaTab Score–Audio Aligner
emoji: 🎸
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# AlphaTab Score–Audio Aligner

Score-informed alignment service for the AlphaTab practice app.

Takes a score (MusicXML / MIDI / Guitar Pro) and a recording (MP3/WAV/OGG/FLAC),
runs **dynamic time warping between their chroma features**, and returns a
warping path plus per-bar (`score_seconds → audio_seconds`) mapping. The
frontend uses that mapping to synchronise the playback cursor to the recording
perfectly — no manual offset tuning required.

PDF support is pluggable (OMR adapters register themselves via the registry);
future **TabCNN / MT3 / YourMT3+ transcribers** plug into the same registry.

## HTTP API

### `POST /align`

Form fields:

| field          | required | notes                                                        |
|----------------|----------|--------------------------------------------------------------|
| `score`        | yes      | MusicXML / `.mxl` / `.mid` / `.gp` / `.gp5` / `.gpx` / `.gp7` |
| `audio`        | yes      | MP3 / WAV / OGG / FLAC / M4A                                 |
| `subseq`       | no       | `true` if the recording is longer than the score (intro/outro). Default `false`. |
| `hop_length`   | no       | Chroma hop length in samples. Default `512`.                 |
| `sample_rate`  | no       | Audio sample rate. Default `22050`.                          |

Returns JSON:

```json
{
  "ok": true,
  "meta": {
    "score_duration_sec": 120.0,
    "audio_duration_sec": 123.4,
    "sample_rate": 22050,
    "hop_length": 512,
    "tempo_bpm": 120.0,
    "aligner": "dtw-cosine"
  },
  "bars": [
    {"index": 1, "score_sec": 0.000, "audio_sec": 0.082},
    {"index": 2, "score_sec": 2.000, "audio_sec": 2.051}
  ],
  "warp_path": [[0.0, 0.08], [0.023, 0.10], ...]
}
```

### `POST /align-pdf`

Same shape but `score` is a PDF. Requires an OMR adapter to be registered
(not bundled by default — see *Extending* below). Returns `501` otherwise.

### `GET /healthz`

Liveness probe.

## Local run

```bash
docker build -t alphatab-align .
docker run --rm -p 7860:7860 alphatab-align
# then POST to http://localhost:7860/align
```

## Deploying to Hugging Face Spaces

1. Create a new Space, SDK = **Docker**, link it to this repo's `align-service/`
   folder (or push it as its own Space — see `.github/workflows/sync-hf-space.yml`
   in the repo root which does this automatically).
2. Set the `HF_TOKEN` GitHub secret with write access to the Space.
3. Push to `main` — the workflow mirrors `align-service/` into the HF Space.

## Extending

Everything pluggable lives in `pipeline/registry.py`. Register a new
transcriber (TabCNN / MT3), aligner (beat-aware DTW, sync-toolbox), or score
loader (Audiveris OMR, oemer) with a decorator:

```python
from pipeline.registry import register_transcriber
from pipeline.base import Transcriber

@register_transcriber("tabcnn")
class TabCNNTranscriber(Transcriber):
    def transcribe(self, audio_path: str) -> list[dict]:
        ...
```

Then select it via `?transcriber=tabcnn` (not wired into `/align` yet, but the
plumbing is in place).
