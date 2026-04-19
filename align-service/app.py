"""FastAPI entry point for the AlphaTab score–audio aligner."""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import registry  # noqa: F401 — triggers adapter registration
from pipeline.aligner_dtw import DtwCosineAligner  # noqa: F401
from pipeline.audio_to_chroma import audio_to_chroma
from pipeline.bar_map import bars_to_audio, downsample_path, warp_path_to_seconds
from pipeline.score_loader import load_score
from pipeline.score_to_chroma import score_to_chroma

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("align")

app = FastAPI(
    title="AlphaTab Score–Audio Aligner",
    version="0.1.0",
    description="Score-informed DTW alignment for the AlphaTab practice app.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOW_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)


class Meta(BaseModel):
    score_duration_sec: float
    audio_duration_sec: float
    sample_rate: int
    hop_length: int
    tempo_bpm: Optional[float] = None
    aligner: str
    score_format: str


class BarPoint(BaseModel):
    index: int
    score_sec: float
    audio_sec: float


class AlignResponse(BaseModel):
    ok: bool
    meta: Meta
    bars: list[BarPoint]
    warp_path: list[tuple[float, float]]


SCORE_SUFFIXES = (
    ".xml", ".musicxml", ".mxl", ".mid", ".midi",
    ".gp", ".gp3", ".gp4", ".gp5", ".gpx", ".gp7",
)
AUDIO_SUFFIXES = (".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus")


def _suffix_of(upload: UploadFile, fallback_exts: tuple[str, ...]) -> str:
    name = (upload.filename or "").lower()
    _, ext = os.path.splitext(name)
    if ext:
        return ext
    # Fall back to the first allowed extension so tempfile naming works.
    return fallback_exts[0]


async def _save_upload(upload: UploadFile, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            while chunk := await upload.read(1024 * 1024):
                f.write(chunk)
    finally:
        await upload.close()
    return path


@app.get("/")
async def root():
    return {
        "service": "alphatab-align",
        "endpoints": ["/healthz", "/align", "/align-pdf"],
        "aligners": sorted(registry.ALIGNERS),
        "transcribers": sorted(registry.TRANSCRIBERS),
        "omr": sorted(registry.OMR_ADAPTERS),
        "score_extensions": sorted(registry.SCORE_LOADERS_BY_EXT),
    }


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/align", response_model=AlignResponse)
async def align(
    score: UploadFile = File(..., description="Score file (MusicXML/MIDI/GP)."),
    audio: UploadFile = File(..., description="Audio recording."),
    subseq: bool = Form(False, description="Set when audio has intro/outro."),
    hop_length: int = Form(512, ge=64, le=8192),
    sample_rate: int = Form(22050, ge=8000, le=48000),
    aligner: str = Form("dtw-cosine"),
):
    score_suffix = _suffix_of(score, SCORE_SUFFIXES)
    audio_suffix = _suffix_of(audio, AUDIO_SUFFIXES)

    if score_suffix not in SCORE_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported score extension '{score_suffix}'. "
                   f"Accepted: {', '.join(SCORE_SUFFIXES)}.",
        )

    score_path = await _save_upload(score, score_suffix)
    audio_path = await _save_upload(audio, audio_suffix)

    try:
        return _align_paths(
            score_path=score_path,
            audio_path=audio_path,
            subseq=subseq,
            hop_length=hop_length,
            sample_rate=sample_rate,
            aligner=aligner,
        )
    finally:
        for p in (score_path, audio_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _align_paths(
    *,
    score_path: str,
    audio_path: str,
    subseq: bool,
    hop_length: int,
    sample_rate: int,
    aligner: str,
) -> AlignResponse:
    aligner_cls = registry.pick_aligner(aligner)
    if aligner_cls is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown aligner '{aligner}'. Available: {sorted(registry.ALIGNERS)}.",
        )

    log.info("Loading score %s", score_path)
    bundle = load_score(score_path)

    log.info(
        "Score: %d notes, %d bars, duration %.1fs, tempo %s, fmt %s",
        len(bundle.notes), len(bundle.bars), bundle.duration_sec,
        bundle.tempo_bpm, bundle.source_format,
    )

    score_chroma = score_to_chroma(
        bundle.notes, bundle.duration_sec,
        sample_rate=sample_rate, hop_length=hop_length,
    )
    log.info("Score chroma: %s", score_chroma.shape)

    ac = audio_to_chroma(audio_path, sample_rate=sample_rate, hop_length=hop_length)
    log.info("Audio chroma: %s (%.1fs)", ac.chroma.shape, ac.duration_sec)

    wp = aligner_cls().align(score_chroma, ac.chroma, subseq=subseq)
    log.info("DTW path length: %d", len(wp))

    path_sec = warp_path_to_seconds(wp, sample_rate, hop_length)
    path_small = downsample_path(path_sec, step_sec=0.05)
    bars = bars_to_audio(bundle.bars, wp, sample_rate, hop_length)

    meta = Meta(
        score_duration_sec=round(bundle.duration_sec, 3),
        audio_duration_sec=round(ac.duration_sec, 3),
        sample_rate=sample_rate,
        hop_length=hop_length,
        tempo_bpm=bundle.tempo_bpm,
        aligner=aligner,
        score_format=bundle.source_format,
    )
    return AlignResponse(
        ok=True,
        meta=meta,
        bars=[BarPoint(**b) for b in bars],
        warp_path=path_small,
    )


@app.post("/align-pdf")
async def align_pdf(
    score: UploadFile = File(...),
    audio: UploadFile = File(...),
    omr: Optional[str] = Form(None),
    subseq: bool = Form(False),
    hop_length: int = Form(512, ge=64, le=8192),
    sample_rate: int = Form(22050, ge=8000, le=48000),
    aligner: str = Form("dtw-cosine"),
):
    adapter_cls = registry.pick_omr(omr)
    if adapter_cls is None:
        raise HTTPException(
            status_code=501,
            detail=(
                "No OMR adapter registered. Add one (Audiveris / oemer) in "
                "pipeline/omr.py and ship it in a Dockerfile.omr build, then "
                "redeploy. See align-service/README.md → Extending."
            ),
        )

    pdf_path = await _save_upload(score, ".pdf")
    audio_path = await _save_upload(audio, _suffix_of(audio, AUDIO_SUFFIXES))
    musicxml_path = pdf_path + ".musicxml"

    try:
        adapter_cls().pdf_to_musicxml(pdf_path, musicxml_path)
        return _align_paths(
            score_path=musicxml_path,
            audio_path=audio_path,
            subseq=subseq,
            hop_length=hop_length,
            sample_rate=sample_rate,
            aligner=aligner,
        )
    finally:
        for p in (pdf_path, audio_path, musicxml_path):
            try:
                os.unlink(p)
            except OSError:
                pass
