"""FastAPI entry point for the AlphaTab score–audio aligner."""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import uuid
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import omr_cache, registry  # noqa: F401 — triggers adapter registration
from pipeline.aligner_dtw import DtwCosineAligner  # noqa: F401
from pipeline.audio_to_chroma import audio_to_chroma
from pipeline.bar_map import bars_to_audio, downsample_path, warp_path_to_seconds
from pipeline.score_loader import load_score
from pipeline.score_to_chroma import score_to_chroma

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("align")

app = FastAPI(
    title="AlphaTab Score–Audio Aligner",
    version="0.2.0",
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
    score_xml: Optional[str] = None  # populated by /align-pdf (OMR output)


class JobStartResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    state: str  # pending | running | done | error
    message: str
    elapsed_sec: float
    result: Optional[AlignResponse] = None
    error: Optional[str] = None


SCORE_SUFFIXES = (
    ".xml", ".musicxml", ".mxl", ".mid", ".midi",
    ".gp", ".gp3", ".gp4", ".gp5", ".gpx", ".gp7",
)
AUDIO_SUFFIXES = (".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus")

# --- In-memory job store -------------------------------------------------
# Good enough for a single-worker HF Space. Jobs are purged 30 min after
# completion so the dict doesn't grow unbounded across a long-running
# container.
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SEC = 30 * 60


def _new_job() -> str:
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "state": "pending",
            "message": "queued",
            "started_at": time.time(),
            "finished_at": None,
            "result": None,
            "error": None,
        }
    return job_id


def _update_job(job_id: str, **kv) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kv)


def _get_job(job_id: str) -> Optional[dict]:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def _gc_jobs() -> None:
    now = time.time()
    with JOBS_LOCK:
        stale = [
            jid for jid, j in JOBS.items()
            if j["finished_at"] and now - j["finished_at"] > JOB_TTL_SEC
        ]
        for jid in stale:
            JOBS.pop(jid, None)


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
        "endpoints": [
            "/healthz",
            "/align", "/align-pdf",                # sync
            "/align-async", "/align-pdf-async",    # submit → job_id
            "/align-jobs/{job_id}",                # poll
        ],
        "aligners": sorted(registry.ALIGNERS),
        "transcribers": sorted(registry.TRANSCRIBERS),
        "omr": sorted(registry.OMR_ADAPTERS),
        "score_extensions": sorted(registry.SCORE_LOADERS_BY_EXT),
    }


@app.get("/healthz")
async def healthz():
    return {"ok": True}


# --- Sync endpoints (kept for simple clients and internal reuse) --------

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
    score_xml: Optional[str] = None,
    progress_cb=None,
) -> AlignResponse:
    aligner_cls = registry.pick_aligner(aligner)
    if aligner_cls is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown aligner '{aligner}'. Available: {sorted(registry.ALIGNERS)}.",
        )

    if progress_cb: progress_cb("Loading score")
    log.info("Loading score %s", score_path)
    bundle = load_score(score_path)

    log.info(
        "Score: %d notes, %d bars, duration %.1fs, tempo %s, fmt %s",
        len(bundle.notes), len(bundle.bars), bundle.duration_sec,
        bundle.tempo_bpm, bundle.source_format,
    )

    if progress_cb: progress_cb("Computing score chroma")
    score_chroma = score_to_chroma(
        bundle.notes, bundle.duration_sec,
        sample_rate=sample_rate, hop_length=hop_length,
    )
    log.info("Score chroma: %s", score_chroma.shape)

    if progress_cb: progress_cb("Computing audio chroma")
    ac = audio_to_chroma(audio_path, sample_rate=sample_rate, hop_length=hop_length)
    log.info("Audio chroma: %s (%.1fs)", ac.chroma.shape, ac.duration_sec)

    if progress_cb: progress_cb("Running DTW")
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
        score_xml=score_xml,
    )


@app.post("/align-pdf", response_model=AlignResponse)
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
            detail="No OMR adapter registered.",
        )

    pdf_path = await _save_upload(score, ".pdf")
    audio_path = await _save_upload(audio, _suffix_of(audio, AUDIO_SUFFIXES))
    owned_musicxml = pdf_path + ".musicxml"  # temp copy — safe to unlink
    to_unlink = [pdf_path, audio_path]

    try:
        pdf_hash = omr_cache.hash_pdf(pdf_path)
        cached = omr_cache.lookup(pdf_hash)
        if cached:
            log.info("OMR cache HIT for %s", pdf_hash[:12])
            musicxml_path = cached  # cache file — do not unlink
        else:
            adapter_cls().pdf_to_musicxml(pdf_path, owned_musicxml)
            omr_cache.store(pdf_hash, owned_musicxml)
            musicxml_path = owned_musicxml
            to_unlink.append(owned_musicxml)

        with open(musicxml_path, "r", encoding="utf-8", errors="replace") as f:
            score_xml_text = f.read()
        return _align_paths(
            score_path=musicxml_path,
            audio_path=audio_path,
            subseq=subseq,
            hop_length=hop_length,
            sample_rate=sample_rate,
            aligner=aligner,
            score_xml=score_xml_text,
        )
    finally:
        for p in to_unlink:
            try:
                os.unlink(p)
            except OSError:
                pass


# --- Async endpoints (upload → job_id → poll) ---------------------------

def _worker_align(
    job_id: str,
    *,
    score_path: str,
    audio_path: str,
    subseq: bool,
    hop_length: int,
    sample_rate: int,
    aligner: str,
) -> None:
    def progress(msg: str) -> None:
        _update_job(job_id, message=msg)
        log.info("[job %s] %s", job_id[:8], msg)

    try:
        _update_job(job_id, state="running", message="starting")
        resp = _align_paths(
            score_path=score_path,
            audio_path=audio_path,
            subseq=subseq,
            hop_length=hop_length,
            sample_rate=sample_rate,
            aligner=aligner,
            progress_cb=progress,
        )
        _update_job(
            job_id,
            state="done",
            message="done",
            result=resp.model_dump(),
            finished_at=time.time(),
        )
    except Exception as e:
        log.exception("align job %s failed", job_id)
        _update_job(
            job_id,
            state="error",
            error=str(e),
            message=f"error: {type(e).__name__}",
            finished_at=time.time(),
        )
    finally:
        for p in (score_path, audio_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _worker_align_pdf(
    job_id: str,
    *,
    omr_name: Optional[str],
    pdf_path: str,
    audio_path: str,
    subseq: bool,
    hop_length: int,
    sample_rate: int,
    aligner: str,
) -> None:
    def progress(msg: str) -> None:
        _update_job(job_id, message=msg)
        log.info("[job %s] %s", job_id[:8], msg)

    adapter_cls = registry.pick_omr(omr_name)
    owned_musicxml = pdf_path + ".musicxml"  # temp — safe to unlink
    to_unlink = [pdf_path, audio_path]

    try:
        _update_job(job_id, state="running", message="starting OMR")

        if adapter_cls is None:
            raise RuntimeError("No OMR adapter registered.")

        progress("Hashing PDF for OMR cache lookup")
        pdf_hash = omr_cache.hash_pdf(pdf_path)
        cached = omr_cache.lookup(pdf_hash)

        if cached:
            progress(f"OMR cache HIT ({pdf_hash[:8]}) — skipping inference")
            musicxml_path = cached  # persistent cache path — do NOT unlink
        else:
            progress("Running OMR on PDF (this can take several minutes per page)")
            try:
                adapter_cls().pdf_to_musicxml(pdf_path, owned_musicxml, progress_cb=progress)
            except TypeError:
                # Adapter without progress_cb keyword — call the old signature.
                adapter_cls().pdf_to_musicxml(pdf_path, owned_musicxml)
            omr_cache.store(pdf_hash, owned_musicxml)
            musicxml_path = owned_musicxml
            to_unlink.append(owned_musicxml)

        progress("Reading OMR MusicXML")
        with open(musicxml_path, "r", encoding="utf-8", errors="replace") as f:
            score_xml_text = f.read()

        resp = _align_paths(
            score_path=musicxml_path,
            audio_path=audio_path,
            subseq=subseq,
            hop_length=hop_length,
            sample_rate=sample_rate,
            aligner=aligner,
            score_xml=score_xml_text,
            progress_cb=progress,
        )
        _update_job(
            job_id,
            state="done",
            message="done",
            result=resp.model_dump(),
            finished_at=time.time(),
        )
    except Exception as e:
        log.exception("align-pdf job %s failed", job_id)
        _update_job(
            job_id,
            state="error",
            error=str(e),
            message=f"error: {type(e).__name__}",
            finished_at=time.time(),
        )
    finally:
        for p in to_unlink:
            try:
                os.unlink(p)
            except OSError:
                pass


@app.post("/align-async", response_model=JobStartResponse)
async def align_async(
    background_tasks: BackgroundTasks,
    score: UploadFile = File(...),
    audio: UploadFile = File(...),
    subseq: bool = Form(False),
    hop_length: int = Form(512, ge=64, le=8192),
    sample_rate: int = Form(22050, ge=8000, le=48000),
    aligner: str = Form("dtw-cosine"),
):
    score_suffix = _suffix_of(score, SCORE_SUFFIXES)
    audio_suffix = _suffix_of(audio, AUDIO_SUFFIXES)
    if score_suffix not in SCORE_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Unsupported score extension '{score_suffix}'.")

    score_path = await _save_upload(score, score_suffix)
    audio_path = await _save_upload(audio, audio_suffix)
    job_id = _new_job()
    _gc_jobs()

    t = threading.Thread(
        target=_worker_align,
        name=f"align-{job_id[:8]}",
        kwargs=dict(
            job_id=job_id,
            score_path=score_path,
            audio_path=audio_path,
            subseq=subseq,
            hop_length=hop_length,
            sample_rate=sample_rate,
            aligner=aligner,
        ),
        daemon=True,
    )
    t.start()
    return JobStartResponse(job_id=job_id)


@app.post("/align-pdf-async", response_model=JobStartResponse)
async def align_pdf_async(
    score: UploadFile = File(...),
    audio: UploadFile = File(...),
    omr: Optional[str] = Form(None),
    subseq: bool = Form(False),
    hop_length: int = Form(512, ge=64, le=8192),
    sample_rate: int = Form(22050, ge=8000, le=48000),
    aligner: str = Form("dtw-cosine"),
):
    pdf_path = await _save_upload(score, ".pdf")
    audio_path = await _save_upload(audio, _suffix_of(audio, AUDIO_SUFFIXES))
    job_id = _new_job()
    _gc_jobs()

    t = threading.Thread(
        target=_worker_align_pdf,
        name=f"align-pdf-{job_id[:8]}",
        kwargs=dict(
            job_id=job_id,
            omr_name=omr,
            pdf_path=pdf_path,
            audio_path=audio_path,
            subseq=subseq,
            hop_length=hop_length,
            sample_rate=sample_rate,
            aligner=aligner,
        ),
        daemon=True,
    )
    t.start()
    return JobStartResponse(job_id=job_id)


@app.get("/align-jobs/{job_id}", response_model=JobStatusResponse)
async def align_job_status(job_id: str):
    j = _get_job(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job.")
    elapsed = (j["finished_at"] or time.time()) - j["started_at"]
    return JobStatusResponse(
        job_id=job_id,
        state=j["state"],
        message=j["message"],
        elapsed_sec=round(elapsed, 1),
        result=AlignResponse(**j["result"]) if j.get("result") else None,
        error=j.get("error"),
    )
