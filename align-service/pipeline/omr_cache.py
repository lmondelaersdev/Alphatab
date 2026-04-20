"""Filesystem cache for OMR results keyed by PDF content hash.

OMR is the slowest step in the pipeline by far — re-running the same PDF
should not pay that cost twice. We SHA-256 the PDF bytes and store the
resulting MusicXML under ``<hash>.musicxml`` in a cache directory.

Cache location priority:
  1. ``$OMR_CACHE_DIR`` if set and writable.
  2. ``/data/omr-cache`` — HF Spaces persistent storage when enabled.
  3. ``/tmp/omr-cache`` — container-lifetime only, still survives retries.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from typing import Optional

log = logging.getLogger("align.omr.cache")


def _pick_cache_root() -> str:
    candidates = []
    explicit = os.environ.get("OMR_CACHE_DIR")
    if explicit:
        candidates.append(explicit)
    candidates.extend(["/data/omr-cache", "/tmp/omr-cache"])

    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            probe = os.path.join(path, ".writable")
            with open(probe, "w") as f:
                f.write("")
            os.unlink(probe)
            return path
        except OSError:
            continue
    # Last resort — even /tmp/omr-cache should have worked above.
    fallback = "/tmp/omr-cache"
    os.makedirs(fallback, exist_ok=True)
    return fallback


CACHE_ROOT = _pick_cache_root()
log.info("OMR cache root: %s", CACHE_ROOT)


def hash_pdf(pdf_path: str) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lookup(pdf_hash: str) -> Optional[str]:
    path = os.path.join(CACHE_ROOT, pdf_hash + ".musicxml")
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    return None


def store(pdf_hash: str, xml_path: str) -> Optional[str]:
    dst = os.path.join(CACHE_ROOT, pdf_hash + ".musicxml")
    try:
        shutil.copyfile(xml_path, dst)
        log.info("OMR cache: stored %s (%d bytes)", pdf_hash[:12],
                 os.path.getsize(dst))
        return dst
    except OSError as e:
        log.warning("OMR cache: store failed for %s: %s", pdf_hash[:12], e)
        return None
