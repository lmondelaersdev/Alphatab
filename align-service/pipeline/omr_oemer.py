"""PDF → MusicXML via the `oemer` end-to-end OMR model.

oemer (https://github.com/BreezeWhite/oemer) is a Python/ONNX model that
takes a rendered page image and produces a MusicXML file. We convert each
PDF page to PNG with PyMuPDF, run oemer per page, then concatenate the
per-page MusicXML files into a single score via music21 so the alignment
path covers the whole piece.

Execution modes (picked automatically):

* **in-process**: import ``oemer.ete.extract`` in the worker thread after
  monkey-patching ``onnxruntime.InferenceSession`` to memoise sessions by
  model path. This keeps the ~200MB of ONNX weights warm across pages —
  on a 3-page PDF, pages 2+ avoid the multi-minute model-load tax.
* **subprocess** (fallback): shell out to the ``oemer`` CLI. Used only if
  the in-process path fails on first try. Slower because each invocation
  re-imports Python and reloads all ONNX weights.
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import shutil
import subprocess
import tempfile
import threading

from .base import OmrAdapter
from .registry import register_omr

log = logging.getLogger("align.omr")

# oemer is not re-entrant, and we only run one job at a time on the free
# HF Space anyway. This lock serialises both in-process inference calls
# and the one-shot ORT patch installation.
_OEMER_LOCK = threading.Lock()
_ORT_PATCHED = False
_INPROCESS_OK: bool | None = None  # None=untested, True=works, False=disabled


def _patch_onnxruntime_cache() -> None:
    """Cache ``onnxruntime.InferenceSession`` instances by (path, providers).

    oemer's ``inference()`` constructs a fresh session on every call, which
    means a 3-page PDF loads the same ~200MB of ONNX weights three times.
    By memoising at the ORT layer we reuse sessions across pages without
    touching oemer's internals. ORT sessions are documented thread-safe
    for ``run()`` so sharing is fine.
    """
    global _ORT_PATCHED
    if _ORT_PATCHED:
        return
    import onnxruntime as ort

    orig_cls = ort.InferenceSession
    cache: dict = {}
    cache_lock = threading.Lock()

    def cached_session(path_or_bytes, *args, **kwargs):
        if not isinstance(path_or_bytes, str):
            return orig_cls(path_or_bytes, *args, **kwargs)
        key = (
            os.path.realpath(path_or_bytes),
            tuple(repr(p) for p in (kwargs.get("providers") or ())),
        )
        with cache_lock:
            sess = cache.get(key)
            if sess is None:
                log.info("ORT: loading new session for %s", key[0])
                sess = orig_cls(path_or_bytes, *args, **kwargs)
                cache[key] = sess
            else:
                log.info("ORT: reusing cached session for %s", key[0])
        return sess

    ort.InferenceSession = cached_session  # type: ignore[assignment]
    _ORT_PATCHED = True
    log.info("Patched onnxruntime.InferenceSession for cross-page reuse")


def _find_produced_musicxml(work_dir: str, png_path: str) -> str | None:
    stem = os.path.splitext(os.path.basename(png_path))[0]
    candidates = [
        os.path.join(work_dir, stem + ".musicxml"),
        os.path.join(work_dir, stem + ".xml"),
    ]
    for root, _, files in os.walk(work_dir):
        for f in files:
            if f.lower().endswith((".musicxml", ".xml")) and stem in f:
                candidates.append(os.path.join(root, f))
    return next((p for p in candidates if os.path.isfile(p)), None)


def _run_oemer_inprocess(png_path: str, work_dir: str) -> str:
    """Run oemer's end-to-end pipeline without forking a subprocess."""
    _patch_onnxruntime_cache()
    from oemer import ete

    ns = argparse.Namespace(
        img_path=png_path,
        output_path=work_dir,
        use_tf=False,
        save_cache=False,
        without_deskew=False,
    )
    # oemer may write intermediate artefacts to the current working
    # directory, so pin cwd to work_dir for tidiness. Restore afterwards
    # so we don't surprise unrelated code in the same process.
    prev_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        produced = ete.extract(ns)
    finally:
        try:
            os.chdir(prev_cwd)
        except OSError:
            pass

    if not (isinstance(produced, str) and os.path.isfile(produced)):
        produced = _find_produced_musicxml(work_dir, png_path)
    if not produced:
        raise RuntimeError(
            f"oemer completed in-process but produced no MusicXML for {png_path}"
        )
    return produced


def _run_oemer_subprocess(png_path: str, work_dir: str) -> str:
    """Fallback: invoke the oemer CLI. Reloads ONNX weights on every call."""
    cmd = ["oemer", "--output-path", work_dir, png_path]
    log.info("Running (subprocess): %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=work_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=1200,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "oemer CLI not found. Install with `pip install oemer` and "
            "ensure its ONNX weights are downloaded."
        ) from e

    if proc.returncode != 0:
        raise RuntimeError(
            "oemer failed (exit %d): %s" % (proc.returncode, proc.stderr[-800:])
        )

    produced = _find_produced_musicxml(work_dir, png_path)
    if not produced:
        raise RuntimeError(
            "oemer completed but produced no MusicXML for %s. stdout tail: %s"
            % (png_path, proc.stdout[-400:])
        )
    return produced


def _run_oemer(png_path: str, work_dir: str) -> str:
    """Run oemer on one page PNG. Picks in-process or subprocess automatically."""
    global _INPROCESS_OK

    # Env override: OEMER_MODE=subprocess to force the CLI path.
    if os.environ.get("OEMER_MODE", "").lower() == "subprocess":
        return _run_oemer_subprocess(png_path, work_dir)

    if _INPROCESS_OK is False:
        return _run_oemer_subprocess(png_path, work_dir)

    try:
        with _OEMER_LOCK:
            result = _run_oemer_inprocess(png_path, work_dir)
        _INPROCESS_OK = True
        return result
    except Exception as e:
        if _INPROCESS_OK is None:
            log.warning(
                "in-process oemer unavailable (%s); falling back to subprocess.",
                e,
            )
            _INPROCESS_OK = False
            return _run_oemer_subprocess(png_path, work_dir)
        # In-process previously worked — this page genuinely failed.
        raise


def _concat_musicxml(xml_paths: list[str], out_path: str) -> str:
    """Stitch per-page MusicXML files into one score.

    Strategy: use the first score as the skeleton (keeps its instrument
    declaration, clef, key, time signature, tempo). For subsequent pages,
    strip leading meta (clef / key / time / tempo) from measure 1 so we
    don't reset the timebase, then append all measures to part 0 of the
    skeleton with renumbered measure indices.
    """
    if not xml_paths:
        raise RuntimeError("No MusicXML pages to concatenate.")

    from music21 import converter, stream, meter, key, clef, tempo

    base = converter.parse(xml_paths[0])
    base_parts = list(base.parts)
    if not base_parts:
        raise RuntimeError("OMR produced a score with no parts.")
    skel_part = base_parts[0]

    measures = list(skel_part.getElementsByClass(stream.Measure))
    next_num = (measures[-1].number + 1) if measures else 1

    for xml_path in xml_paths[1:]:
        page = converter.parse(xml_path)
        page_parts = list(page.parts)
        if not page_parts:
            log.warning("Page MusicXML has no part, skipping: %s", xml_path)
            continue
        page_part = page_parts[0]

        page_measures = list(page_part.getElementsByClass(stream.Measure))
        if not page_measures:
            continue

        # Strip leading meta from the very first measure of this page so we
        # don't re-declare clef / key / meter / tempo mid-piece.
        first = page_measures[0]
        for cls in (meter.TimeSignature, key.KeySignature, clef.Clef,
                    tempo.MetronomeMark):
            for elem in list(first.getElementsByClass(cls)):
                first.remove(elem)

        for m in page_measures:
            m_copy = copy.deepcopy(m)
            m_copy.number = next_num
            next_num += 1
            skel_part.append(m_copy)

    base.write("musicxml", fp=out_path)
    log.info(
        "Concatenated %d page(s) → %s (%d measures)",
        len(xml_paths), out_path, next_num - 1,
    )
    return out_path


@register_omr("oemer")
class OemerAdapter(OmrAdapter):
    """Adapter around the ``oemer`` CLI, with multi-page PDF support."""

    def pdf_to_musicxml(self, pdf_path: str, out_path: str, progress_cb=None) -> str:
        import fitz  # PyMuPDF

        def _report(msg: str) -> None:
            if progress_cb:
                try: progress_cb(msg)
                except Exception: pass

        with tempfile.TemporaryDirectory() as work_dir:
            doc = fitz.open(pdf_path)
            if doc.page_count == 0:
                raise RuntimeError("PDF contains no pages")

            log.info("PDF has %d page(s); running oemer on each.", doc.page_count)
            _report(f"OMR: {doc.page_count} page(s) to process")

            page_xmls: list[str] = []
            for i in range(doc.page_count):
                page = doc.load_page(i)
                # 200dpi ≈ 1654×2339 for A4 — still plenty for oemer, and
                # ~2× faster than 300dpi on the free HF Space CPU.
                pix = page.get_pixmap(dpi=200, alpha=False)
                png_path = os.path.join(work_dir, f"page_{i + 1:03d}.png")
                pix.save(png_path)
                log.info(
                    "Rendered page %d/%d → %s (%dx%d)",
                    i + 1, doc.page_count, png_path, pix.width, pix.height,
                )
                _report(f"OMR page {i + 1}/{doc.page_count}: rendering → inference")
                try:
                    xml_path = _run_oemer(png_path, work_dir)
                except Exception as e:
                    # One bad page shouldn't kill the whole run — skip it.
                    log.error("oemer failed on page %d: %s", i + 1, e)
                    _report(f"OMR page {i + 1}/{doc.page_count}: FAILED ({e})")
                    continue
                _report(f"OMR page {i + 1}/{doc.page_count}: done")
                page_xmls.append(xml_path)
            doc.close()

            if not page_xmls:
                raise RuntimeError("oemer produced no MusicXML for any page.")

            if len(page_xmls) == 1:
                shutil.copyfile(page_xmls[0], out_path)
                log.info("Single usable page → %s", out_path)
                return out_path

            _report(f"Stitching {len(page_xmls)} page MusicXML files")
            _concat_musicxml(page_xmls, out_path)
            return out_path
