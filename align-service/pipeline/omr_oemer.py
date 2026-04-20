"""PDF → MusicXML via the `oemer` end-to-end OMR model.

oemer (https://github.com/BreezeWhite/oemer) is a Python/ONNX model that
takes a rendered page image and produces a MusicXML file. We convert each
PDF page to PNG with PyMuPDF, run oemer per page, then concatenate the
per-page MusicXML files into a single score via music21 so the alignment
path covers the whole piece.
"""
from __future__ import annotations

import copy
import logging
import os
import shutil
import subprocess
import tempfile

from .base import OmrAdapter
from .registry import register_omr

log = logging.getLogger("align.omr")


def _run_oemer(png_path: str, work_dir: str) -> str:
    """Run oemer on one page PNG. Returns the path to the produced MusicXML."""
    cmd = ["oemer", "--output-path", work_dir, png_path]
    log.info("Running: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=work_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
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

    stem = os.path.splitext(os.path.basename(png_path))[0]
    candidates = [
        os.path.join(work_dir, stem + ".musicxml"),
        os.path.join(work_dir, stem + ".xml"),
    ]
    for root, _, files in os.walk(work_dir):
        for f in files:
            if f.lower().endswith((".musicxml", ".xml")) and stem in f:
                candidates.append(os.path.join(root, f))

    produced = next((p for p in candidates if os.path.isfile(p)), None)
    if not produced:
        raise RuntimeError(
            "oemer completed but produced no MusicXML for %s. stdout tail: %s"
            % (png_path, proc.stdout[-400:])
        )
    return produced


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

    def pdf_to_musicxml(self, pdf_path: str, out_path: str) -> str:
        import fitz  # PyMuPDF

        with tempfile.TemporaryDirectory() as work_dir:
            doc = fitz.open(pdf_path)
            if doc.page_count == 0:
                raise RuntimeError("PDF contains no pages")

            log.info("PDF has %d page(s); running oemer on each.", doc.page_count)

            page_xmls: list[str] = []
            for i in range(doc.page_count):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=300, alpha=False)
                png_path = os.path.join(work_dir, f"page_{i + 1:03d}.png")
                pix.save(png_path)
                log.info(
                    "Rendered page %d/%d → %s (%dx%d)",
                    i + 1, doc.page_count, png_path, pix.width, pix.height,
                )
                try:
                    xml_path = _run_oemer(png_path, work_dir)
                except Exception as e:
                    # One bad page shouldn't kill the whole run — skip it.
                    log.error("oemer failed on page %d: %s", i + 1, e)
                    continue
                page_xmls.append(xml_path)
            doc.close()

            if not page_xmls:
                raise RuntimeError("oemer produced no MusicXML for any page.")

            if len(page_xmls) == 1:
                shutil.copyfile(page_xmls[0], out_path)
                log.info("Single usable page → %s", out_path)
                return out_path

            _concat_musicxml(page_xmls, out_path)
            return out_path
