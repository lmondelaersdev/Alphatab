"""PDF → MusicXML via the `oemer` end-to-end OMR model.

oemer (https://github.com/BreezeWhite/oemer) is a Python/ONNX model that
takes a rendered page image and produces a MusicXML file. We convert PDFs
to PNG with PyMuPDF and run one page through oemer's CLI, then read the
resulting ``.musicxml`` back into memory.

Multi-page PDFs: we process only the first page. Multi-page stitching is
unreliable without rest/repeat inference; users should pre-trim the PDF to
the excerpt they want aligned. A warning is logged when extra pages are
dropped so the behaviour is transparent.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

from .base import OmrAdapter
from .registry import register_omr

log = logging.getLogger("align.omr")


@register_omr("oemer")
class OemerAdapter(OmrAdapter):
    """Adapter around the ``oemer`` CLI."""

    def pdf_to_musicxml(self, pdf_path: str, out_path: str) -> str:
        import fitz  # PyMuPDF

        with tempfile.TemporaryDirectory() as work_dir:
            doc = fitz.open(pdf_path)
            if doc.page_count == 0:
                raise RuntimeError("PDF contains no pages")
            if doc.page_count > 1:
                log.warning(
                    "PDF has %d pages; oemer adapter only processes page 1. "
                    "Trim the PDF to the excerpt you want aligned.",
                    doc.page_count,
                )

            page = doc.load_page(0)
            # 300dpi gives oemer enough detail without blowing up memory.
            pix = page.get_pixmap(dpi=300, alpha=False)
            png_path = os.path.join(work_dir, "page.png")
            pix.save(png_path)
            doc.close()

            log.info("Rendered PDF page 1 → %s (%dx%d)", png_path, pix.width, pix.height)

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

            # oemer writes <input-stem>.musicxml next to the input by default.
            candidates = [
                os.path.join(work_dir, "page.musicxml"),
                os.path.join(work_dir, "page.xml"),
            ]
            for root, _, files in os.walk(work_dir):
                for f in files:
                    if f.lower().endswith((".musicxml", ".xml")):
                        candidates.append(os.path.join(root, f))

            produced = next((p for p in candidates if os.path.isfile(p)), None)
            if not produced:
                raise RuntimeError(
                    "oemer completed but produced no MusicXML. stdout tail: %s"
                    % proc.stdout[-400:]
                )

            shutil.copyfile(produced, out_path)
            log.info("OMR output → %s", out_path)
            return out_path
