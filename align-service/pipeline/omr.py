"""PDF OMR adapter point.

No adapter bundled by default — the base image stays slim. To enable PDF
upload support, add a heavier Dockerfile (e.g. `Dockerfile.omr`) that
installs Audiveris or `oemer` and register an adapter here::

    from pipeline.base import OmrAdapter
    from pipeline.registry import register_omr

    @register_omr("audiveris")
    class AudiverisAdapter(OmrAdapter):
        def pdf_to_musicxml(self, pdf_path: str, out_path: str) -> str:
            # call `audiveris -batch -export <pdf> -output <dir>`
            ...
            return out_path

The `/align-pdf` endpoint will then pick it up automatically via
`registry.pick_omr()`.
"""
