"""Pipeline components for score-informed audio alignment.

The public surface is deliberately small:

    from pipeline import registry, base
    from pipeline.score_loader import load_score
    from pipeline.audio_to_chroma import audio_to_chroma
    from pipeline.score_to_chroma import score_to_chroma
    from pipeline.aligner_dtw import DtwCosineAligner

Everything else is swappable via `registry.register_*` decorators.
"""

from . import registry       # noqa: F401
from . import aligner_dtw    # noqa: F401 — registers "dtw-cosine"
from . import score_loader   # noqa: F401 — registers MusicXML/MIDI/GP loaders
from . import omr            # noqa: F401 — registration point for OMR adapters

# Heavier optional adapters: only register if their runtime deps import.
# Order matters — pick_omr(None) returns the first registered adapter, so
# put the TAB-aware one first since guitar scores are the primary target.
try:
    from . import omr_tab  # noqa: F401 — registers "tabocr"
except Exception as _e:  # pragma: no cover — logged, not fatal
    import logging
    logging.getLogger("align").warning("tabocr adapter not loaded: %s", _e)

try:
    from . import omr_oemer  # noqa: F401 — registers "oemer"
except Exception as _e:  # pragma: no cover — logged, not fatal
    import logging
    logging.getLogger("align").warning("oemer adapter not loaded: %s", _e)
