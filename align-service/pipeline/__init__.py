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
