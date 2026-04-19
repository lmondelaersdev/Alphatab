"""Dynamic time warping aligner with cosine cost."""
from __future__ import annotations

import numpy as np

from .base import Aligner
from .registry import register_aligner


@register_aligner("dtw-cosine")
class DtwCosineAligner(Aligner):
    """Vanilla DTW between two chroma matrices with 1 − cosine cost.

    Set ``subseq=True`` when the audio is expected to contain material outside
    the score (intros, outros, applause). Then the score is matched as a
    subsequence inside the audio, and the returned path only covers the score
    region.
    """

    def align(
        self,
        score_chroma: np.ndarray,
        audio_chroma: np.ndarray,
        *,
        subseq: bool = False,
    ) -> np.ndarray:
        import librosa

        _D, wp = librosa.sequence.dtw(
            X=score_chroma,
            Y=audio_chroma,
            metric="cosine",
            subseq=subseq,
            backtrack=True,
        )
        # librosa returns the path end → start; reverse to ascending.
        wp = np.asarray(wp, dtype=np.int64)[::-1]
        return wp
