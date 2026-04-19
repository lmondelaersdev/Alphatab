"""Synthesise a chroma matrix directly from score note events.

Much cheaper than rendering audio through a soft-synth + CQT. The resulting
12×N matrix is column-normalised so cosine DTW produces sensible costs.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .base import NoteEvent


def score_to_chroma(
    notes: Iterable[NoteEvent],
    duration_sec: float,
    *,
    sample_rate: int = 22050,
    hop_length: int = 512,
    harmonic_spread: bool = True,
) -> np.ndarray:
    """Build a 12 × n_frames chroma from a list of note events.

    Each note contributes a constant weight to its pitch class over its
    duration. When ``harmonic_spread`` is on we also add attenuated energy to
    the perfect fifth and major third of each note — this makes score chroma
    look closer to real audio chroma (which always contains harmonics) and
    markedly improves DTW accuracy on distorted/overdriven recordings.
    """
    n_frames = int(np.ceil(duration_sec * sample_rate / hop_length)) + 1
    chroma = np.zeros((12, n_frames), dtype=np.float32)

    fifth_weight = 0.35 if harmonic_spread else 0.0
    third_weight = 0.15 if harmonic_spread else 0.0

    for ev in notes:
        s = max(0, int(np.floor(ev.start_sec * sample_rate / hop_length)))
        e = min(n_frames, int(np.ceil(ev.end_sec * sample_rate / hop_length)))
        if e <= s:
            e = s + 1
        chroma[ev.pitch_class, s:e] += 1.0
        if harmonic_spread:
            chroma[(ev.pitch_class + 7) % 12, s:e] += fifth_weight
            chroma[(ev.pitch_class + 4) % 12, s:e] += third_weight

    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return chroma / norms
