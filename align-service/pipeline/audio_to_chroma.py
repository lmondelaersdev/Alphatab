"""Audio → chroma matrix for DTW matching."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AudioChroma:
    chroma: np.ndarray           # 12 × n_frames, column-normalised
    duration_sec: float
    sample_rate: int
    hop_length: int


def audio_to_chroma(
    path: str,
    *,
    sample_rate: int = 22050,
    hop_length: int = 512,
) -> AudioChroma:
    """Load an audio file (any format ffmpeg can decode) and return a chroma
    matrix from a CQT spectrogram.

    CQT chroma is more robust than STFT chroma for harmonic content on guitar
    and piano-style recordings. Column-normalised so cosine DTW is sensible.
    """
    import librosa

    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    duration_sec = float(len(y)) / sr

    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=hop_length, n_chroma=12,
    ).astype(np.float32)

    # Replace any silent / zero-norm columns with a flat 1/12 distribution
    # so librosa's cosine DTW doesn't produce NaN cost entries.
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    zero_cols = (norms[0] < 1e-8)
    if zero_cols.any():
        chroma[:, zero_cols] = 1.0 / 12.0
        norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    chroma = chroma / norms

    return AudioChroma(
        chroma=chroma,
        duration_sec=duration_sec,
        sample_rate=sr,
        hop_length=hop_length,
    )
