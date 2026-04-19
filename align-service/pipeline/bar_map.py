"""Turn a DTW warping path into a clean per-bar mapping the frontend can use."""
from __future__ import annotations

import bisect
from typing import Sequence

import numpy as np

from .base import BarMark


def frames_to_seconds(frames: np.ndarray, sample_rate: int, hop_length: int) -> np.ndarray:
    return frames.astype(np.float64) * hop_length / sample_rate


def warp_path_to_seconds(
    wp: np.ndarray, sample_rate: int, hop_length: int
) -> list[tuple[float, float]]:
    """(score_frame, audio_frame) → (score_sec, audio_sec) list."""
    if wp.size == 0:
        return []
    score_sec = frames_to_seconds(wp[:, 0], sample_rate, hop_length)
    audio_sec = frames_to_seconds(wp[:, 1], sample_rate, hop_length)
    return list(zip(score_sec.tolist(), audio_sec.tolist()))


def downsample_path(
    path: list[tuple[float, float]], step_sec: float = 0.05
) -> list[tuple[float, float]]:
    """Drop monotone dense points so the JSON stays small. Keeps ~20 points/s."""
    if not path:
        return path
    out = [path[0]]
    for s, a in path[1:]:
        if s - out[-1][0] >= step_sec:
            out.append((s, a))
    if out[-1] != path[-1]:
        out.append(path[-1])
    return out


def bars_to_audio(
    bars: Sequence[BarMark],
    wp: np.ndarray,
    sample_rate: int,
    hop_length: int,
) -> list[dict]:
    """For each bar start in score time, look up the corresponding audio time
    via the warping path (nearest-neighbour on the score axis).
    """
    if wp.size == 0 or not bars:
        return []

    score_frames = wp[:, 0]
    audio_frames = wp[:, 1]

    sorted_idx = np.argsort(score_frames, kind="mergesort")
    sf_sorted = score_frames[sorted_idx]
    af_sorted = audio_frames[sorted_idx]
    sf_list = sf_sorted.tolist()

    out = []
    for bar in bars:
        target_frame = int(round(bar.start_sec * sample_rate / hop_length))
        idx = bisect.bisect_left(sf_list, target_frame)
        idx = min(max(idx, 0), len(sf_list) - 1)
        # Prefer the closer of (idx-1, idx)
        if idx > 0 and abs(sf_list[idx - 1] - target_frame) < abs(sf_list[idx] - target_frame):
            idx -= 1
        af = af_sorted[idx]
        out.append({
            "index": int(bar.index),
            "score_sec": round(float(bar.start_sec), 4),
            "audio_sec": round(float(af) * hop_length / sample_rate, 4),
        })
    return out
