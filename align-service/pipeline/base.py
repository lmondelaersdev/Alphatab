"""Abstract base classes for pluggable pipeline components."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class NoteEvent:
    """A time-stamped pitched event extracted from a score."""
    start_sec: float
    end_sec: float
    pitch_class: int      # 0..11
    midi: int             # 0..127
    part_index: int = 0   # which track/part it came from


@dataclass(frozen=True)
class BarMark:
    """A bar boundary in score time."""
    index: int            # 1-based measure number as in the source
    start_sec: float


@dataclass(frozen=True)
class ScoreBundle:
    """Everything we need from a parsed score."""
    notes: Sequence[NoteEvent]
    bars: Sequence[BarMark]
    duration_sec: float
    tempo_bpm: float | None
    source_format: str     # 'musicxml' | 'midi' | 'gp' | ...


class ScoreLoader(ABC):
    """Parse a score file into a :class:`ScoreBundle`."""

    #: file extensions this loader handles, e.g. ('.xml', '.musicxml')
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, path: str) -> ScoreBundle: ...


class Aligner(ABC):
    """Align two chroma matrices (12 x N) and return a warping path."""

    name: str = "unnamed"

    @abstractmethod
    def align(
        self,
        score_chroma: np.ndarray,
        audio_chroma: np.ndarray,
        *,
        subseq: bool = False,
    ) -> np.ndarray:
        """Return an (M, 2) int array of (score_frame, audio_frame) pairs,
        ordered from earliest to latest."""


class Transcriber(ABC):
    """Optional: audio → note events.

    Not used by `/align` yet — reserved for future TabCNN / MT3 integrations
    so the frontend can fall back to transcription when no score is supplied.
    """

    name: str = "unnamed"

    @abstractmethod
    def transcribe(self, audio_path: str) -> Sequence[NoteEvent]: ...


class OmrAdapter(ABC):
    """Optional: PDF → MusicXML. Not bundled by default."""

    name: str = "unnamed"

    @abstractmethod
    def pdf_to_musicxml(self, pdf_path: str, out_path: str) -> str: ...
