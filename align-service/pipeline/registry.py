"""Plugin registry.

Components (score loaders, aligners, transcribers, OMR adapters) register
themselves here with a decorator so that `app.py` stays agnostic and the whole
pipeline stays open for extension without modification.
"""
from __future__ import annotations

from typing import TypeVar

from .base import Aligner, OmrAdapter, ScoreLoader, Transcriber

T = TypeVar("T")

ALIGNERS: dict[str, type[Aligner]] = {}
TRANSCRIBERS: dict[str, type[Transcriber]] = {}
OMR_ADAPTERS: dict[str, type[OmrAdapter]] = {}
SCORE_LOADERS_BY_EXT: dict[str, type[ScoreLoader]] = {}


def register_aligner(name: str):
    def deco(cls: type[Aligner]) -> type[Aligner]:
        cls.name = name
        ALIGNERS[name] = cls
        return cls
    return deco


def register_transcriber(name: str):
    def deco(cls: type[Transcriber]) -> type[Transcriber]:
        cls.name = name
        TRANSCRIBERS[name] = cls
        return cls
    return deco


def register_omr(name: str):
    def deco(cls: type[OmrAdapter]) -> type[OmrAdapter]:
        cls.name = name
        OMR_ADAPTERS[name] = cls
        return cls
    return deco


def register_score_loader(*extensions: str):
    def deco(cls: type[ScoreLoader]) -> type[ScoreLoader]:
        cls.extensions = tuple(e.lower() for e in extensions)
        for ext in cls.extensions:
            SCORE_LOADERS_BY_EXT[ext] = cls
        return cls
    return deco


def pick_score_loader(extension: str) -> type[ScoreLoader] | None:
    return SCORE_LOADERS_BY_EXT.get(extension.lower())


def pick_aligner(name: str) -> type[Aligner] | None:
    return ALIGNERS.get(name)


def pick_omr(name: str | None) -> type[OmrAdapter] | None:
    if name:
        return OMR_ADAPTERS.get(name)
    return next(iter(OMR_ADAPTERS.values()), None)
