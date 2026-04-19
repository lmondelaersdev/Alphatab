"""Score loaders: file → :class:`ScoreBundle`.

Supported out of the box:
  - MusicXML (`.xml`, `.musicxml`, `.mxl`)         via music21
  - MIDI     (`.mid`, `.midi`)                     via music21
  - Guitar Pro (`.gp`, `.gp3`–`.gp5`, `.gpx`, `.gp7`)  via PyGuitarPro

PDF is NOT handled here — register an :class:`OmrAdapter` that emits MusicXML
and re-dispatch through the MusicXML loader.
"""
from __future__ import annotations

import os

from .base import BarMark, NoteEvent, ScoreBundle, ScoreLoader
from .registry import SCORE_LOADERS_BY_EXT, pick_score_loader, register_score_loader


def load_score(path: str) -> ScoreBundle:
    """Entry point: dispatch on file extension."""
    ext = os.path.splitext(path)[1].lower()
    loader_cls = pick_score_loader(ext)
    if loader_cls is None:
        known = sorted(SCORE_LOADERS_BY_EXT)
        raise ValueError(
            f"No score loader registered for '{ext}'. Known extensions: {known}"
        )
    return loader_cls().load(path)


# ─────────────────────────────────────────────────────────────────────────────
# MusicXML / MIDI via music21
# ─────────────────────────────────────────────────────────────────────────────

@register_score_loader(".xml", ".musicxml", ".mxl", ".mid", ".midi")
class Music21Loader(ScoreLoader):
    def load(self, path: str) -> ScoreBundle:
        from music21 import chord as m21chord
        from music21 import converter, note as m21note, stream as m21stream, tempo as m21tempo

        score = converter.parse(path)

        # Tempo map in quarter-length coordinates. Fall back to 120 BPM if the
        # source has no metronome marks — DTW is tempo-invariant anyway so
        # this only matters for the bar_sec/note_sec return values.
        boundaries = score.metronomeMarkBoundaries()
        if not boundaries:
            boundaries = [(0.0, float("inf"), m21tempo.MetronomeMark(number=120))]

        def ql_to_sec(ql: float) -> float:
            t = 0.0
            for (o_start, o_end, mm) in boundaries:
                bpm = float(mm.number or 120.0)
                if ql <= o_start:
                    return t
                span = min(ql, o_end) - o_start
                if span > 0:
                    t += span * 60.0 / bpm
                if ql <= o_end:
                    return t
            return t

        # Notes (across all parts)
        notes: list[NoteEvent] = []
        parts = list(score.parts) or [score]
        for part_index, part in enumerate(parts):
            for n in part.flatten().notes:
                try:
                    start_ql = float(n.getOffsetInHierarchy(score))
                except Exception:
                    start_ql = float(n.offset)
                end_ql = start_ql + float(n.quarterLength or 0.0)
                start_s = ql_to_sec(start_ql)
                end_s = ql_to_sec(end_ql)
                if isinstance(n, m21note.Note):
                    p = n.pitch
                    notes.append(NoteEvent(
                        start_sec=start_s, end_sec=end_s,
                        pitch_class=p.pitchClass, midi=int(p.midi),
                        part_index=part_index,
                    ))
                elif isinstance(n, m21chord.Chord):
                    for p in n.pitches:
                        notes.append(NoteEvent(
                            start_sec=start_s, end_sec=end_s,
                            pitch_class=p.pitchClass, midi=int(p.midi),
                            part_index=part_index,
                        ))

        # Bar boundaries (from first part)
        bars: list[BarMark] = []
        first_part = parts[0]
        for m in first_part.getElementsByClass(m21stream.Measure):
            mn = m.measureNumber
            if mn is None:
                continue
            try:
                start_ql = float(m.getOffsetInHierarchy(score))
            except Exception:
                start_ql = float(m.offset)
            bars.append(BarMark(index=int(mn), start_sec=ql_to_sec(start_ql)))
        bars.sort(key=lambda b: b.start_sec)

        # Overall duration
        if notes:
            duration = max(n.end_sec for n in notes)
        elif bars:
            duration = bars[-1].start_sec + 2.0
        else:
            duration = 0.0

        tempo_bpm: float | None = None
        if boundaries and boundaries[0][2].number:
            tempo_bpm = float(boundaries[0][2].number)

        ext = os.path.splitext(path)[1].lower()
        fmt = "midi" if ext in {".mid", ".midi"} else "musicxml"

        return ScoreBundle(
            notes=tuple(notes),
            bars=tuple(bars),
            duration_sec=duration,
            tempo_bpm=tempo_bpm,
            source_format=fmt,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Guitar Pro via PyGuitarPro
# ─────────────────────────────────────────────────────────────────────────────

@register_score_loader(".gp", ".gp3", ".gp4", ".gp5", ".gpx", ".gp7")
class GuitarProLoader(ScoreLoader):
    """Convert beats/notes to timed events using the file's tempo map.

    Fret/string info is folded into MIDI pitch; alignment doesn't need the tab
    structure (AlphaTab already renders the .gp on the frontend)."""

    def load(self, path: str) -> ScoreBundle:
        import guitarpro

        song = guitarpro.parse(path)
        tempo_bpm = float(song.tempo or 120.0)
        TICKS_PER_QUARTER = guitarpro.Duration.quarterTime  # typically 960

        def ticks_to_sec(ticks: int, bpm: float) -> float:
            q = ticks / TICKS_PER_QUARTER
            return q * (60.0 / bpm)

        notes: list[NoteEvent] = []
        bars: list[BarMark] = []
        duration_sec = 0.0

        for part_index, track in enumerate(song.tracks):
            tuning = [s.value for s in track.strings]  # MIDI of open strings
            cur_tempo = tempo_bpm
            t = 0.0
            for measure_idx, measure in enumerate(track.measures, start=1):
                bar_start = t
                if part_index == 0:
                    bars.append(BarMark(index=measure_idx, start_sec=bar_start))

                measure_end = bar_start
                for voice in measure.voices:
                    t_voice = bar_start
                    for beat in voice.beats:
                        mix = getattr(getattr(beat, "effect", None), "mixTableChange", None)
                        if mix and mix.tempo and mix.tempo.value:
                            cur_tempo = float(mix.tempo.value)

                        beat_sec = ticks_to_sec(beat.duration.time, cur_tempo)

                        if beat.status.name != "rest":
                            for gp_note in beat.notes:
                                if gp_note.type.name == "rest":
                                    continue
                                string_idx = gp_note.string
                                if not (1 <= string_idx <= len(tuning)):
                                    continue
                                midi = tuning[string_idx - 1] + gp_note.value
                                notes.append(NoteEvent(
                                    start_sec=t_voice,
                                    end_sec=t_voice + beat_sec,
                                    pitch_class=midi % 12,
                                    midi=int(midi),
                                    part_index=part_index,
                                ))

                        t_voice += beat_sec
                    measure_end = max(measure_end, t_voice)

                t = measure_end
                duration_sec = max(duration_sec, t)

        # Dedupe bars (we only wrote from track 0 but keep the safeguard)
        seen: dict[int, BarMark] = {}
        for b in bars:
            seen.setdefault(b.index, b)
        bars = sorted(seen.values(), key=lambda b: b.start_sec)

        ext = os.path.splitext(path)[1].lower()
        return ScoreBundle(
            notes=tuple(notes),
            bars=tuple(bars),
            duration_sec=duration_sec,
            tempo_bpm=tempo_bpm,
            source_format=f"gp ({ext.lstrip('.')})",
        )
