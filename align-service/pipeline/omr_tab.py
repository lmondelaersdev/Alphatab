"""PDF guitar-TAB → MusicXML by direct parsing of the tablature stave.

Instead of relying on a generic OMR model (which gets guitar scores wrong
because it can't distinguish notation from TAB staves), this adapter
detects the six-line TAB grid, finds the fret-number digits on each
string, and converts them into note events. The result is MusicXML with
``<technical><string/><fret/></technical>`` markup so alphaTab can render
both the standard notation and a proper TAB stave.

Pipeline
========

1. **Rasterise** each PDF page with PyMuPDF at 200 dpi.
2. **Staff detection** — horizontal morphological opening isolates long
   horizontal lines. Cluster them in y; groups of six roughly equally
   spaced lines are TAB staves.
3. **Bar-line detection** — within a stave's vertical range, find
   columns that are dark from top to bottom.
4. **Digit OCR per string** — crop a thin strip centred on each of the
   six string lines and run tesseract with a digit-only whitelist,
   keeping each recognised word's bounding box. This handles 2-digit
   frets naturally.
5. **Chord clustering** — digits within ±``CHORD_X_TOLERANCE_PX`` in x
   are treated as a simultaneous chord event on different strings.
6. **Rhythm** — spread events proportionally by x-position within each
   detected measure, at a user-configurable tempo / time signature.
7. **Theory cleanup** — estimate the key from the pitch-class histogram
   using Krumhansl-Schmuckler, then for each note try the obvious
   OCR-confusion alternatives (0↔8, 3↔8, 6↔9, 1↔7) and keep whichever
   variant best fits the key *and* is closest to the local pitch
   median.
8. **Emit MusicXML** directly via ``ElementTree`` with per-note string
   and fret so alphaTab renders a TAB stave.

Standard tuning is assumed for now. The top printed line = string 1 =
high E (MIDI 64); bottom line = string 6 = low E (MIDI 40).
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import OmrAdapter
from .registry import register_omr

log = logging.getLogger("align.omr.tab")


# --- Tunable parameters -------------------------------------------------

# Minimum horizontal line length (as fraction of page width) to count as a
# potential staff line. Short ledger dashes should be ignored.
STAFF_LINE_MIN_LEN_FRAC = 0.25

# Two lines are considered part of the same staff if their spacing is
# within ±this factor of the median spacing within the group.
STAFF_SPACING_TOLERANCE = 0.50

# Minimum / maximum vertical space between adjacent lines of a TAB stave.
# Calibrated for 200dpi A4; broad enough for most sheet sizes.
STAFF_LINE_MIN_SPACING_PX = 5
STAFF_LINE_MAX_SPACING_PX = 45

# Two detected lines closer than this are merged (morphology sometimes
# splits one thick line into two CCs).
STAFF_LINE_MERGE_PX = 3

# When we crop a horizontal strip around a string line for OCR, this is
# the half-height (pixels above + below the line).
DIGIT_STRIP_HALF_HEIGHT_PX = 22

# Digits within this x-distance of each other are treated as a chord
# event on different strings of the same beat.
CHORD_X_TOLERANCE_PX = 12

# A bar line is a column inside the staff whose column-sum of dark
# pixels exceeds this fraction of the staff height.
BARLINE_DARK_FRACTION = 0.80

# Bar-line columns that are this close get clustered together (a printed
# bar line is usually ~2-3 px wide at 200 dpi).
BARLINE_CLUSTER_WIDTH_PX = 6

# If rhythm inference isn't otherwise constrained, assume this.
DEFAULT_TEMPO_BPM = 110.0
DEFAULT_BEATS_PER_MEASURE = 4
DEFAULT_BEAT_UNIT = 4  # a quarter-note gets one beat

# MIDI numbers for the open strings, strings 1..6 (high E down to low E).
STANDARD_TUNING_MIDI = [64, 59, 55, 50, 45, 40]

# OCR pair substitutions to try during theory cleanup.
OCR_CONFUSIONS: dict[str, list[str]] = {
    "0": ["8", "6", "9"],
    "8": ["0", "3", "9"],
    "3": ["8"],
    "6": ["9", "0"],
    "9": ["6", "0"],
    "1": ["7", "4"],
    "7": ["1"],
    "4": ["1"],
    "5": ["6", "3"],
    "2": ["3"],
}


# --- Data classes -------------------------------------------------------

@dataclass
class TabStave:
    """One detected 6-line TAB staff on a page."""
    line_ys: list[int]          # 6 y-coordinates, top (high E) → bottom (low E)
    x_start: int
    x_end: int

    @property
    def y_top(self) -> int: return min(self.line_ys)
    @property
    def y_bottom(self) -> int: return max(self.line_ys)
    @property
    def height(self) -> int: return self.y_bottom - self.y_top


@dataclass
class Digit:
    """One OCR'd fret number sitting on a given string."""
    string_idx: int   # 0..5, 0 = high E line
    fret: int         # 0..24 typically
    x_center: float
    confidence: float


@dataclass
class TabEvent:
    """A tablature event — one or more simultaneous notes (a chord)."""
    x_center: float
    notes: list[Digit] = field(default_factory=list)  # string_idx unique

    @property
    def midis(self) -> list[int]:
        return [STANDARD_TUNING_MIDI[d.string_idx] + d.fret for d in self.notes]


@dataclass
class TabPage:
    staves: list[TabStave]
    events_per_stave: list[list[TabEvent]]
    barlines_per_stave: list[list[int]]  # x-coords


# --- Image utilities ----------------------------------------------------

def _to_binary(img_gray: np.ndarray) -> np.ndarray:
    """Otsu threshold → uint8 binary image where 255 = foreground (dark)."""
    import cv2
    _, bw = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return bw


def _detect_horizontal_lines(bw: np.ndarray, min_len_px: int) -> np.ndarray:
    """Return a binary mask isolating long horizontal runs (staff lines)."""
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len_px, 1))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)


def _cluster_line_ys(line_mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (y_center, x_start, x_end) for each detected horizontal line."""
    import cv2
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        line_mask, connectivity=8
    )
    lines = []
    h, w = line_mask.shape
    for i in range(1, num):
        x, y, cw, ch, _area = stats[i]
        # Must be thin and wide enough to be a staff line.
        if ch > 8 or cw < w * STAFF_LINE_MIN_LEN_FRAC * 0.5:
            continue
        lines.append((y + ch // 2, x, x + cw))
    lines.sort()
    return lines


def _merge_close_lines(
    lines: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Merge detections closer than ``STAFF_LINE_MERGE_PX`` — morphology
    artefacts can split one printed line into two adjacent CCs."""
    if not lines:
        return lines
    out: list[tuple[int, int, int]] = [lines[0]]
    for y, xs, xe in lines[1:]:
        py, pxs, pxe = out[-1]
        if y - py <= STAFF_LINE_MERGE_PX:
            out[-1] = ((py + y) // 2, min(pxs, xs), max(pxe, xe))
        else:
            out.append((y, xs, xe))
    return out


def _group_staves(lines: list[tuple[int, int, int]]) -> list[TabStave]:
    """Greedily grow groups of horizontal lines with similar spacing.

    Keeps groups of exactly 6 (TAB staves). 5-line notation staves and
    stray rules are skipped. Uses the UNION of line x-ranges for the
    stave's horizontal extent — a single short line at the end of a
    system shouldn't tighten the whole OCR region.
    """
    lines = _merge_close_lines(lines)
    if len(lines) < 6:
        log.info("  only %d horizontal lines after merge — need >=6", len(lines))
        return []

    ys_all = [L[0] for L in lines]
    gaps_preview = [ys_all[i + 1] - ys_all[i] for i in range(len(ys_all) - 1)]
    log.info("  %d lines; gap stats min=%d max=%d median=%d",
             len(lines), min(gaps_preview), max(gaps_preview),
             int(np.median(gaps_preview)))

    staves: list[TabStave] = []
    i = 0
    n = len(lines)

    while i <= n - 6:
        group: list[tuple[int, int, int]] = [lines[i]]
        j = i + 1
        while j < n and len(group) < 6:
            gap = lines[j][0] - group[-1][0]
            if not (STAFF_LINE_MIN_SPACING_PX <= gap <= STAFF_LINE_MAX_SPACING_PX):
                break
            if len(group) >= 2:
                prior = [group[k + 1][0] - group[k][0] for k in range(len(group) - 1)]
                med = float(np.median(prior))
                if abs(gap - med) > STAFF_SPACING_TOLERANCE * max(med, 1.0):
                    break
            group.append(lines[j])
            j += 1

        if len(group) == 6:
            ys = [g[0] for g in group]
            x_start = min(g[1] for g in group)
            x_end = max(g[2] for g in group)
            if x_end - x_start >= 80:
                staves.append(TabStave(
                    line_ys=ys, x_start=x_start, x_end=x_end,
                ))
                log.info("  ✓ TAB stave ys=%s x=%d-%d", ys, x_start, x_end)
                i = j  # skip past consumed lines
                continue
            else:
                log.info("  reject narrow 6-group ys=%s width=%d",
                         ys, x_end - x_start)
        i += 1
    return staves


def _detect_barlines(bw: np.ndarray, stave: TabStave) -> list[int]:
    """Column positions where the stave is crossed top-to-bottom."""
    y0, y1 = stave.y_top, stave.y_bottom + 1
    x0, x1 = stave.x_start, stave.x_end
    strip = bw[y0:y1, x0:x1]
    col_sums = strip.sum(axis=0) / 255.0           # dark pixels per column
    threshold = (y1 - y0) * BARLINE_DARK_FRACTION
    cols = np.where(col_sums >= threshold)[0]
    if cols.size == 0:
        return []

    # Cluster close columns.
    clusters: list[list[int]] = [[int(cols[0])]]
    for c in cols[1:]:
        if c - clusters[-1][-1] <= BARLINE_CLUSTER_WIDTH_PX:
            clusters[-1].append(int(c))
        else:
            clusters.append([int(c)])
    return [x0 + int(np.mean(cl)) for cl in clusters]


def _ocr_string_strip(
    img_gray: np.ndarray, y_line: int, x_start: int, x_end: int,
) -> list[tuple[int, float, float]]:
    """OCR one string's horizontal strip. Returns list of (fret, x_center, conf)."""
    import pytesseract

    h = img_gray.shape[0]
    y0 = max(0, y_line - DIGIT_STRIP_HALF_HEIGHT_PX)
    y1 = min(h, y_line + DIGIT_STRIP_HALF_HEIGHT_PX)
    strip = img_gray[y0:y1, x_start:x_end]

    # Increase contrast slightly; digits sit on a printed line so the
    # baseline itself adds glyph confusion — inverting lightly helps.
    try:
        data = pytesseract.image_to_data(
            strip,
            config="-c tessedit_char_whitelist=0123456789 --psm 7",
            output_type=pytesseract.Output.DICT,
        )
    except Exception as e:
        log.warning("tesseract failed on strip y=%d: %s", y_line, e)
        return []

    results: list[tuple[int, float, float]] = []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        if not text or not text.isdigit():
            continue
        try:
            fret = int(text)
        except ValueError:
            continue
        if not (0 <= fret <= 24):
            continue
        conf_raw = data["conf"][i]
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        cx = x_start + float(data["left"][i]) + float(data["width"][i]) / 2.0
        results.append((fret, cx, conf))
    return results


def _cluster_into_events(digits: list[Digit]) -> list[TabEvent]:
    """Group digits by near-equal x-centre (simultaneous strings)."""
    if not digits:
        return []
    digits_sorted = sorted(digits, key=lambda d: d.x_center)
    events: list[TabEvent] = [TabEvent(x_center=digits_sorted[0].x_center)]
    events[-1].notes.append(digits_sorted[0])
    for d in digits_sorted[1:]:
        last = events[-1]
        if abs(d.x_center - last.x_center) <= CHORD_X_TOLERANCE_PX:
            # Same event: ensure one digit per string (keep higher confidence).
            dup = next((n for n in last.notes if n.string_idx == d.string_idx), None)
            if dup:
                if d.confidence > dup.confidence:
                    last.notes.remove(dup)
                    last.notes.append(d)
            else:
                last.notes.append(d)
                # Update event x to mean of members.
                last.x_center = float(np.mean([n.x_center for n in last.notes]))
        else:
            events.append(TabEvent(x_center=d.x_center, notes=[d]))
    return events


# --- Music-theory cleanup ----------------------------------------------

# Krumhansl-Schmuckler major/minor profiles (normalised).
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _estimate_key_pcs(midis: list[int]) -> set[int]:
    """Return the pitch-class set of the best-matching major/minor key."""
    if not midis:
        return set(range(12))
    hist = np.zeros(12, dtype=np.float64)
    for m in midis:
        hist[m % 12] += 1.0
    if hist.sum() > 0:
        hist /= hist.sum()

    best_score = -1.0
    best_pcs: set[int] = set(range(12))
    for tonic in range(12):
        for profile, scale_pcs in (
            (_KS_MAJOR, [0, 2, 4, 5, 7, 9, 11]),
            (_KS_MINOR, [0, 2, 3, 5, 7, 8, 10]),
        ):
            rotated = np.roll(profile, tonic)
            rotated = rotated / np.linalg.norm(rotated)
            h = hist / (np.linalg.norm(hist) + 1e-12)
            score = float(np.dot(rotated, h))
            if score > best_score:
                best_score = score
                best_pcs = {(tonic + p) % 12 for p in scale_pcs}
    return best_pcs


def _clean_events_with_theory(
    events: list[TabEvent],
) -> list[TabEvent]:
    """Try OCR-confusion alternatives for frets that miss the key + outlier
    test. Keeps the alternative only when it both lands in key and is
    closer to the rolling pitch median than the original."""
    if not events:
        return events

    all_midis = [m for e in events for m in e.midis]
    in_key = _estimate_key_pcs(all_midis)

    # Rolling median for outlier detection (local, window = 9 events).
    def rolling_median(i: int, span: int = 9) -> float:
        lo = max(0, i - span // 2)
        hi = min(len(events), i + span // 2 + 1)
        local = [m for e in events[lo:hi] for m in e.midis]
        return float(np.median(local)) if local else 60.0

    for i, ev in enumerate(events):
        local_med = rolling_median(i)
        for d in ev.notes:
            orig_midi = STANDARD_TUNING_MIDI[d.string_idx] + d.fret
            orig_pc = orig_midi % 12
            orig_dist = abs(orig_midi - local_med)

            if orig_pc in in_key and orig_dist <= 18:
                continue  # looks fine

            # Try single-digit alternatives only (2-digit alts are a
            # different risk bucket — leave them alone for now).
            if d.fret >= 10:
                continue
            for alt_char in OCR_CONFUSIONS.get(str(d.fret), []):
                alt = int(alt_char)
                if not (0 <= alt <= 24):
                    continue
                alt_midi = STANDARD_TUNING_MIDI[d.string_idx] + alt
                alt_pc = alt_midi % 12
                alt_dist = abs(alt_midi - local_med)
                if (
                    alt_pc in in_key
                    and (orig_pc not in in_key or alt_dist < orig_dist)
                ):
                    log.info(
                        "theory-cleanup: fret %d → %d on string %d (md=%.1f)",
                        d.fret, alt, d.string_idx + 1, local_med,
                    )
                    d.fret = alt
                    break
    return events


# --- MusicXML emission --------------------------------------------------

def _midi_to_pitch_elements(midi: int) -> tuple[str, int, int]:
    """Return (step, alter, octave) for a MIDI number.

    Uses a fixed sharp-spelling (C, C#, D, D#, ...). Good enough for
    alphaTab rendering; the TAB stave is the primary output anyway.
    """
    names = [
        ("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
        ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0),
    ]
    step, alter = names[midi % 12]
    octave = midi // 12 - 1
    return step, alter, octave


def _duration_to_xml(divisions: int, quarter_fraction: float) -> tuple[int, str, int]:
    """Pick a MusicXML (duration, type, dots) from a duration in quarters.

    Very simple mapping; alphaTab is lenient about non-matching values as
    long as ``divisions`` is consistent.
    """
    quarter_fraction = max(quarter_fraction, 1.0 / 16.0)
    dur = max(1, int(round(quarter_fraction * divisions)))
    # Pick a note-type whose default quarter-count is closest.
    candidates = [
        ("whole", 4.0), ("half", 2.0), ("quarter", 1.0),
        ("eighth", 0.5), ("16th", 0.25), ("32nd", 0.125),
    ]
    best = min(candidates, key=lambda c: abs(c[1] - quarter_fraction))
    return dur, best[0], 0


def _build_musicxml(
    events_per_page: list[list[TabEvent]],
    barlines_per_page: list[list[int]],
    stave_x_ranges_per_page: list[tuple[int, int]],
    *,
    tempo_bpm: float,
    beats_per_measure: int,
    beat_unit: int,
) -> bytes:
    """Build a single-part MusicXML document with string/fret technicals."""
    divisions = 16  # 1 quarter = 16 → lets us express 16ths as integers
    beats_per_sec = tempo_bpm / 60.0
    secs_per_beat = 1.0 / beats_per_sec
    secs_per_measure = secs_per_beat * beats_per_measure

    root = ET.Element("score-partwise", version="4.0")
    pl = ET.SubElement(root, "part-list")
    sp = ET.SubElement(pl, "score-part", id="P1")
    ET.SubElement(sp, "part-name").text = "Guitar"
    scin = ET.SubElement(sp, "score-instrument", id="P1-I1")
    ET.SubElement(scin, "instrument-name").text = "Guitar"

    part = ET.SubElement(root, "part", id="P1")
    measure_counter = 0

    # Flatten events across pages into measures, using per-page bar lines.
    for page_idx, events in enumerate(events_per_page):
        barlines = barlines_per_page[page_idx]
        x_start, x_end = stave_x_ranges_per_page[page_idx]
        if not barlines:
            barlines = [x_start, x_end]
        else:
            if barlines[0] > x_start + 20:
                barlines = [x_start] + barlines
            if barlines[-1] < x_end - 20:
                barlines = barlines + [x_end]

        # Group events into measures.
        for m_idx in range(len(barlines) - 1):
            m_x0, m_x1 = barlines[m_idx], barlines[m_idx + 1]
            if m_x1 - m_x0 < 20:
                continue
            measure_counter += 1
            events_in_m = [e for e in events if m_x0 <= e.x_center < m_x1]
            events_in_m.sort(key=lambda e: e.x_center)

            measure = ET.SubElement(part, "measure", number=str(measure_counter))

            if measure_counter == 1:
                attrs = ET.SubElement(measure, "attributes")
                ET.SubElement(attrs, "divisions").text = str(divisions)
                key_el = ET.SubElement(attrs, "key")
                ET.SubElement(key_el, "fifths").text = "0"
                time_el = ET.SubElement(attrs, "time")
                ET.SubElement(time_el, "beats").text = str(beats_per_measure)
                ET.SubElement(time_el, "beat-type").text = str(beat_unit)
                clef = ET.SubElement(attrs, "clef")
                ET.SubElement(clef, "sign").text = "G"
                ET.SubElement(clef, "line").text = "2"
                staff_details = ET.SubElement(attrs, "staff-details")
                ET.SubElement(staff_details, "staff-lines").text = "6"
                # Standard tuning.
                tuning = [("E", 2), ("A", 2), ("D", 3), ("G", 3), ("B", 3), ("E", 4)]
                for line_no, (step, oct_) in enumerate(tuning, start=1):
                    st = ET.SubElement(staff_details, "staff-tuning", line=str(line_no))
                    ET.SubElement(st, "tuning-step").text = step
                    ET.SubElement(st, "tuning-octave").text = str(oct_)

                direction = ET.SubElement(measure, "direction", placement="above")
                dt = ET.SubElement(direction, "direction-type")
                metro = ET.SubElement(dt, "metronome")
                ET.SubElement(metro, "beat-unit").text = "quarter"
                ET.SubElement(metro, "per-minute").text = f"{tempo_bpm:.0f}"
                ET.SubElement(direction, "sound", tempo=f"{tempo_bpm:.2f}")

            # If the measure is empty, emit a whole rest so MusicXML is valid.
            if not events_in_m:
                note = ET.SubElement(measure, "note")
                ET.SubElement(note, "rest")
                ET.SubElement(note, "duration").text = str(divisions * beats_per_measure)
                ET.SubElement(note, "type").text = "whole"
                continue

            # Compute per-event quarter-note durations from x-span.
            span = max(1.0, float(m_x1 - m_x0))
            event_beats: list[float] = []
            for i, ev in enumerate(events_in_m):
                if i + 1 < len(events_in_m):
                    dx = events_in_m[i + 1].x_center - ev.x_center
                else:
                    dx = m_x1 - ev.x_center
                frac = max(1.0 / 16.0, dx / span)  # guard against 0
                event_beats.append(frac * beats_per_measure)

            for ev_idx, ev in enumerate(events_in_m):
                beats = event_beats[ev_idx]
                dur_units, type_name, _dots = _duration_to_xml(divisions, beats)
                # Emit the chord's notes: first is the "head", rest carry <chord/>.
                sorted_notes = sorted(ev.notes, key=lambda d: d.string_idx)
                for chord_idx, d in enumerate(sorted_notes):
                    note_el = ET.SubElement(measure, "note")
                    if chord_idx > 0:
                        ET.SubElement(note_el, "chord")
                    midi = STANDARD_TUNING_MIDI[d.string_idx] + d.fret
                    step, alter, octave = _midi_to_pitch_elements(midi)
                    pitch = ET.SubElement(note_el, "pitch")
                    ET.SubElement(pitch, "step").text = step
                    if alter:
                        ET.SubElement(pitch, "alter").text = str(alter)
                    ET.SubElement(pitch, "octave").text = str(octave)
                    ET.SubElement(note_el, "duration").text = str(dur_units)
                    ET.SubElement(note_el, "type").text = type_name
                    notations = ET.SubElement(note_el, "notations")
                    tech = ET.SubElement(notations, "technical")
                    s_el = ET.SubElement(tech, "string")
                    s_el.text = str(d.string_idx + 1)  # 1 = high E in MusicXML
                    f_el = ET.SubElement(tech, "fret")
                    f_el.text = str(d.fret)

    buf = io.BytesIO()
    ET.ElementTree(root).write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()


# --- Public adapter -----------------------------------------------------

@register_omr("tabocr")
class TabOcrAdapter(OmrAdapter):
    """Detect guitar TAB staves in a PDF and emit MusicXML with tab data."""

    def pdf_to_musicxml(self, pdf_path: str, out_path: str, progress_cb=None) -> str:
        import cv2
        import fitz  # PyMuPDF

        def _report(msg: str) -> None:
            if progress_cb:
                try: progress_cb(msg)
                except Exception: pass

        tempo_bpm = float(os.environ.get("TABOCR_TEMPO_BPM", DEFAULT_TEMPO_BPM))
        beats_per_measure = int(os.environ.get(
            "TABOCR_BEATS_PER_MEASURE", DEFAULT_BEATS_PER_MEASURE,
        ))

        events_per_page: list[list[TabEvent]] = []
        barlines_per_page: list[list[int]] = []
        stave_x_ranges_per_page: list[tuple[int, int]] = []
        total_tab_staves = 0

        with tempfile.TemporaryDirectory() as work_dir:
            doc = fitz.open(pdf_path)
            if doc.page_count == 0:
                raise RuntimeError("PDF contains no pages")

            _report(f"TAB-OCR: {doc.page_count} page(s) to scan")
            for pi in range(doc.page_count):
                page = doc.load_page(pi)
                pix = page.get_pixmap(dpi=200, alpha=False)
                png_path = os.path.join(work_dir, f"page_{pi + 1:03d}.png")
                pix.save(png_path)

                img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    log.warning("Could not read rendered page %d", pi + 1)
                    continue
                bw = _to_binary(img)
                min_len_px = max(60, int(img.shape[1] * STAFF_LINE_MIN_LEN_FRAC))
                line_mask = _detect_horizontal_lines(bw, min_len_px)
                lines = _cluster_line_ys(line_mask)
                staves = _group_staves(lines)

                log.info(
                    "Page %d/%d: %d horizontal lines → %d TAB staves",
                    pi + 1, doc.page_count, len(lines), len(staves),
                )
                _report(
                    f"TAB-OCR page {pi + 1}/{doc.page_count}: "
                    f"{len(staves)} TAB stave(s) detected"
                )

                for st_idx, stave in enumerate(staves):
                    barlines = _detect_barlines(bw, stave)
                    digits: list[Digit] = []
                    for s_idx, y in enumerate(stave.line_ys):
                        found = _ocr_string_strip(img, y, stave.x_start, stave.x_end)
                        for fret, x_center, conf in found:
                            digits.append(Digit(
                                string_idx=s_idx, fret=fret,
                                x_center=x_center, confidence=conf,
                            ))
                    events = _cluster_into_events(digits)
                    events = _clean_events_with_theory(events)
                    log.info(
                        "  stave %d: %d digits → %d events, %d barlines",
                        st_idx + 1, len(digits), len(events), len(barlines),
                    )
                    events_per_page.append(events)
                    barlines_per_page.append(barlines)
                    stave_x_ranges_per_page.append((stave.x_start, stave.x_end))
                    total_tab_staves += 1

            doc.close()

        if total_tab_staves == 0:
            raise RuntimeError(
                "No 6-line TAB staves detected in PDF. Use omr=oemer for "
                "standard-notation scores, or upload MusicXML/.gp for best "
                "results."
            )

        _report(
            f"TAB-OCR: emitting MusicXML from {sum(len(e) for e in events_per_page)} "
            f"events across {total_tab_staves} stave(s)"
        )
        xml_bytes = _build_musicxml(
            events_per_page, barlines_per_page, stave_x_ranges_per_page,
            tempo_bpm=tempo_bpm,
            beats_per_measure=beats_per_measure,
            beat_unit=DEFAULT_BEAT_UNIT,
        )
        with open(out_path, "wb") as f:
            f.write(xml_bytes)
        log.info("TAB-OCR wrote %d bytes → %s", len(xml_bytes), out_path)
        return out_path
