#!/usr/bin/env python3
"""
==============================================================================
 DRUM + BASS HUMANIZATION STUDIO
==============================================================================

WHAT THIS PROGRAM DOES
-----------------------
A single-window tool that takes a full song's drum MIDI (and its matching bass
MIDI), segments the drums into themes, and lets you polish each segment
through three phases:

  Phase 1  DRUM HUMANIZE     — run the trained humanizer model on this segment.
  Phase 2  RUSH / DRAG       — manually nudge syncopated hits earlier/later,
                                and optionally pull the strong beats back to
                                the grid, per instrument group.
  Phase 3  BASS SYNC         — lock nearby bass notes to the (now-humanized)
                                kick/snare, with a small randomized delay so
                                the two transients stay audibly distinct.

Every phase's settings are remembered PER SEGMENT — select a different
segment, adjust it, come back, and your earlier settings are exactly as you
left them. Phases are gated (Phase 2 needs Phase 1's output to adjust; Phase 3
needs Phase 2's output to sync against) but NEVER re-lock: revisit an earlier
phase, change something, and every downstream phase recomputes from the new
result the next time you preview or render it.

REUSED FROM THE OTHER PROJECT SCRIPTS (see their own docstrings for detail)
------------------------------------------------------------------------------
  drum_humanizer_v3.py        — Phase 1's model (humanize_file), DrumEvent,
                                 the MIDI grid/Config, load/write helpers.
  drum_theme_segmentation.py  — the segmentation model (compute_segment_
                                 boundaries, bar_to_seconds).
  groove_finder_ui.py         — MidiPlayer (in-process MIDI playback engine),
                                 tempo helpers, error-reporting, and the
                                 persisted-settings pattern.
Phase 2 and Phase 3's actual signal processing, and the whole window, are new.

HOW IT IS USED
---------------
  pip install torch pretty_midi mido numpy
  python drum_bass_studio.py
Then, in the window: drop a drum MIDI (top), drop the matching bass MIDI
(below it), click a segment, work through Phase 1 → 2 → 3, repeat for other
segments, then hit "Render Song" to write the final processed drum + bass
files.

Search for "DESIGN:" for inline rationale on specific decisions.
"""

import os
import sys
import json
import time
import uuid
import shutil
import random
import tempfile
import threading
import traceback
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

# sibling project scripts must be alongside this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import drum_humanizer_v3 as dhu
    HAS_HUMANIZER = True
except ImportError:
    HAS_HUMANIZER = False

try:
    import drum_theme_segmentation as dts
    HAS_SEGMENTATION = True
except ImportError:
    HAS_SEGMENTATION = False

try:
    import groove_finder_ui as gfx    # MidiPlayer, tempo helpers, error/settings helpers
    HAS_GFX = True
except ImportError:
    HAS_GFX = False


# =============================================================================
# CONSTANTS
# =============================================================================
# Every constant lives in config.py (kept alongside this script) — see that
# file's header for why, and for the full reasoning behind each value.
from config import (
    CUSTOMIZED_MARKER_COLOR, GLOBAL_SEED, GROUP_KICK, GROUP_OTHER, GROUP_SNARE,
    GROUP_TOMS, PHASE1_BASE_TEMPERATURE_OFF, PHASE1_BASE_TEMPERATURE_VEL,
    PHASE1_DEFAULT_FAST_HIT_CAP, PHASE1_DEFAULT_FEEL_VARIATION,
    PHASE1_DEFAULT_INTENSITY, PHASE1_DEFAULT_INTENSITY_AUTO,
    PHASE1_DEFAULT_INTENSITY_VALUE, PHASE1_DEFAULT_STRENGTH,
    PHASE1_FEEL_VARIATION_RANGE, PHASE1_FEEL_VARIATION_TO_TEMP_OFF,
    PHASE1_FEEL_VARIATION_TO_TEMP_VEL, PHASE1_INTENSITY_RANGE,
    PHASE1_SECTION_STARTS_OPEN, PHASE1_STRENGTH_RANGE, PHASE1_TEMP_OFF_SPREAD,
    PHASE1_TEMP_VEL_SPREAD, PHASE2_DEFAULT_QUANTIZE, PHASE2_DEFAULT_RUSH_DRAG,
    PHASE2_DENSITY_UNUSED_THRESHOLD, PHASE2_MAX_RUSH_DRAG_BEATS,
    PHASE2_QUANTIZE_RANGE, PHASE2_RUSH_DRAG_RANGE, PHASE2_SAFE_OFFSET_FRACTION,
    PHASE2_SECTION_STARTS_OPEN, PHASE2_WEAK_BEAT_DAMPING,
    PHASE3_DEFAULT_DELAY_AMOUNT, PHASE3_DEFAULT_SNAP_STRENGTH,
    PHASE3_DELAY_AMOUNT_RANGE, PHASE3_DELAY_HIGH_MAX_MS, PHASE3_DELAY_HIGH_MIN_MS,
    PHASE3_DELAY_LOW_MAX_MS, PHASE3_DELAY_LOW_MIN_MS, PHASE3_SECTION_STARTS_OPEN,
    PHASE3_SNAP_STRENGTH_RANGE, PHASE3_SNAP_THRESHOLD_BEATS,
    SEGMENTATION_CONFIDENCE_THRESHOLD, SEGMENTATION_CONTEXT_OVERLAP,
    SEGMENT_COLORS, SEGMENT_MIN_RECT_WIDTH_PX, SEGMENT_OUTLINE_WIDTH_NORMAL,
    SEGMENT_OUTLINE_WIDTH_SELECTED, SELECTED_OUTLINE_COLOR,
)


def seed_everything(seed: int = GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)


seed_everything(GLOBAL_SEED)


# =============================================================================
# SHARED MIDI HELPERS
# =============================================================================

def slice_midi_to_temp(source_path: str, start_sec: float, end_sec: float,
                       temp_dir: str, drums_only: bool = False) -> str:
    """
    Write a NEW small MIDI file containing only the notes inside
    [start_sec, end_sec) of source_path, times shifted to start at 0.
    DESIGN: same approach as groove_finder_ui.py's _slice_segment_to_temp,
    generalized to work for either the drum file (drums_only=True, since a
    stray non-drum track should never leak in) or the bass file (drums_only=
    False — a bass track is not marked is_drum in a MIDI file).
    """
    if not HAS_PRETTY_MIDI:
        raise RuntimeError("pretty_midi is required to slice segments.")
    src = pretty_midi.PrettyMIDI(source_path)
    tempo = gfx._tempo_at_time(src, start_sec) if HAS_GFX else 120.0
    out = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0, is_drum=drums_only)
    for orig in src.instruments:
        if drums_only and not orig.is_drum:
            continue
        for n in orig.notes:
            if start_sec <= n.start < end_sec:
                inst.notes.append(pretty_midi.Note(
                    velocity=n.velocity, pitch=n.pitch,
                    start=n.start - start_sec,
                    end=max(n.start - start_sec + 0.01, n.end - start_sec)))
    out.instruments.append(inst)
    path = os.path.join(temp_dir, f"slice_{uuid.uuid4().hex[:8]}.mid")
    out.write(path)
    return path, tempo, len(inst.notes)


def _report_error(context: str, exc: BaseException) -> str:
    if HAS_GFX:
        return gfx._report_error(context, exc)
    print(f"[ERROR] {context}\n  {type(exc).__name__}: {exc}")
    return f"{context}: {exc}"


# =============================================================================
# PHASE 1 — DRUM HUMANIZE  (thin, tested reuse of drum_humanizer_v3.humanize_file)
# =============================================================================

def run_phase1_humanize(checkpoint: str, input_path: str, output_path: str,
                        settings: Dict) -> None:
    """
    settings keys: strength (float, 0..2), intensity (float 0..1 or None =
    auto), feel_variation (float 0..1, maps to temperature_vel/off), fast_hit_
    cap (bool). Everything else uses humanize_file's own sensible defaults.
    """
    lo_v, hi_v = PHASE1_FEEL_VARIATION_TO_TEMP_VEL
    lo_o, hi_o = PHASE1_FEEL_VARIATION_TO_TEMP_OFF
    fv = float(np.clip(settings.get('feel_variation', PHASE1_DEFAULT_FEEL_VARIATION), 0.0, 1.0))
    temperature_vel = lo_v + fv * (hi_v - lo_v)
    temperature_off = lo_o + fv * (hi_o - lo_o)
    dhu.humanize_file(
        checkpoint, input_path, output_path,
        strength=float(settings.get('strength', PHASE1_DEFAULT_STRENGTH)),
        intensity=settings.get('intensity', PHASE1_DEFAULT_INTENSITY),
        temperature_vel=temperature_vel, temperature_off=temperature_off,
        fast_hit_cap=bool(settings.get('fast_hit_cap', PHASE1_DEFAULT_FAST_HIT_CAP)),
    )


# =============================================================================
# PHASE 2 — MANUAL RUSH / DRAG + QUANTIZE
# =============================================================================
# DESIGN: classify each note by its position WITHIN its beat, on the
# humanizer's own fine grid (cfg.grid_resolution steps per beat). A 16th-note
# subdivision is grid_resolution/4 fine steps, so "e"/"&"/"a" sit at fine
# positions 4/8/12 of each 16-step beat (for the default grid_resolution=16).
#   sixteenth_in_beat 0 -> the beat itself ("1","2","3","4")
#   sixteenth_in_beat 1 -> "e"        -> RUSH/DRAG, full strength
#   sixteenth_in_beat 2 -> "&"(and)   -> untouched by Phase 2 (not requested)
#   sixteenth_in_beat 3 -> "a"        -> RUSH/DRAG, full strength
# Beats 1 & 3 (0-indexed 0,2) -> QUANTIZE target ("the main beats").
# Beats 2 & 4 (0-indexed 1,3) -> RUSH/DRAG target too, but DAMPED
# (PHASE2_WEAK_BEAT_DAMPING) — "the 2 and the 4" always included, per spec,
# always at a reduced effect relative to e/a.

def classify_phase2_role(grid_step: int, cfg) -> str:
    """Returns 'strong_beat' (1,3 -> quantize target), 'syncopation' (e,a ->
    full rush/drag), 'weak_beat' (2,4 -> damped rush/drag), or 'none' (the
    "&"/and position, or an off-16th-grid note — untouched either way)."""
    fine_per_16th = max(1, cfg.grid_resolution // 4)
    beat_in_bar = (grid_step // cfg.grid_resolution) % cfg.beats_per_bar
    fine_in_beat = grid_step % cfg.grid_resolution
    sixteenth_in_beat = round(fine_in_beat / fine_per_16th) % 4
    if sixteenth_in_beat == 0:
        return 'strong_beat' if beat_in_bar % 2 == 0 else 'weak_beat'
    elif sixteenth_in_beat in (1, 3):
        return 'syncopation'
    return 'none'


def _instrument_group(inst: int) -> str:
    if inst in GROUP_KICK:  return 'kick'
    if inst in GROUP_SNARE: return 'snare'
    if inst in GROUP_TOMS:  return 'toms'
    return 'other'


def run_phase2_adjust(input_path: str, output_path: str, settings: Dict, cfg=None) -> None:
    """
    settings keys: rush_drag_{kick,snare,toms,other} (-1..1),
                   quantize_{kick,snare,toms,other} (0..1).
    Adds the rush/drag delta ON TOP of whatever offset is already present
    (from Phase 1 or a prior Phase 2 pass), then clips to the model's own
    representable range (cfg.max_offset_ticks) so a note never strays past
    what a single grid cell can express.
    """
    cfg = cfg or dhu.Config()
    events = dhu.load_midi_events(input_path, cfg)
    if not events:
        raise ValueError(f"No drum events found in {input_path}")
    tempo = events[0].tempo_bpm if events else 120.0
    max_delta_ticks = PHASE2_MAX_RUSH_DRAG_BEATS * cfg.ticks_per_beat
    safe_limit = cfg.ticks_per_grid * PHASE2_SAFE_OFFSET_FRACTION
    for e in events:
        role = classify_phase2_role(e.grid_step, cfg)
        if role == 'none':
            continue
        group = _instrument_group(e.instrument)
        if role in ('syncopation', 'weak_beat'):
            slider = float(settings.get(f'rush_drag_{group}', PHASE2_DEFAULT_RUSH_DRAG))
            damping = PHASE2_WEAK_BEAT_DAMPING if role == 'weak_beat' else 1.0
            new_offset = e.offset_ticks + slider * max_delta_ticks * damping
        else:   # strong_beat -> quantize toward the grid line
            slider = float(np.clip(settings.get(f'quantize_{group}', PHASE2_DEFAULT_QUANTIZE), 0.0, 1.0))
            new_offset = e.offset_ticks * (1.0 - slider)
        e.offset_ticks = int(round(np.clip(new_offset, -safe_limit, safe_limit)))
    dhu.events_to_midi(events, output_path, cfg, tempo=tempo)


# =============================================================================
# PHASE 3 — BASS SYNC TO DRUMS
# =============================================================================
# DESIGN: works entirely in ABSOLUTE TIME (seconds) via plain pretty_midi, on
# BOTH sides — never through drum_humanizer_v3's grid+offset representation.
# This sidesteps the round-trip-reassignment risk documented above (Phase 3
# only ever READS the processed drum file, never writes it back through the
# grid loader), and it's the natural representation for bass notes anyway
# (they were never drum-voice-classified or grid-quantized in the first
# place). Kick/snare hit times are identified via drum_humanizer_v3's own
# GM_DRUM_MAP (class 0 = kick, class 1 = snare), so the definition of "kick"/
# "snare" stays identical to everywhere else in this project.

def _kick_snare_times(processed_drum_path: str) -> List[float]:
    pm = pretty_midi.PrettyMIDI(processed_drum_path)
    times = []
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for n in inst.notes:
            cls = dhu.GM_DRUM_MAP.get(n.pitch, -1)
            if cls in (0, 1):   # kick or snare
                times.append(n.start)
    return sorted(times)


def _interpolated_delay_range_ms(slider: float) -> Tuple[float, float]:
    """slider 0..1 -> (min_ms, max_ms) of the random post-hit delay draw,
    interpolating linearly from the LOW range to the HIGH range."""
    s = float(np.clip(slider, 0.0, 1.0))
    lo = PHASE3_DELAY_LOW_MIN_MS + s * (PHASE3_DELAY_HIGH_MIN_MS - PHASE3_DELAY_LOW_MIN_MS)
    hi = PHASE3_DELAY_LOW_MAX_MS + s * (PHASE3_DELAY_HIGH_MAX_MS - PHASE3_DELAY_LOW_MAX_MS)
    return lo, hi


def run_phase3_sync(processed_drum_path: str, bass_path: str, output_bass_path: str,
                    settings: Dict, tempo: float = 120.0, rng: Optional[random.Random] = None) -> None:
    """
    settings keys: snap_strength (0..1), delay_amount (0..1).
    A bass note within PHASE3_SNAP_THRESHOLD_BEATS of the nearest kick/snare
    hit is pulled toward it, proportional to snap_strength (0=untouched,
    1=exact snap). EVERY bass note then gets a small random post-hit delay
    (see _interpolated_delay_range_ms) so the two transients stay distinct —
    applied unconditionally, not just to snapped notes, per spec.
    """
    rng = rng or random.Random(GLOBAL_SEED)
    snap_strength = float(np.clip(settings.get('snap_strength', PHASE3_DEFAULT_SNAP_STRENGTH), 0.0, 1.0))
    delay_amount = float(np.clip(settings.get('delay_amount', PHASE3_DEFAULT_DELAY_AMOUNT), 0.0, 1.0))

    hit_times = _kick_snare_times(processed_drum_path)
    bass = pretty_midi.PrettyMIDI(bass_path)

    snap_threshold_sec = PHASE3_SNAP_THRESHOLD_BEATS * (60.0 / tempo)
    delay_lo_ms, delay_hi_ms = _interpolated_delay_range_ms(delay_amount)

    def nearest_hit(t):
        if not hit_times:
            return None
        import bisect
        i = bisect.bisect_left(hit_times, t)
        candidates = [hit_times[j] for j in (i - 1, i) if 0 <= j < len(hit_times)]
        if not candidates:
            return None
        return min(candidates, key=lambda h: abs(h - t))

    out = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    out_inst = pretty_midi.Instrument(program=bass.instruments[0].program if bass.instruments else 33,
                                      is_drum=False,
                                      name=bass.instruments[0].name if bass.instruments else "Bass")
    for inst in bass.instruments:
        for n in inst.notes:
            new_start = n.start
            hit = nearest_hit(n.start)
            if hit is not None and snap_strength > 0.0 and abs(hit - n.start) <= snap_threshold_sec:
                new_start = n.start + snap_strength * (hit - n.start)
            if delay_amount > 0.0:
                delay_sec = rng.uniform(delay_lo_ms, delay_hi_ms) / 1000.0
                new_start += delay_sec
            shift = new_start - n.start
            out_inst.notes.append(pretty_midi.Note(
                velocity=n.velocity, pitch=n.pitch,
                start=max(0.0, new_start), end=max(0.0, n.end + shift)))
    out_inst.notes.sort(key=lambda n: n.start)
    out.instruments.append(out_inst)
    out.write(output_bass_path)


# =============================================================================
# PER-SEGMENT STATE
# =============================================================================
# DESIGN: "never re-lock" — a phase's settings/output are independent instance
# state, always re-derivable from the phase before it. Gating only controls
# whether a phase's CONTROLS are enabled (has the prior phase been run at
# least once for this segment?), never whether they can be revisited. Reopening
# Phase 1 after Phase 2/3 were already applied does NOT wipe those downstream
# results — they simply become STALE (flagged in the UI) until re-applied,
# which always recomputes from whatever Phase 1's CURRENT output now is.

def default_phase1_settings() -> Dict:
    return {'strength': PHASE1_DEFAULT_STRENGTH, 'intensity': PHASE1_DEFAULT_INTENSITY,
            'feel_variation': PHASE1_DEFAULT_FEEL_VARIATION, 'fast_hit_cap': PHASE1_DEFAULT_FAST_HIT_CAP}


def default_phase2_settings() -> Dict:
    d = {}
    for grp in ('kick', 'snare', 'toms', 'other'):
        d[f'rush_drag_{grp}'] = PHASE2_DEFAULT_RUSH_DRAG
        d[f'quantize_{grp}'] = PHASE2_DEFAULT_QUANTIZE
    return d


def default_phase3_settings() -> Dict:
    return {'snap_strength': PHASE3_DEFAULT_SNAP_STRENGTH, 'delay_amount': PHASE3_DEFAULT_DELAY_AMOUNT}


@dataclass
class SegmentSettings:
    """Everything remembered for ONE segment. `stale` flags mean "the phase
    before this one changed since I was last computed" — shown in the UI as a
    hint to re-apply, never auto-recomputed (that could mean silently re-
    running the Phase 1 MODEL on every minor slider tweak elsewhere)."""
    index: int
    start_sec: float
    end_sec: float
    start_bar: int
    end_bar: int

    phase1_settings: Dict = field(default_factory=default_phase1_settings)
    phase1_done: bool = False
    phase1_stale: bool = False
    phase1_raw_path: Optional[str] = None      # this segment's sliced RAW drums (pre-Phase-1)
    phase1_output_path: Optional[str] = None   # processed drum MIDI for this segment

    phase2_settings: Dict = field(default_factory=default_phase2_settings)
    phase2_done: bool = False
    phase2_stale: bool = False
    phase2_output_path: Optional[str] = None

    phase3_settings: Dict = field(default_factory=default_phase3_settings)
    phase3_done: bool = False
    phase3_stale: bool = False
    phase3_output_path: Optional[str] = None   # processed BASS MIDI for this segment
    bass_input_path: Optional[str] = None      # this segment's sliced RAW bass

    def is_customized(self) -> bool:
        return (self.phase1_settings != default_phase1_settings()
                or self.phase2_settings != default_phase2_settings()
                or self.phase3_settings != default_phase3_settings())

    def reset(self):
        self.phase1_settings = default_phase1_settings()
        self.phase2_settings = default_phase2_settings()
        self.phase3_settings = default_phase3_settings()
        self.phase1_done = self.phase2_done = self.phase3_done = False
        self.phase1_stale = self.phase2_stale = self.phase3_stale = False
        self.phase1_raw_path = None
        self.phase1_output_path = self.phase2_output_path = self.phase3_output_path = None


# =============================================================================
# SMALL REUSABLE WIDGET: a collapsible ("accordion") section
# =============================================================================

class CollapsibleSection(ttk.Frame):
    def __init__(self, parent, title: str, start_open: bool = False):
        super().__init__(parent)
        self._open = tk.BooleanVar(value=start_open)
        header = ttk.Frame(self)
        header.pack(fill='x')
        self._toggle_btn = ttk.Button(header, text=self._arrow() + " " + title,
                                      command=self._toggle)
        self._toggle_btn.pack(fill='x')
        self.body = ttk.Frame(self, relief='groove', borderwidth=1)
        if start_open:
            self.body.pack(fill='x', pady=(2, 6))

    def _arrow(self):
        return "\u25bc" if self._open.get() else "\u25b6"

    def _toggle(self):
        self._open.set(not self._open.get())
        title = self._toggle_btn.cget('text').split(' ', 1)[1]
        self._toggle_btn.config(text=self._arrow() + " " + title)
        if self._open.get():
            self.body.pack(fill='x', pady=(2, 6))
        else:
            self.body.pack_forget()

    def set_title(self, title: str):
        self._toggle_btn.config(text=self._arrow() + " " + title)

    def open(self):
        if not self._open.get():
            self._toggle()

    def set_enabled(self, enabled: bool):
        self._toggle_btn.config(state='normal' if enabled else 'disabled')
        state = 'normal' if enabled else 'disabled'
        def _set(w):
            try:
                w.configure(state=state)
            except tk.TclError:
                pass
            for c in w.winfo_children():
                _set(c)
        for c in self.body.winfo_children():
            _set(c)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class StudioApp:
    def __init__(self, root):
        self.root = root
        root.title("Drum + Bass Humanization Studio")
        root.geometry("900x820")
        root.minsize(720, 560)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.drum_path: Optional[str] = None
        self.bass_path: Optional[str] = None
        self.segments: List[SegmentSettings] = []
        self.selected_index: Optional[int] = None
        self.segment_click_targets: Dict[int, Dict] = {}
        self._restoring = False   # guard: True while pushing a segment's saved
                                  # values into the widgets, so those widget
                                  # callbacks don't misfire as user edits

        self.seg_model = None
        self.seg_cfg = None
        self.seg_checkpoint_path: Optional[str] = None
        self.hum_checkpoint_path: Optional[str] = None

        self.temp_dir = tempfile.mkdtemp(prefix='drum_bass_studio_')
        self.player = gfx.MidiPlayer() if HAS_GFX else None
        self.playing_path: Optional[str] = None

        self._build_widgets()
        self._refresh_gating()

    # ------------------------------------------------------------------ UI --
    def _build_widgets(self):
        # -- model loaders --
        top = ttk.Frame(self.root); top.pack(fill='x', padx=8, pady=(8, 2))
        ttk.Label(top, text="Humanizer model:").pack(side='left')
        self.hum_model_label = ttk.Label(top, text="(none loaded)", foreground='gray')
        self.hum_model_label.pack(side='left', padx=6)
        ttk.Button(top, text="Load...", command=self._on_load_hum_model).pack(side='right')

        top2 = ttk.Frame(self.root); top2.pack(fill='x', padx=8, pady=(0, 6))
        ttk.Label(top2, text="Segmentation model:").pack(side='left')
        self.seg_model_label = ttk.Label(top2, text="(none loaded)", foreground='gray')
        self.seg_model_label.pack(side='left', padx=6)
        ttk.Button(top2, text="Load...", command=self._on_load_seg_model).pack(side='right')

        # -- drum drop zone --
        ttk.Label(self.root, text="Drum MIDI (full song):").pack(anchor='w', padx=8)
        self.drum_drop = tk.Label(self.root, text=self._drop_text("drum"), relief='groove',
                                  bd=2, height=2, bg='#f5f5f5', fg='#555', cursor='hand2')
        self.drum_drop.pack(fill='x', padx=8, pady=(0, 4))
        self.drum_drop.bind('<Button-1>', self._on_browse_drum)

        # -- segment timeline --
        self.seg_canvas = tk.Canvas(self.root, height=0, bg='#e8e8e8', highlightthickness=0)
        self.seg_canvas.pack(fill='x', padx=8, pady=(0, 4))
        self.seg_label = ttk.Label(self.root, text="", foreground='gray')
        self.seg_label.pack(anchor='w', padx=8)

        # -- bass drop zone --
        ttk.Label(self.root, text="Bass MIDI (matching song, same tempo/alignment):").pack(anchor='w', padx=8, pady=(6, 0))
        self.bass_drop = tk.Label(self.root, text=self._drop_text("bass"), relief='groove',
                                  bd=2, height=2, bg='#f5f5f5', fg='#555', cursor='hand2')
        self.bass_drop.pack(fill='x', padx=8, pady=(0, 6))
        self.bass_drop.bind('<Button-1>', self._on_browse_bass)

        if HAS_DND:
            for widget, kind in ((self.drum_drop, 'drum'), (self.bass_drop, 'bass')):
                try:
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind('<<Drop>>', lambda e, k=kind: self._on_drop(e, k))
                except Exception as exc:
                    _report_error("enabling drag-and-drop (falling back to click-to-browse)", exc)

        ttk.Separator(self.root).pack(fill='x', padx=8, pady=4)

        # -- body holding the three phase sections --
        outer = ttk.Frame(self.root); outer.pack(fill='both', expand=True, padx=8)
        self.phase1_section = CollapsibleSection(outer, "Phase 1 -- Drum Humanize", start_open=PHASE1_SECTION_STARTS_OPEN)
        self.phase1_section.pack(fill='x')
        self.phase2_section = CollapsibleSection(outer, "Phase 2 -- Rush / Drag", start_open=PHASE2_SECTION_STARTS_OPEN)
        self.phase2_section.pack(fill='x')
        self.phase3_section = CollapsibleSection(outer, "Phase 3 -- Bass Sync", start_open=PHASE3_SECTION_STARTS_OPEN)
        self.phase3_section.pack(fill='x')
        self._build_phase1_controls(self.phase1_section.body)
        self._build_phase2_controls(self.phase2_section.body)
        self._build_phase3_controls(self.phase3_section.body)

        ttk.Separator(self.root).pack(fill='x', padx=8, pady=4)
        bottom = ttk.Frame(self.root); bottom.pack(fill='x', padx=8, pady=(0, 8))
        ttk.Button(bottom, text="Reset segment", command=self._on_reset_segment).pack(side='left')
        ttk.Button(bottom, text="Render Song...", command=self._on_render_song).pack(side='right')
        self.status_label = ttk.Label(bottom, text="Load models, drop a drum MIDI to begin.", foreground='gray')
        self.status_label.pack(side='left', padx=10)

    def _drop_text(self, kind):
        base = f"Drop {kind} MIDI here" if HAS_DND else f"Click to browse for {kind} MIDI"
        return base + ("\n(or click to browse)" if HAS_DND else "\n(drag-and-drop needs tkinterdnd2)")

    # ------------------------------------------------------------- models --
    def _on_load_hum_model(self):
        path = filedialog.askopenfilename(title="Select humanizer checkpoint",
                                          filetypes=[("Checkpoint", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        if not HAS_HUMANIZER:
            messagebox.showerror("Missing module", "drum_humanizer_v3.py not found alongside this script.")
            return
        self.hum_checkpoint_path = path
        self.hum_model_label.config(text=os.path.basename(path), foreground='black')
        self._set_status(f"Humanizer model set: {os.path.basename(path)}")

    def _on_load_seg_model(self):
        path = filedialog.askopenfilename(title="Select segmentation checkpoint",
                                          filetypes=[("Checkpoint", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        if not HAS_SEGMENTATION:
            messagebox.showerror("Missing module", "drum_theme_segmentation.py not found alongside this script.")
            return
        self._set_status("Loading segmentation model...", busy=True)
        threading.Thread(target=self._load_seg_model_worker, args=(path,), daemon=True).start()

    def _load_seg_model_worker(self, path):
        try:
            device = dts.torch.device('cuda' if dts.torch.cuda.is_available() else 'cpu')
            model, cfg = dts.load_model(path, device)
        except Exception as exc:
            msg = _report_error(f"loading segmentation model '{path}'", exc)
            self.root.after(0, lambda: messagebox.showerror("Failed to load model", msg))
            return
        self.root.after(0, lambda: self._on_seg_model_loaded(path, model, cfg))

    def _on_seg_model_loaded(self, path, model, cfg):
        self.seg_model = model
        self.seg_cfg = cfg
        self.seg_checkpoint_path = path
        self.seg_model_label.config(text=os.path.basename(path), foreground='black')
        self._set_status(f"Segmentation model loaded: {os.path.basename(path)}")

    # ---------------------------------------------------------- drop zones --
    def _on_browse_drum(self, event=None):
        path = filedialog.askopenfilename(title="Select drum MIDI",
                                          filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")])
        if path:
            self._load_drum_file(path)

    def _on_browse_bass(self, event=None):
        path = filedialog.askopenfilename(title="Select bass MIDI",
                                          filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")])
        if path:
            self._load_bass_file(path)

    def _on_drop(self, event, kind):
        raw = getattr(event, 'data', None)
        if not raw:
            return
        paths = self.root.tk.splitlist(raw)
        if not paths:
            return
        if kind == 'drum':
            self._load_drum_file(paths[0])
        else:
            self._load_bass_file(paths[0])

    def _load_bass_file(self, path):
        self.bass_path = path
        self.bass_drop.config(text=os.path.basename(path), foreground='black')
        self._set_status(f"Bass file set: {os.path.basename(path)}")

    def _load_drum_file(self, path):
        if self.seg_model is None:
            messagebox.showwarning("No segmentation model",
                                   "Load a segmentation model first (top of window).")
            return
        self.drum_path = path
        self.drum_drop.config(text=os.path.basename(path), foreground='black')
        self.selected_index = None
        self.segments = []
        self.segment_click_targets = {}
        self.seg_canvas.delete('all')
        self._set_status(f"Segmenting '{os.path.basename(path)}'...", busy=True)
        threading.Thread(target=self._segment_worker, args=(path,), daemon=True).start()

    def _segment_worker(self, path):
        try:
            result = dts.compute_segment_boundaries(self.seg_model, self.seg_cfg, path,
                                                     threshold=SEGMENTATION_CONFIDENCE_THRESHOLD,
                                                     context_overlap=SEGMENTATION_CONTEXT_OVERLAP)
            starts = result['starts']
            total_measures = result['total_measures']
            midi = result['midi']
            segs = []
            for i, start_bar in enumerate(starts):
                end_bar = starts[i + 1] if i + 1 < len(starts) else total_measures
                if end_bar <= start_bar:
                    continue
                start_sec = dts.bar_to_seconds(start_bar, midi, self.seg_cfg)
                end_sec = dts.bar_to_seconds(end_bar, midi, self.seg_cfg)
                segs.append(SegmentSettings(index=len(segs), start_sec=start_sec, end_sec=end_sec,
                                            start_bar=start_bar, end_bar=end_bar))
        except Exception as exc:
            msg = _report_error(f"segmenting '{path}'", exc)
            self.root.after(0, lambda: messagebox.showerror("Segmentation failed", msg))
            return
        self.root.after(0, lambda: self._on_segmentation_done(path, segs))

    def _on_segmentation_done(self, path, segs):
        if path != self.drum_path:
            return   # a newer file was dropped before this finished
        self.segments = segs
        self._draw_timeline()
        if not segs:
            self._set_status("No segments detected.")
            return
        self._set_status(f"{len(segs)} segments detected -- click one to begin.")

    # ------------------------------------------------------------ timeline --
    def _draw_timeline(self):
        self.seg_canvas.delete('all')
        self.segment_click_targets = {}
        if not self.segments:
            self.seg_canvas.config(height=0)
            return
        self.seg_canvas.config(height=44)
        self.root.update_idletasks()
        canvas_w = max(200, self.seg_canvas.winfo_width())
        total_bars = sum(s.end_bar - s.start_bar for s in self.segments)
        if total_bars <= 0:
            return
        x = 0
        for i, seg in enumerate(self.segments):
            length = seg.end_bar - seg.start_bar
            w = max(SEGMENT_MIN_RECT_WIDTH_PX, round(canvas_w * length / total_bars))
            color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
            selected = (i == self.selected_index)
            rect = self.seg_canvas.create_rectangle(
                x, 2, x + w, 40, fill=color,
                outline=SELECTED_OUTLINE_COLOR if selected else 'white',
                width=SEGMENT_OUTLINE_WIDTH_SELECTED if selected else SEGMENT_OUTLINE_WIDTH_NORMAL,
                tags=(f'seg{i}',))
            label = f"{seg.start_bar+1}-{seg.end_bar}"
            if w > 28:
                self.seg_canvas.create_text(x + w / 2, 21, text=label, fill='white',
                                            font=('', 8), tags=(f'seg{i}',))
            if seg.is_customized():
                self.seg_canvas.create_oval(x + w - 10, 4, x + w - 2, 12,
                                            fill=CUSTOMIZED_MARKER_COLOR, outline='',
                                            tags=(f'seg{i}',))
            self.seg_canvas.tag_bind(f'seg{i}', '<Button-1>', lambda e, idx=i: self._on_segment_selected(idx))
            self.segment_click_targets[i] = {'rect': rect, 'x0': x, 'x1': x + w}
            x += w

    def _on_segment_selected(self, idx):
        self.selected_index = idx
        self._draw_timeline()
        self._restore_segment_to_widgets(self.segments[idx])
        self._refresh_gating()
        seg = self.segments[idx]
        self._set_status(f"Segment {idx+1} selected (measures {seg.start_bar+1}-{seg.end_bar}).")

    # ------------------------------------------------------------- gating --
    def _current_segment(self) -> Optional[SegmentSettings]:
        if self.selected_index is None or self.selected_index >= len(self.segments):
            return None
        return self.segments[self.selected_index]

    def _refresh_gating(self):
        seg = self._current_segment()
        has_seg = seg is not None
        self.phase1_section.set_enabled(has_seg)
        self.phase2_section.set_enabled(has_seg and seg.phase1_done)
        self.phase3_section.set_enabled(has_seg and seg.phase2_done and self.bass_path is not None)
        # staleness hints
        if has_seg:
            self.phase2_stale_label.config(
                text="\u26a0 Phase 1 changed since this was applied -- re-apply to update."
                if seg.phase2_done and seg.phase2_stale else "")
            self.phase3_stale_label.config(
                text="\u26a0 Phase 2 changed since this was applied -- re-apply to update."
                if seg.phase3_done and seg.phase3_stale else "")
        else:
            self.phase2_stale_label.config(text="")
            self.phase3_stale_label.config(text="")

    def _set_status(self, text, busy=False, phase=None):
        prefix = f"[{phase}] " if phase else ""
        self.status_label.config(text=prefix + text, foreground=('#0066cc' if busy else 'gray'))

    # -------------------------------------------------- restore per-segment --
    def _restore_segment_to_widgets(self, seg: SegmentSettings):
        self._restoring = True
        try:
            p1 = seg.phase1_settings
            self.var_strength.set(p1['strength'])
            self.var_intensity_auto.set(p1['intensity'] is None)
            self.var_intensity.set(p1['intensity'] if p1['intensity'] is not None else PHASE1_DEFAULT_INTENSITY_VALUE)
            self.var_feel_variation.set(p1['feel_variation'])
            self.var_fast_hit_cap.set(p1['fast_hit_cap'])

            p2 = seg.phase2_settings
            for grp in ('kick', 'snare', 'toms', 'other'):
                self.var_rush_drag[grp].set(p2[f'rush_drag_{grp}'])
                self.var_quantize[grp].set(p2[f'quantize_{grp}'])

            p3 = seg.phase3_settings
            self.var_snap_strength.set(p3['snap_strength'])
            self.var_delay_amount.set(p3['delay_amount'])
        finally:
            self._restoring = False

    def _save_widgets_to_segment(self, seg: SegmentSettings):
        """Pull current widget values back into the segment's settings dict --
        called right before Running/Applying a phase, so what gets processed
        always matches exactly what's currently shown."""
        seg.phase1_settings = {
            'strength': self.var_strength.get(),
            'intensity': None if self.var_intensity_auto.get() else self.var_intensity.get(),
            'feel_variation': self.var_feel_variation.get(),
            'fast_hit_cap': self.var_fast_hit_cap.get(),
        }
        seg.phase2_settings = {}
        for grp in ('kick', 'snare', 'toms', 'other'):
            seg.phase2_settings[f'rush_drag_{grp}'] = self.var_rush_drag[grp].get()
            seg.phase2_settings[f'quantize_{grp}'] = self.var_quantize[grp].get()
        seg.phase3_settings = {
            'snap_strength': self.var_snap_strength.get(),
            'delay_amount': self.var_delay_amount.get(),
        }

    # -------------------------------------------------------- slider helper --
    def _make_slider_row(self, parent, label, var, frm, to, width_label=16):
        row = ttk.Frame(parent); row.pack(fill='x', padx=8, pady=2)
        ttk.Label(row, text=label, width=width_label, anchor='w').pack(side='left')
        val_label = ttk.Label(row, text=f"{var.get():+.2f}", width=6, anchor='e')
        val_label.pack(side='right')
        def _on_move(v, var=var, val_label=val_label):
            val_label.config(text=f"{var.get():+.2f}")
            if not self._restoring:
                self._on_any_control_changed()
        scale = ttk.Scale(row, from_=frm, to=to, orient='horizontal', variable=var, command=_on_move)
        scale.pack(side='left', fill='x', expand=True, padx=6)
        return scale

    def _on_any_control_changed(self):
        """Live-persist widget edits into the currently selected segment's
        settings immediately (not just when a phase button is pressed) so
        switching segments never silently drops an in-progress tweak."""
        seg = self._current_segment()
        if seg is None:
            return
        self._save_widgets_to_segment(seg)
        self._draw_timeline()   # the "customized" dot may need to appear/disappear

    # ============================================================ PHASE 1 ==
    def _build_phase1_controls(self, parent):
        self.var_strength = tk.DoubleVar(value=PHASE1_DEFAULT_STRENGTH)
        self.var_intensity_auto = tk.BooleanVar(value=PHASE1_DEFAULT_INTENSITY_AUTO)
        self.var_intensity = tk.DoubleVar(value=PHASE1_DEFAULT_INTENSITY_VALUE)
        self.var_feel_variation = tk.DoubleVar(value=PHASE1_DEFAULT_FEEL_VARIATION)
        self.var_fast_hit_cap = tk.BooleanVar(value=PHASE1_DEFAULT_FAST_HIT_CAP)

        self._make_slider_row(parent, "Strength", self.var_strength, *PHASE1_STRENGTH_RANGE)
        self._make_slider_row(parent, "Feel variation", self.var_feel_variation, *PHASE1_FEEL_VARIATION_RANGE)

        row = ttk.Frame(parent); row.pack(fill='x', padx=8, pady=2)
        ttk.Checkbutton(row, text="Auto intensity", variable=self.var_intensity_auto,
                        command=self._on_any_control_changed).pack(side='left')
        self.intensity_scale = self._make_slider_row(parent, "Intensity", self.var_intensity, *PHASE1_INTENSITY_RANGE)

        row2 = ttk.Frame(parent); row2.pack(fill='x', padx=8, pady=2)
        ttk.Checkbutton(row2, text="Fast-hit velocity cap (blast beats)",
                        variable=self.var_fast_hit_cap,
                        command=self._on_any_control_changed).pack(side='left')

        btnrow = ttk.Frame(parent); btnrow.pack(fill='x', padx=8, pady=6)
        self.phase1_run_btn = ttk.Button(btnrow, text="Run Phase 1", command=self._on_run_phase1)
        self.phase1_run_btn.pack(side='left')
        ttk.Button(btnrow, text="Apply to ALL segments", command=self._on_apply_phase1_all).pack(side='left', padx=6)
        ttk.Button(btnrow, text="\u25b6 Audition raw", command=lambda: self._on_audition('phase1_raw')).pack(side='right')
        ttk.Button(btnrow, text="\u25b6 Audition result", command=lambda: self._on_audition('phase1')).pack(side='right', padx=6)
        self.phase1_status = ttk.Label(parent, text="", foreground='gray')
        self.phase1_status.pack(anchor='w', padx=8, pady=(0, 4))

    def _on_run_phase1(self):
        seg = self._current_segment()
        if seg is None:
            return
        if self.hum_checkpoint_path is None or not HAS_HUMANIZER:
            messagebox.showwarning("No humanizer model", "Load the humanizer model first (top of window).")
            return
        if self.drum_path is None:
            return
        self._save_widgets_to_segment(seg)
        self.phase1_status.config(text="Running...", foreground='#0066cc')
        threading.Thread(target=self._run_phase1_worker, args=(seg,), daemon=True).start()

    def _run_phase1_worker(self, seg: SegmentSettings):
        try:
            raw_path, tempo, n = slice_midi_to_temp(self.drum_path, seg.start_sec, seg.end_sec,
                                                     self.temp_dir, drums_only=True)
            out_path = os.path.join(self.temp_dir, f"seg{seg.index}_phase1_{uuid.uuid4().hex[:6]}.mid")
            run_phase1_humanize(self.hum_checkpoint_path, raw_path, out_path, seg.phase1_settings)
        except Exception as exc:
            msg = _report_error(f"running Phase 1 on segment {seg.index+1}", exc)
            self.root.after(0, lambda: self._on_phase1_error(seg, msg))
            return
        self.root.after(0, lambda: self._on_phase1_done(seg, raw_path, out_path))

    def _on_phase1_done(self, seg: SegmentSettings, raw_path, out_path):
        seg.phase1_output_path = out_path
        seg.phase1_raw_path = raw_path
        seg.phase1_done = True
        seg.phase1_stale = False
        # downstream phases are now stale relative to this fresh output
        if seg.phase2_done:
            seg.phase1_stale = False
            seg.phase2_stale = True
        self.phase1_status.config(text=f"Done ({os.path.basename(out_path)}).", foreground='green')
        self._refresh_gating()
        self._draw_timeline()

    def _on_phase1_error(self, seg, msg):
        self.phase1_status.config(text="Failed (see console).", foreground='#b00000')
        messagebox.showerror("Phase 1 failed", msg)

    def _on_apply_phase1_all(self):
        if not self.segments:
            return
        if self.hum_checkpoint_path is None or not HAS_HUMANIZER:
            messagebox.showwarning("No humanizer model", "Load the humanizer model first (top of window).")
            return
        seg = self._current_segment()
        if seg is None:
            messagebox.showinfo("Select a segment", "Select and configure a segment first, "
                                "then Apply to ALL will copy its Phase 1 settings to every segment.")
            return
        self._save_widgets_to_segment(seg)
        settings_copy = dict(seg.phase1_settings)
        n = len(self.segments)
        if not messagebox.askyesno("Apply Phase 1 to all segments?",
                                   f"This will run the humanizer model on all {n} segments "
                                   f"using segment {seg.index+1}'s current settings. This can "
                                   f"take a while for a long song. Continue?"):
            return
        self._set_status(f"Applying Phase 1 to all {n} segments...", busy=True, phase="Phase 1")
        threading.Thread(target=self._apply_phase1_all_worker, args=(settings_copy,), daemon=True).start()

    def _apply_phase1_all_worker(self, settings_copy):
        for s in self.segments:
            s.phase1_settings = dict(settings_copy)
            try:
                raw_path, tempo, n = slice_midi_to_temp(self.drum_path, s.start_sec, s.end_sec,
                                                         self.temp_dir, drums_only=True)
                out_path = os.path.join(self.temp_dir, f"seg{s.index}_phase1_{uuid.uuid4().hex[:6]}.mid")
                run_phase1_humanize(self.hum_checkpoint_path, raw_path, out_path, s.phase1_settings)
                s.phase1_output_path = out_path
                s.phase1_raw_path = raw_path
                s.phase1_done = True
                s.phase1_stale = False
                if s.phase2_done:
                    s.phase2_stale = True
            except Exception as exc:
                _report_error(f"applying Phase 1 to segment {s.index+1} (apply-to-all)", exc)
            self.root.after(0, lambda idx=s.index: self._set_status(
                f"Applying Phase 1: segment {idx+1}/{len(self.segments)} done...", busy=True, phase="Phase 1"))
        self.root.after(0, self._on_apply_all_done)

    def _on_apply_all_done(self):
        self._set_status("Apply-to-all finished.")
        if self._current_segment() is not None:
            self._restore_segment_to_widgets(self._current_segment())
        self._refresh_gating()
        self._draw_timeline()

    # ============================================================ PHASE 2 ==
    def _build_phase2_controls(self, parent):
        self.var_rush_drag = {}
        self.var_quantize = {}
        ttk.Label(parent, text="Rush / Drag  (-1 = drag before the grid, "
                               "+1 = rush after the grid; e/a full strength, "
                               "2 & 4 at 75%)", foreground='gray').pack(anchor='w', padx=8, pady=(4, 0))
        for grp, label in (('kick', 'Kick'), ('snare', 'Snare'), ('toms', 'Toms'), ('other', 'Everything else')):
            var = tk.DoubleVar(value=PHASE2_DEFAULT_RUSH_DRAG)
            self.var_rush_drag[grp] = var
            self._make_slider_row(parent, label, var, *PHASE2_RUSH_DRAG_RANGE)

        ttk.Separator(parent).pack(fill='x', padx=8, pady=4)
        ttk.Label(parent, text="Quantize the 1 & 3  (0 = untouched, "
                               "1 = snapped exactly to the grid)", foreground='gray').pack(anchor='w', padx=8)
        for grp, label in (('kick', 'Kick'), ('snare', 'Snare'), ('toms', 'Toms'), ('other', 'Everything else')):
            var = tk.DoubleVar(value=PHASE2_DEFAULT_QUANTIZE)
            self.var_quantize[grp] = var
            self._make_slider_row(parent, label, var, *PHASE2_QUANTIZE_RANGE)

        btnrow = ttk.Frame(parent); btnrow.pack(fill='x', padx=8, pady=6)
        ttk.Button(btnrow, text="Apply Phase 2", command=self._on_apply_phase2).pack(side='left')
        ttk.Button(btnrow, text="Apply to ALL segments", command=self._on_apply_phase2_all).pack(side='left', padx=6)
        ttk.Button(btnrow, text="\u25b6 Audition result", command=lambda: self._on_audition('phase2')).pack(side='right')
        self.phase2_status = ttk.Label(parent, text="", foreground='gray')
        self.phase2_status.pack(anchor='w', padx=8)
        self.phase2_stale_label = ttk.Label(parent, text="", foreground='#b06a00')
        self.phase2_stale_label.pack(anchor='w', padx=8, pady=(0, 4))

    def _on_apply_phase2(self):
        seg = self._current_segment()
        if seg is None or not seg.phase1_done:
            return
        self._save_widgets_to_segment(seg)
        self.phase2_status.config(text="Applying...", foreground='#0066cc')
        threading.Thread(target=self._apply_phase2_worker, args=(seg,), daemon=True).start()

    def _apply_phase2_worker(self, seg: SegmentSettings):
        try:
            out_path = os.path.join(self.temp_dir, f"seg{seg.index}_phase2_{uuid.uuid4().hex[:6]}.mid")
            cfg = dhu.Config()
            run_phase2_adjust(seg.phase1_output_path, out_path, seg.phase2_settings, cfg=cfg)
        except Exception as exc:
            msg = _report_error(f"applying Phase 2 to segment {seg.index+1}", exc)
            self.root.after(0, lambda: self._on_phase2_error(msg))
            return
        self.root.after(0, lambda: self._on_phase2_done(seg, out_path))

    def _on_phase2_done(self, seg: SegmentSettings, out_path):
        seg.phase2_output_path = out_path
        seg.phase2_done = True
        seg.phase2_stale = False
        seg.phase1_stale = False
        if seg.phase3_done:
            seg.phase3_stale = True
        self.phase2_status.config(text=f"Done ({os.path.basename(out_path)}).", foreground='green')
        self._refresh_gating()
        self._draw_timeline()

    def _on_phase2_error(self, msg):
        self.phase2_status.config(text="Failed (see console).", foreground='#b00000')
        messagebox.showerror("Phase 2 failed", msg)

    def _on_apply_phase2_all(self):
        ready = [s for s in self.segments if s.phase1_done]
        if not ready:
            messagebox.showinfo("Nothing to apply", "Run Phase 1 on at least one segment first.")
            return
        seg = self._current_segment()
        if seg is None or not seg.phase1_done:
            messagebox.showinfo("Select a segment", "Select a Phase-1-completed segment first, "
                                "then Apply to ALL will copy its Phase 2 settings to every "
                                "Phase-1-completed segment.")
            return
        self._save_widgets_to_segment(seg)
        settings_copy = dict(seg.phase2_settings)
        if not messagebox.askyesno("Apply Phase 2 to all segments?",
                                   f"This will apply these rush/drag + quantize settings to all "
                                   f"{len(ready)} segments that have completed Phase 1. Continue?"):
            return
        self._set_status(f"Applying Phase 2 to {len(ready)} segments...", busy=True, phase="Phase 2")
        threading.Thread(target=self._apply_phase2_all_worker, args=(ready, settings_copy), daemon=True).start()

    def _apply_phase2_all_worker(self, ready, settings_copy):
        cfg = dhu.Config()
        for s in ready:
            s.phase2_settings = dict(settings_copy)
            try:
                out_path = os.path.join(self.temp_dir, f"seg{s.index}_phase2_{uuid.uuid4().hex[:6]}.mid")
                run_phase2_adjust(s.phase1_output_path, out_path, s.phase2_settings, cfg=cfg)
                s.phase2_output_path = out_path
                s.phase2_done = True
                s.phase2_stale = False
                if s.phase3_done:
                    s.phase3_stale = True
            except Exception as exc:
                _report_error(f"applying Phase 2 to segment {s.index+1} (apply-to-all)", exc)
        self.root.after(0, self._on_apply_all_done)

    # ============================================================ PHASE 3 ==
    def _build_phase3_controls(self, parent):
        self.var_snap_strength = tk.DoubleVar(value=PHASE3_DEFAULT_SNAP_STRENGTH)
        self.var_delay_amount = tk.DoubleVar(value=PHASE3_DEFAULT_DELAY_AMOUNT)
        self._make_slider_row(parent, "Snap to kick/snare", self.var_snap_strength, *PHASE3_SNAP_STRENGTH_RANGE)
        self._make_slider_row(parent, "Audibility delay", self.var_delay_amount, *PHASE3_DELAY_AMOUNT_RANGE)

        btnrow = ttk.Frame(parent); btnrow.pack(fill='x', padx=8, pady=6)
        ttk.Button(btnrow, text="Apply Phase 3", command=self._on_apply_phase3).pack(side='left')
        ttk.Button(btnrow, text="Apply to ALL segments", command=self._on_apply_phase3_all).pack(side='left', padx=6)
        ttk.Button(btnrow, text="\u25b6 Audition result", command=lambda: self._on_audition('phase3')).pack(side='right')
        self.phase3_status = ttk.Label(parent, text="", foreground='gray')
        self.phase3_status.pack(anchor='w', padx=8)
        self.phase3_stale_label = ttk.Label(parent, text="", foreground='#b06a00')
        self.phase3_stale_label.pack(anchor='w', padx=8, pady=(0, 4))

    def _on_apply_phase3(self):
        seg = self._current_segment()
        if seg is None or not seg.phase2_done or self.bass_path is None:
            return
        self._save_widgets_to_segment(seg)
        self.phase3_status.config(text="Applying...", foreground='#0066cc')
        threading.Thread(target=self._apply_phase3_worker, args=(seg,), daemon=True).start()

    def _apply_phase3_worker(self, seg: SegmentSettings):
        try:
            bass_raw, tempo, n = slice_midi_to_temp(self.bass_path, seg.start_sec, seg.end_sec,
                                                     self.temp_dir, drums_only=False)
            out_path = os.path.join(self.temp_dir, f"seg{seg.index}_phase3_{uuid.uuid4().hex[:6]}.mid")
            rng = random.Random(GLOBAL_SEED + seg.index)
            run_phase3_sync(seg.phase2_output_path, bass_raw, out_path, seg.phase3_settings,
                            tempo=tempo, rng=rng)
        except Exception as exc:
            msg = _report_error(f"applying Phase 3 to segment {seg.index+1}", exc)
            self.root.after(0, lambda: self._on_phase3_error(msg))
            return
        self.root.after(0, lambda: self._on_phase3_done(seg, bass_raw, out_path))

    def _on_phase3_done(self, seg: SegmentSettings, bass_raw, out_path):
        seg.bass_input_path = bass_raw
        seg.phase3_output_path = out_path
        seg.phase3_done = True
        seg.phase3_stale = False
        seg.phase2_stale = False
        self.phase3_status.config(text=f"Done ({os.path.basename(out_path)}).", foreground='green')
        self._refresh_gating()
        self._draw_timeline()

    def _on_phase3_error(self, msg):
        self.phase3_status.config(text="Failed (see console).", foreground='#b00000')
        messagebox.showerror("Phase 3 failed", msg)

    def _on_apply_phase3_all(self):
        ready = [s for s in self.segments if s.phase2_done]
        if not ready or self.bass_path is None:
            messagebox.showinfo("Nothing to apply", "Run Phase 2 on at least one segment (and "
                                "load a bass file) first.")
            return
        seg = self._current_segment()
        if seg is None or not seg.phase2_done:
            messagebox.showinfo("Select a segment", "Select a Phase-2-completed segment first, "
                                "then Apply to ALL will copy its Phase 3 settings to every "
                                "Phase-2-completed segment.")
            return
        self._save_widgets_to_segment(seg)
        settings_copy = dict(seg.phase3_settings)
        if not messagebox.askyesno("Apply Phase 3 to all segments?",
                                   f"This will sync the bass to drums using these settings for "
                                   f"all {len(ready)} segments that have completed Phase 2. Continue?"):
            return
        self._set_status(f"Applying Phase 3 to {len(ready)} segments...", busy=True, phase="Phase 3")
        threading.Thread(target=self._apply_phase3_all_worker, args=(ready, settings_copy), daemon=True).start()

    def _apply_phase3_all_worker(self, ready, settings_copy):
        for s in ready:
            s.phase3_settings = dict(settings_copy)
            try:
                bass_raw, tempo, n = slice_midi_to_temp(self.bass_path, s.start_sec, s.end_sec,
                                                         self.temp_dir, drums_only=False)
                out_path = os.path.join(self.temp_dir, f"seg{s.index}_phase3_{uuid.uuid4().hex[:6]}.mid")
                rng = random.Random(GLOBAL_SEED + s.index)
                run_phase3_sync(s.phase2_output_path, bass_raw, out_path, s.phase3_settings,
                                tempo=tempo, rng=rng)
                s.bass_input_path = bass_raw
                s.phase3_output_path = out_path
                s.phase3_done = True
                s.phase3_stale = False
            except Exception as exc:
                _report_error(f"applying Phase 3 to segment {s.index+1} (apply-to-all)", exc)
        self.root.after(0, self._on_apply_all_done)

    # -------------------------------------------------------------- audition --
    def _on_audition(self, which):
        seg = self._current_segment()
        if seg is None or self.player is None:
            return
        path = {
            'phase1_raw': getattr(seg, 'phase1_raw_path', None),
            'phase1': seg.phase1_output_path,
            'phase2': seg.phase2_output_path,
            'phase3': seg.phase3_output_path,
        }.get(which)
        if not path or not os.path.exists(path):
            messagebox.showinfo("Nothing to play", "This phase hasn't produced output for "
                                "this segment yet.")
            return
        self.playing_path = path
        self._set_status(f"Playing {which} preview...")
        threading.Thread(target=self._audition_worker, args=(path,), daemon=True).start()

    def _audition_worker(self, path):
        try:
            self.player.play(path, on_finished=lambda err: self.root.after(
                0, lambda: self._set_status(f"Playback error: {err}" if err else "Ready.")))
        except Exception as exc:
            msg = _report_error(f"auditioning '{path}'", exc)
            self.root.after(0, lambda: self._set_status(msg))

    # ---------------------------------------------------------------- reset --
    def _on_reset_segment(self):
        seg = self._current_segment()
        if seg is None:
            return
        if not messagebox.askyesno("Reset segment?",
                                   f"Reset segment {seg.index+1} to defaults? This clears all "
                                   f"three phases' results for this segment."):
            return
        seg.reset()
        self._restore_segment_to_widgets(seg)
        self._refresh_gating()
        self._draw_timeline()
        self._set_status(f"Segment {seg.index+1} reset.")

    # --------------------------------------------------------------- render --
    def _on_render_song(self):
        if not self.segments:
            return
        missing = [s.index + 1 for s in self.segments if not s.phase2_done]
        if missing:
            if not messagebox.askyesno("Some segments incomplete",
                                       f"Segment(s) {missing} haven't completed Phase 2 -- "
                                       f"their ORIGINAL (unprocessed) drum audio will be used "
                                       f"for those spans, and they'll be skipped in the bass "
                                       f"output. Continue?"):
                return
        drum_out = filedialog.asksaveasfilename(title="Save processed DRUM MIDI as",
                                                defaultextension=".mid",
                                                filetypes=[("MIDI files", "*.mid")])
        if not drum_out:
            return
        bass_out = None
        if self.bass_path is not None:
            bass_out = filedialog.asksaveasfilename(title="Save processed BASS MIDI as",
                                                    defaultextension=".mid",
                                                    filetypes=[("MIDI files", "*.mid")])
        self._set_status("Rendering...", busy=True)
        threading.Thread(target=self._render_worker, args=(drum_out, bass_out), daemon=True).start()

    def _render_worker(self, drum_out, bass_out):
        try:
            out_drum = pretty_midi.PrettyMIDI(initial_tempo=120)
            drum_inst = pretty_midi.Instrument(program=0, is_drum=True)
            out_bass = pretty_midi.PrettyMIDI(initial_tempo=120) if bass_out else None
            bass_inst = pretty_midi.Instrument(program=33, is_drum=False) if bass_out else None

            for seg in self.segments:
                drum_src_path = seg.phase2_output_path or seg.phase1_output_path
                if drum_src_path and os.path.exists(drum_src_path):
                    src = pretty_midi.PrettyMIDI(drum_src_path)
                    for inst in src.instruments:
                        for n in inst.notes:
                            drum_inst.notes.append(pretty_midi.Note(
                                velocity=n.velocity, pitch=n.pitch,
                                start=n.start + seg.start_sec, end=n.end + seg.start_sec))
                elif os.path.exists(self.drum_path):
                    raw_path, _, _ = slice_midi_to_temp(self.drum_path, seg.start_sec, seg.end_sec,
                                                        self.temp_dir, drums_only=True)
                    src = pretty_midi.PrettyMIDI(raw_path)
                    for inst in src.instruments:
                        for n in inst.notes:
                            drum_inst.notes.append(pretty_midi.Note(
                                velocity=n.velocity, pitch=n.pitch,
                                start=n.start + seg.start_sec, end=n.end + seg.start_sec))

                if bass_out and seg.phase3_output_path and os.path.exists(seg.phase3_output_path):
                    src = pretty_midi.PrettyMIDI(seg.phase3_output_path)
                    for inst in src.instruments:
                        for n in inst.notes:
                            bass_inst.notes.append(pretty_midi.Note(
                                velocity=n.velocity, pitch=n.pitch,
                                start=n.start + seg.start_sec, end=n.end + seg.start_sec))

            drum_inst.notes.sort(key=lambda n: n.start)
            out_drum.instruments.append(drum_inst)
            out_drum.write(drum_out)
            if bass_out:
                bass_inst.notes.sort(key=lambda n: n.start)
                out_bass.instruments.append(bass_inst)
                out_bass.write(bass_out)
        except Exception as exc:
            msg = _report_error("rendering the final song", exc)
            self.root.after(0, lambda: messagebox.showerror("Render failed", msg))
            return
        self.root.after(0, lambda: self._on_render_done(drum_out, bass_out))

    def _on_render_done(self, drum_out, bass_out):
        self._set_status(f"Rendered -> {os.path.basename(drum_out)}"
                         + (f" + {os.path.basename(bass_out)}" if bass_out else ""))
        messagebox.showinfo("Render complete", f"Drum: {drum_out}" +
                            (f"\nBass: {bass_out}" if bass_out else ""))

    # ---------------------------------------------------------------- close --
    def _on_close(self):
        try:
            if self.player:
                self.player.close()
        except Exception:
            pass
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
        self.root.destroy()


def main():
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
        print("(tkinterdnd2 not installed -- drag-and-drop disabled, "
              "click-to-browse still works: pip install tkinterdnd2)")
    StudioApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
