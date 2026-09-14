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

  Phase 1  DRUM HUMANIZE     - run the trained humanizer model on this segment.
  Phase 2  RUSH / DRAG       - manually nudge syncopated hits earlier/later,
                                and optionally pull the strong beats back to
                                the grid, per instrument group.
  Phase 3  BASS SYNC         - lock nearby bass notes to the (now-humanized)
                                kick/snare, with a small randomized delay so
                                the two transients stay audibly distinct.

Every phase's settings are remembered PER SEGMENT - select a different
segment, adjust it, come back, and your earlier settings are exactly as you
left them. Phases are gated (Phase 2 needs Phase 1's output to adjust; Phase 3
needs Phase 2's output to sync against) but NEVER re-lock: revisit an earlier
phase, change something, and every downstream phase recomputes from the new
result the next time you preview or render it.

REUSED FROM THE OTHER PROJECT SCRIPTS (see their own docstrings for detail)
------------------------------------------------------------------------------
  drum_humanizer_v3.py        - Phase 1's model (humanize_file), DrumEvent,
                                 the MIDI grid/Config, load/write helpers.
  drum_theme_segmentation.py  - the segmentation model (compute_segment_
                                 boundaries, bar_to_seconds).
  find_similar_grooves.py     - groove-library index + fingerprint/similarity,
                                 used for the per-segment groove search.
The MIDI playback engine (MidiPlayer) below is shared with what used to be a
separate groove_finder_ui.py standalone app, now folded into this file.
Phase 2 and Phase 3's actual signal processing, and the whole window, are new.

HOW IT IS USED
---------------
  pip install torch pretty_midi mido numpy
  python drum_bass_studio.py
Then, in the window: drop a drum MIDI (top), drop the matching bass MIDI
(below it), click a segment, work through Phase 1 -> 2 -> 3, repeat for other
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
import zipfile
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
    import mido
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False

try:
    from tkinterdnd2 import DND_FILES, COPY, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

# sibling project scripts (config.py, drum_humanizer_v3.py, etc.) live in modules/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules'))
try:
    import drum_humanizer_v3 as dhu
    HAS_HUMANIZER = True
except ImportError:
    HAS_HUMANIZER = False

try:
    from download_pretrained import ensure_pretrained_model, pretrained_models_missing
    HAS_PRETRAINED_DOWNLOADER = True
except ImportError:
    HAS_PRETRAINED_DOWNLOADER = False

try:
    import drum_theme_segmentation as dts
    HAS_SEGMENTATION = True
except ImportError:
    HAS_SEGMENTATION = False

try:
    import find_similar_grooves as fsg   # groove-library index + fingerprint/similarity
    HAS_GROOVE_FINDER = True
except ImportError:
    HAS_GROOVE_FINDER = False

# DESIGN: resolved relative to THIS FILE's directory (the project root), not
# the current working directory - so the bundled index is found regardless of
# where this script is launched from.
DEFAULT_GROOVE_INDEX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'cache', 'groove_index.pkl')
GROOVE_SEARCH_TOP_K = 20  # candidates cached per segment, shown in the results list
SWAPPED_GROOVE_MARKER_COLOR = '#FFFFFF'   # timeline marker: this segment's source
                                          # audio was swapped for a library groove -
                                          # white (+ outline) reads clearly against
                                          # every SEGMENT_COLORS entry, unlike a
                                          # picked hue that could clash with one

SESSION_FILE_EXT = '.dbhproj'
SESSION_FILE_VERSION = 1   # bump whenever the session.json schema changes incompatibly


# =============================================================================
# CONSTANTS
# =============================================================================
# Every constant lives in config.py (kept alongside this script) - see that
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
# MIDI PLAYBACK ENGINE  (formerly groove_finder_ui.py's MidiPlayer/_tempo_at_time)
# =============================================================================
# DESIGN: the output port is an INJECTED dependency (anything with a .send(msg)
# method works) specifically so this class can be unit-tested with a mock
# recorder instead of real MIDI hardware. On a real Windows machine,
# `mido.open_output()` with no argument name opens the system default port,
# which is the built-in GS Wavetable Synth.

def _tempo_at_time(pm, t: float) -> float:
    """Return the tempo (bpm) actually in effect at absolute time t, tempo-change
    aware - the LAST tempo change at or before t, not blindly the file's first."""
    try:
        times, tempi = pm.get_tempo_changes()
    except Exception:
        return 120.0
    if len(tempi) == 0:
        return 120.0
    idx = 0
    for i, ct in enumerate(times):
        if ct <= t:
            idx = i
        else:
            break
    return float(tempi[idx])


class MidiPlayer:
    """Plays one MIDI file at a time on a background thread. Forces all note
    events onto MIDI channel 10 (GM drum channel) so playback always uses the
    synth's drum kit sounds regardless of the source file's original channel -
    SD3-exported grooves are drums-only files, so this is always correct here."""

    DRUM_CHANNEL = 9   # 0-indexed == "channel 10" in 1-indexed MIDI terminology

    def __init__(self, outport_factory: Optional[Callable[[], object]] = None):
        # outport_factory: zero-arg callable returning an object with .send(msg).
        # Defaults to mido's real default output (the Windows built-in synth).
        self._outport_factory = outport_factory or self._default_outport_factory
        self._outport = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.is_playing = False

    @staticmethod
    def _default_outport_factory():
        if not HAS_MIDO:
            raise RuntimeError("mido is not installed - run: pip install mido python-rtmidi")
        try:
            names = mido.get_output_names()
        except Exception as exc:
            raise RuntimeError(f"could not list MIDI output ports ({exc}). Is "
                               f"python-rtmidi installed? (pip install python-rtmidi)")
        if not names:
            raise RuntimeError("no MIDI output ports found on this system - "
                               "Windows should always have 'Microsoft GS Wavetable "
                               "Synth'; check Windows Sound settings / MIDI devices.")
        # DESIGN: prefer the built-in Windows synth by name if it's present, rather
        # than trusting whichever port happens to enumerate first - a machine with
        # other MIDI hardware/software installed could otherwise route audition
        # to something unexpected (or silent).
        preferred = next((n for n in names if 'gs wavetable' in n.lower()), None)
        return mido.open_output(preferred or names[0])

    def _ensure_port(self):
        if self._outport is None:
            self._outport = self._outport_factory()
        return self._outport

    def play(self, path: str, on_finished: Optional[Callable[[Optional[str]], None]] = None,
             tempo_scale: float = 1.0, force_drum_channel: bool = True):
        """Start playback in the background. on_finished(error_message_or_None)
        is called (from the playback thread) when playback ends, whether by
        completing naturally, being stopped, or erroring.
        tempo_scale: uniformly speeds up (>1) or slows down (<1) the WHOLE
        performance's timing - e.g. 1.5 plays 50% faster. Used to retime a
        library groove to the query's tempo when auditioning it; leave at 1.0
        (default) to play a file at its own native tempo, unchanged.
        force_drum_channel: this class's ORIGINAL purpose (see class docstring)
        was auditioning drums-ONLY grooves, so every note is forced onto the
        drum channel unconditionally by default. Pass False for a file that
        carries its own correct per-instrument channel/program (e.g. a bass
        part, or a mixed drum+bass file already using is_drum correctly) -
        forcing it to the drum channel would make it play with a drum kit
        sound regardless of its actual instrument."""
        if self.is_playing:
            self.stop()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
        if tempo_scale is None or tempo_scale <= 0:
            tempo_scale = 1.0   # defensive: a bad ratio must never divide-by-zero or reverse time
        self._stop_event.clear()
        self.is_playing = True
        self._thread = threading.Thread(target=self._play_worker,
                                        args=(path, on_finished, tempo_scale, force_drum_channel),
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _play_worker(self, path: str, on_finished, tempo_scale: float = 1.0,
                     force_drum_channel: bool = True):
        error = None
        try:
            port = self._ensure_port()
            midi = mido.MidiFile(path)
            # DESIGN: mido.MidiFile.play() sleeps INSIDE its generator between
            # messages, so the stop flag would only be checked once per message -
            # a groove with a longer gap between hits could make Stop take as long
            # as that gap to respond. We track absolute time ourselves and sleep in
            # SMALL (15ms) increments, checking the stop flag between every
            # increment, so Stop (and switching tracks) is always responsive
            # quickly regardless of how sparse the groove is.
            tempo = 500000   # microseconds per beat, MIDI default (120 bpm)
            start = time.monotonic()
            elapsed_ticks = 0
            for msg in mido.merge_tracks(midi.tracks):
                elapsed_ticks += msg.time   # delta in ticks
                # DESIGN: tempo_scale rescales the WHOLE performance's timing axis
                # uniformly (not the file's own embedded tempo value itself) - this
                # preserves the groove's internal feel/swing exactly, just played
                # faster or slower to land on a different target tempo.
                target_time = mido.tick2second(elapsed_ticks, midi.ticks_per_beat, tempo) / tempo_scale
                while True:
                    remaining = target_time - (time.monotonic() - start)
                    if remaining <= 0:
                        break
                    if self._stop_event.wait(timeout=min(0.015, remaining)):
                        break
                if self._stop_event.is_set():
                    break
                if msg.is_meta:
                    if msg.type == 'set_tempo':
                        tempo = msg.tempo
                    continue
                if force_drum_channel and hasattr(msg, 'channel'):
                    msg = msg.copy(channel=self.DRUM_CHANNEL)
                port.send(msg)
        except Exception as exc:
            error = _report_error(f"playing '{os.path.basename(path)}'", exc)
        finally:
            self._all_notes_off()
            self.is_playing = False
            if on_finished is not None:
                on_finished(error)

    def _all_notes_off(self):
        """Safety net: always silence every channel when playback ends, so a
        Stop mid-note (or an error mid-playback) never leaves a note hanging."""
        if self._outport is None:
            return
        try:
            for ch in range(16):
                self._outport.send(mido.Message('control_change', control=123, value=0, channel=ch))
        except Exception:
            pass   # best-effort cleanup; nothing more useful to do if this fails

    def close(self):
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._outport is not None:
            try:
                self._outport.close()
            except Exception:
                pass
            self._outport = None


# =============================================================================
# SHARED MIDI HELPERS
# =============================================================================

def _sliced_instrument(source_path: str, start_sec: float, end_sec: float,
                       drums_only: bool, program: int) -> pretty_midi.Instrument:
    """Notes from source_path inside [start_sec, end_sec), time-shifted to start
    at 0, as a single Instrument. Shared by slice_midi_to_temp (one part, its own
    file) and audition_both_to_temp (both parts, one file - see there for why)."""
    src = pretty_midi.PrettyMIDI(source_path)
    inst = pretty_midi.Instrument(program=program, is_drum=drums_only)
    for orig in src.instruments:
        if drums_only and not orig.is_drum:
            continue
        for n in orig.notes:
            if start_sec <= n.start < end_sec:
                inst.notes.append(pretty_midi.Note(
                    velocity=n.velocity, pitch=n.pitch,
                    start=n.start - start_sec,
                    end=max(n.start - start_sec + 0.01, n.end - start_sec)))
    return inst


def slice_midi_to_temp(source_path: str, start_sec: float, end_sec: float,
                       temp_dir: str, drums_only: bool = False, program: int = 0) -> str:
    """
    Write a NEW small MIDI file containing only the notes inside
    [start_sec, end_sec) of source_path, times shifted to start at 0.
    DESIGN: generalized to work for either the drum file (drums_only=True, since a
    stray non-drum track should never leak in) or the bass file (drums_only=
    False - a bass track is not marked is_drum in a MIDI file).

    program: GM program for the output instrument (ignored when drums_only,
    since is_drum=True already selects the percussion kit regardless of
    program). Defaults to 0 (Acoustic Grand Piano) for existing callers that
    only feed this into pitch/timing processing and never play it directly;
    pass an explicit program (e.g. 33, "Electric Bass (finger)" - same as
    the render step's own bass_inst) when the slice will be auditioned.
    """
    if not HAS_PRETTY_MIDI:
        raise RuntimeError("pretty_midi is required to slice segments.")
    src = pretty_midi.PrettyMIDI(source_path)
    tempo = _tempo_at_time(src, start_sec)
    out = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = _sliced_instrument(source_path, start_sec, end_sec, drums_only, program)
    out.instruments.append(inst)
    path = os.path.join(temp_dir, f"slice_{uuid.uuid4().hex[:8]}.mid")
    out.write(path)
    return path, tempo, len(inst.notes)


def audition_both_to_temp(drum_path: str, bass_path: Optional[str], start_sec: float,
                          end_sec: float, temp_dir: str) -> str:
    """
    Same idea as slice_midi_to_temp, but writes the drum AND bass slices into ONE
    file as two instrument tracks sharing a single MIDI clock, instead of two
    separate files. DESIGN: MidiPlayer plays one file at a time, so this is
    what makes "play both simultaneously" both possible AND perfectly synced -
    two independently-started playbacks would drift/race, one merged file has
    no sync problem to begin with. bass_path is optional (drum-only segment if
    no bass file is loaded yet).
    """
    if not HAS_PRETTY_MIDI:
        raise RuntimeError("pretty_midi is required to slice segments.")
    tempo = _tempo_at_time(pretty_midi.PrettyMIDI(drum_path), start_sec)
    out = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    out.instruments.append(_sliced_instrument(drum_path, start_sec, end_sec, True, 0))
    if bass_path:
        out.instruments.append(_sliced_instrument(bass_path, start_sec, end_sec, False, 33))
    path = os.path.join(temp_dir, f"slice_both_{uuid.uuid4().hex[:8]}.mid")
    out.write(path)
    return path


def force_all_drums_to_temp(source_path: str, temp_dir: str) -> str:
    """
    The 'Drum MIDI' drop zone is an explicit statement of intent - whatever file
    lands there IS the drum part, regardless of which MIDI channel/program the
    exporting DAW put it on. Reaper (and others) routinely export a single drum
    track on channel 1 rather than the MIDI-standard channel 10, so pretty_midi's
    is_drum flag misses it entirely - any GM playback device then falls back to
    channel 1's default program (Acoustic Grand Piano), and every is_drum-gated
    step downstream (segmentation, humanizing, slicing) sees zero drum notes.

    drum_humanizer_v3.py/drum_theme_segmentation.py guess at this with a careful
    single-track/name-match fallback because THEY scan a whole unlabeled library
    where guessing wrong risks folding a bass or melodic line into "drums". That
    ambiguity doesn't exist here - the user already resolved it by dropping the
    file into this specific control - so this forces EVERY note in the file onto
    one is_drum=True instrument, unconditionally. Only .instruments is replaced;
    tempo/time-signature data is untouched, so segment timing on multi-tempo
    songs is unaffected.
    """
    midi = pretty_midi.PrettyMIDI(source_path)
    forced = pretty_midi.Instrument(program=0, is_drum=True, name="Drums (forced)")
    for inst in midi.instruments:
        forced.notes.extend(inst.notes)
    forced.notes.sort(key=lambda n: n.start)
    midi.instruments = [forced]
    out_path = os.path.join(temp_dir, f"forced_drums_{uuid.uuid4().hex[:8]}.mid")
    midi.write(out_path)
    return out_path


def fit_groove_to_segment(groove_path: str, target_bars: int, target_tempo: float,
                          temp_dir: str) -> str:
    """
    The inverse of slicing: takes a library groove (its own tempo, its own
    length) and fits it into a segment's slot - retimed to the segment's
    tempo, then looped end-to-end to fill the segment's bar length and
    trimmed to exactly that length (the standard "tap a loop into a section"
    behavior, same idea as Superior Drummer 3's Tap2Find).

    DESIGN: assumes a fixed 4/4 bar, same simplifying assumption
    drum_theme_segmentation.py itself makes (segments are only ever detected
    on 4/4 bar boundaries in the first place, so this is consistent, not a
    new limitation). The loop unit is snapped to the nearest whole bar at the
    RETIMED tempo, so the seam lands on a downbeat rather than mid-bar.
    """
    if not HAS_PRETTY_MIDI:
        raise RuntimeError("pretty_midi is required to use a groove.")
    if not HAS_GROOVE_FINDER:
        raise RuntimeError("find_similar_grooves.py is required to use a groove.")
    src = pretty_midi.PrettyMIDI(groove_path)
    notes, _source = fsg._select_drum_notes(src)
    if not notes:
        raise ValueError(f"'{os.path.basename(groove_path)}' has no drum notes to use.")
    native_tempo = _tempo_at_time(src, 0.0)
    scale = (native_tempo / target_tempo) if target_tempo > 0 else 1.0
    bar_dur = 240.0 / target_tempo if target_tempo > 0 else 2.0   # seconds/bar, 4/4, at target tempo

    retimed = sorted((n.start * scale, n.end * scale, n.pitch, n.velocity) for n in notes)
    retimed_end = max(e for _, e, _, _ in retimed)
    loop_bars = max(1, round(retimed_end / bar_dur)) if bar_dur > 0 else 1
    loop_dur = loop_bars * bar_dur
    target_dur = max(bar_dur, target_bars * bar_dur)

    out = pretty_midi.PrettyMIDI(initial_tempo=target_tempo)
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Swapped groove")
    n_repeats = max(1, int(np.ceil(target_dur / loop_dur)))
    for rep in range(n_repeats):
        offset = rep * loop_dur
        if offset >= target_dur:
            break
        for s, e, pitch, vel in retimed:
            ns = s + offset
            if ns >= target_dur:
                continue
            inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch,
                                               start=ns, end=min(e + offset, target_dur)))
    inst.notes.sort(key=lambda n: n.start)
    out.instruments.append(inst)
    out_path = os.path.join(temp_dir, f"swap_{uuid.uuid4().hex[:8]}.mid")
    out.write(out_path)
    return out_path


def _error_location(exc: BaseException) -> str:
    tb = exc.__traceback__
    last = None
    while tb is not None:
        last = tb
        tb = tb.tb_next
    if last is None:
        return "unknown location"
    f = last.tb_frame
    return f"{os.path.basename(f.f_code.co_filename)}:{last.tb_lineno} in {f.f_code.co_name}()"


def _report_error(context: str, exc: BaseException) -> str:
    loc = _error_location(exc)
    msg = f"{context}\n-> {type(exc).__name__} at {loc}: {exc}"
    print(f"[ERROR] {msg}")
    return msg


# =============================================================================
# PHASE 1 - DRUM HUMANIZE  (thin, tested reuse of drum_humanizer_v3.humanize_file)
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
# PHASE 2 - MANUAL RUSH / DRAG + QUANTIZE
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
# (PHASE2_WEAK_BEAT_DAMPING) - "the 2 and the 4" always included, per spec,
# always at a reduced effect relative to e/a.

def classify_phase2_role(grid_step: int, cfg) -> str:
    """Returns 'strong_beat' (1,3 -> quantize target), 'syncopation' (e,a ->
    full rush/drag), 'weak_beat' (2,4 -> damped rush/drag), or 'none' (the
    "&"/and position, or an off-16th-grid note - untouched either way)."""
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
# PHASE 3 - BASS SYNC TO DRUMS
# =============================================================================
# DESIGN: works entirely in ABSOLUTE TIME (seconds) via plain pretty_midi, on
# BOTH sides - never through drum_humanizer_v3's grid+offset representation.
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
    (see _interpolated_delay_range_ms) so the two transients stay distinct -
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
# DESIGN: "never re-lock" - a phase's settings/output are independent instance
# state, always re-derivable from the phase before it. Gating only controls
# whether a phase's CONTROLS are enabled (has the prior phase been run at
# least once for this segment?), never whether they can be revisited. Reopening
# Phase 1 after Phase 2/3 were already applied does NOT wipe those downstream
# results - they simply become STALE (flagged in the UI) until re-applied,
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
    before this one changed since I was last computed" - shown in the UI as a
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

    # ── Groove swap: replacing this segment's original drum recording with a
    # similar groove found in the library (see fit_groove_to_segment). None =
    # using the original recording, unchanged. ─────────────────────────────
    swapped_groove_path: Optional[str] = None    # fitted (retimed/looped/trimmed) file
    swapped_groove_source: Optional[str] = None  # the library file it came from, for display
    groove_results: Optional[List[Dict]] = None  # cached similar-groove search results
    groove_search_error: Optional[str] = None

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
        self.swapped_groove_path = None
        self.swapped_groove_source = None


# =============================================================================
# SMALL REUSABLE WIDGET: a collapsible ("accordion") section
# =============================================================================

class CollapsibleSection(ttk.Frame):
    """Self-packing: manages its own fill/expand so that whichever section(s)
    are currently open claim any extra vertical space the window is resized
    into (rather than it collecting as dead space below the accordion), while
    collapsed sections stay compact. See _toggle()."""
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
            self.body.pack(fill='both', expand=True, pady=(2, 6))
        self.pack(fill='both' if start_open else 'x', expand=start_open)

    def _arrow(self):
        return "\u25bc" if self._open.get() else "\u25b6"

    def _toggle(self):
        self._open.set(not self._open.get())
        title = self._toggle_btn.cget('text').split(' ', 1)[1]
        self._toggle_btn.config(text=self._arrow() + " " + title)
        if self._open.get():
            self.body.pack(fill='both', expand=True, pady=(2, 6))
        else:
            self.body.pack_forget()
        self.pack_configure(fill='both' if self._open.get() else 'x', expand=self._open.get())

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
        root.resizable(True, True)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.drum_path: Optional[str] = None
        self.drum_original_path: Optional[str] = None   # the file AS DROPPED, before
                                                         # force_all_drums_to_temp - needed
                                                         # to reload it fresh on session load,
                                                         # since the forced temp copy doesn't
                                                         # survive an app restart
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
        self.seg_threshold_var = tk.DoubleVar(value=SEGMENTATION_CONFIDENCE_THRESHOLD)

        self.groove_index = None
        self.groove_index_path: Optional[str] = None
        self.groove_selected_result: Optional[Dict] = None   # currently highlighted row
                                                              # in the Find Similar Groove list

        self.temp_dir = tempfile.mkdtemp(prefix='drum_bass_studio_')
        self.player = MidiPlayer()
        self.playing_path: Optional[str] = None
        self.playing_kind: Optional[str] = None   # 'drum'/'bass'/'both' if one of the
                                                   # three segment-audition buttons is
                                                   # currently playing, else None

        self._build_widgets()
        # DESIGN: pretrained/ is gitignored (see README), so a fresh clone/machine has
        # no models at all yet. Rather than silently reaching out to Google Drive on
        # every launch, ask ONCE (a local, no-network filesystem check decides whether
        # to even ask) and only fetch what's missing if the user agrees. This runs
        # before mainloop() starts, so it's a one-time blocking delay only when a
        # model is genuinely missing AND the user opts in; every later launch finds
        # them locally and skips both the prompt and the network check entirely.
        if HAS_PRETRAINED_DOWNLOADER:
            missing = pretrained_models_missing()
            if missing and messagebox.askyesno(
                    "Download pretrained models?",
                    "The following pretrained model(s) were not found in pretrained/:\n\n  "
                    + "\n  ".join(missing) +
                    "\n\nDownload them now from the shared Google Drive folder?"):
                ensure_pretrained_model()
        self._load_default_pretrained()
        self._load_default_seg_model()
        self._load_default_groove_index()
        self._refresh_gating()

    def _set_model_status(self, dot, state):
        """Very small colored dot beside each model label confirming it actually
        loaded. 'ok' (green) / 'error' (red, load was attempted and failed) /
        'none' (gray, nothing loaded yet)."""
        color = {'ok': '#0ca30c', 'error': '#d03b3b', 'none': '#c3c2b7'}[state]
        dot.config(foreground=color)

    def _load_default_pretrained(self):
        """Auto-select the bundled pretrained/humanizer_best.pt, if present, so the
        app is usable without a manual 'Load...' click. Still fully overridable -
        this only sets the same state a manual Load does.

        Best-effort: a failed/skipped download just leaves the label at "(none
        loaded)" (and the status dot gray), same as on a checkout with no pretrained/
        folder at all - the user can still Load... manually.
        """
        if not HAS_HUMANIZER:
            return
        path = dhu.DEFAULT_PRETRAINED_CHECKPOINT
        if os.path.exists(path):
            self.hum_checkpoint_path = path
            self.hum_model_label.config(text=f"{os.path.basename(path)} (bundled default)",
                                        foreground='black')
            self._set_model_status(self.hum_status_dot, 'ok')

    def _load_default_seg_model(self):
        """Auto-load the bundled pretrained/segmentation_best.pt, if present, so
        segment detection works without a manual 'Load...' click. Unlike the
        humanizer (a lazy checkpoint path, only read when Phase 1 actually runs),
        the segmentation model is an eagerly-loaded in-memory nn.Module - so this
        reuses the same threaded worker as the manual Load button, just pointed at
        the bundled default and with the failure messagebox suppressed (this runs
        automatically at startup, not from a user click)."""
        if not HAS_SEGMENTATION:
            return
        path = dts.DEFAULT_PRETRAINED_CHECKPOINT
        if not os.path.exists(path):
            return
        threading.Thread(target=self._load_seg_model_worker, args=(path,),
                         kwargs={'show_error': False, 'is_default': True}, daemon=True).start()

    def _load_default_groove_index(self):
        """Auto-load the bundled cache/groove_index.pkl, if present, so the
        Find Similar Groove panel works without a manual 'Load...' click -
        same pattern as the segmentation model default above."""
        if not HAS_GROOVE_FINDER:
            return
        path = DEFAULT_GROOVE_INDEX_PATH
        if not os.path.exists(path):
            return
        threading.Thread(target=self._load_groove_index_worker, args=(path,),
                         kwargs={'is_default': True}, daemon=True).start()

    def _on_load_groove_index(self):
        path = filedialog.askopenfilename(title="Select groove index cache",
                                          filetypes=[("Index cache", "*.pkl"), ("All files", "*.*")])
        if not path:
            return
        if not HAS_GROOVE_FINDER:
            messagebox.showerror("Missing module", "find_similar_grooves.py not found alongside this script.")
            return
        self._set_status("Loading groove index...", busy=True)
        threading.Thread(target=self._load_groove_index_worker, args=(path,), daemon=True).start()

    def _load_groove_index_worker(self, path, is_default=False):
        try:
            index = fsg.load_index(path)
        except Exception as exc:
            msg = _report_error(f"loading groove index '{path}'", exc)
            self.root.after(0, lambda: self._on_groove_index_load_error(msg, is_default))
            return
        self.root.after(0, lambda: self._on_groove_index_loaded(path, index, is_default))

    def _on_groove_index_loaded(self, path, index, is_default=False):
        self.groove_index = index
        self.groove_index_path = path
        n = len(index.get('paths', []))
        suffix = " (bundled default)" if is_default else ""
        self.groove_index_label.config(text=f"{os.path.basename(path)}{suffix}  ({n} files)",
                                       foreground='black')
        self._set_model_status(self.groove_index_status_dot, 'ok')
        self._set_status(f"Groove index loaded ({n} files).")
        # if segments already exist, search each of them now
        if self.segments and self.drum_path:
            self._start_groove_batch_search()

    def _on_groove_index_load_error(self, msg, is_default=False):
        self._set_model_status(self.groove_index_status_dot, 'error')
        if not is_default:
            messagebox.showerror("Failed to load groove index", msg)
        self._set_status("Failed to load groove index.")

    # ------------------------------------------------------------------ UI --
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Save Session...", command=self._on_save_session,
                              accelerator="Ctrl+S")
        file_menu.add_command(label="Load Session...", command=self._on_load_session,
                              accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)
        self.root.bind('<Control-s>', lambda e: self._on_save_session())
        self.root.bind('<Control-o>', lambda e: self._on_load_session())

    def _build_widgets(self):
        self._build_menu()
        # -- model loaders --
        top = ttk.Frame(self.root); top.pack(fill='x', padx=8, pady=(8, 2))
        ttk.Label(top, text="Humanizer model:").pack(side='left')
        self.hum_status_dot = tk.Label(top, text="●", foreground='#c3c2b7', font=('Segoe UI', 8))
        self.hum_status_dot.pack(side='left', padx=(6, 0))
        self.hum_model_label = ttk.Label(top, text="(none loaded)", foreground='gray')
        self.hum_model_label.pack(side='left', padx=6)
        ttk.Button(top, text="Load...", command=self._on_load_hum_model).pack(side='right')

        top2 = ttk.Frame(self.root); top2.pack(fill='x', padx=8, pady=(0, 6))
        ttk.Label(top2, text="Segmentation model:").pack(side='left')
        self.seg_status_dot = tk.Label(top2, text="●", foreground='#c3c2b7', font=('Segoe UI', 8))
        self.seg_status_dot.pack(side='left', padx=(6, 0))
        self.seg_model_label = ttk.Label(top2, text="(none loaded)", foreground='gray')
        self.seg_model_label.pack(side='left', padx=6)
        ttk.Button(top2, text="Load...", command=self._on_load_seg_model).pack(side='right')

        top2b = ttk.Frame(self.root); top2b.pack(fill='x', padx=8, pady=(0, 6))
        ttk.Label(top2b, text="Groove index:").pack(side='left')
        self.groove_index_status_dot = tk.Label(top2b, text="●", foreground='#c3c2b7', font=('Segoe UI', 8))
        self.groove_index_status_dot.pack(side='left', padx=(6, 0))
        self.groove_index_label = ttk.Label(top2b, text="(none loaded)", foreground='gray')
        self.groove_index_label.pack(side='left', padx=6)
        ttk.Button(top2b, text="Load...", command=self._on_load_groove_index).pack(side='right')

        # -- segmentation sensitivity --
        # Maps directly to the model's boundary-probability threshold (see
        # SEGMENTATION_CONFIDENCE_THRESHOLD in config.py): 0.05-0.95 is the same
        # range drum_theme_segmentation.py's own sweep_threshold() scores at
        # training time, so every value here is one the model was actually
        # evaluated at. The scale is reversed (from_=0.95 to=0.05) so dragging
        # right raises sensitivity (lower threshold -> more, subtler boundaries)
        # and dragging left lowers it (higher threshold -> fewer, more confident
        # ones) - "more sensitive" reads naturally as "further right".
        top3 = ttk.Frame(self.root); top3.pack(fill='x', padx=8, pady=(0, 6))
        ttk.Label(top3, text="Segmentation sensitivity:").pack(side='left')
        self.seg_threshold_scale = ttk.Scale(top3, from_=0.95, to=0.05, orient='horizontal',
                                             variable=self.seg_threshold_var,
                                             command=self._on_seg_threshold_drag)
        self.seg_threshold_scale.pack(side='left', fill='x', expand=True, padx=(8, 4))
        self.seg_threshold_label = ttk.Label(top3, text=f"{SEGMENTATION_CONFIDENCE_THRESHOLD:.2f}", width=5)
        self.seg_threshold_label.pack(side='left')
        self.seg_threshold_scale.bind('<ButtonRelease-1>', self._on_seg_threshold_release)
        # Arrow-key adjustment (after Tab-focusing the slider) doesn't fire a mouse
        # ButtonRelease, so re-segmenting would otherwise never trigger for keyboard
        # users - <KeyRelease> covers that the same way.
        self.seg_threshold_scale.bind('<KeyRelease>', self._on_seg_threshold_release)

        # -- drum drop zone --
        drum_label_row = ttk.Frame(self.root); drum_label_row.pack(fill='x', padx=8)
        ttk.Label(drum_label_row, text="Drum MIDI (full song):").pack(side='left')
        both_btn = ttk.Button(drum_label_row, text="▶ Audition selected segment",
                              command=lambda: self._toggle_audition('both'))
        both_btn.pack(side='right')
        drum_btn = ttk.Button(drum_label_row, text="▶ Audition selected drum segment",
                              command=lambda: self._toggle_audition('drum'))
        drum_btn.pack(side='right', padx=(0, 6))
        self.drum_drop = tk.Label(self.root, text=self._drop_text("drum"), relief='groove',
                                  bd=2, height=6, bg='#f5f5f5', fg='#555', cursor='hand2')
        self.drum_drop.pack(fill='x', padx=8, pady=(0, 4))
        self.drum_drop.bind('<Button-1>', self._on_browse_drum)

        # Segmentation results OVERLAP the drop zone itself (same footprint) once
        # computed, instead of taking their own row below it - saves the vertical
        # space a separate timeline strip used to cost. place(in_=...,
        # relwidth=1, relheight=1) tracks the drop zone's size/position exactly,
        # including through a window resize; place_forget() reveals the drop
        # zone's own text again whenever there's nothing to show. See _draw_timeline.
        self.seg_canvas = tk.Canvas(self.root, bg='#e8e8e8', highlightthickness=0)
        self.seg_canvas.bind('<Configure>', lambda e: self._redraw_timeline_canvas(self.seg_canvas, self.drum_drop))

        # -- bass drop zone --
        bass_label_row = ttk.Frame(self.root); bass_label_row.pack(fill='x', padx=8, pady=(6, 0))
        ttk.Label(bass_label_row, text="Bass MIDI (matching song, same tempo/alignment):").pack(side='left')
        bass_btn = ttk.Button(bass_label_row, text="▶ Audition selected bass segment",
                              command=lambda: self._toggle_audition('bass'))
        bass_btn.pack(side='right')
        self._audition_buttons = {'drum': drum_btn, 'bass': bass_btn, 'both': both_btn}
        self.bass_drop = tk.Label(self.root, text=self._drop_text("bass"), relief='groove',
                                  bd=2, height=6, bg='#f5f5f5', fg='#555', cursor='hand2')
        self.bass_drop.pack(fill='x', padx=8, pady=(0, 6))
        self.bass_drop.bind('<Button-1>', self._on_browse_bass)

        # Same overlap treatment on the bass drop zone - the segment timeline is
        # one song structure, shown wherever there's a drop zone for it.
        self.seg_canvas_bass = tk.Canvas(self.root, bg='#e8e8e8', highlightthickness=0)
        self.seg_canvas_bass.bind('<Configure>', lambda e: self._redraw_timeline_canvas(self.seg_canvas_bass, self.bass_drop))

        if HAS_DND:
            for widget, kind in ((self.drum_drop, 'drum'), (self.bass_drop, 'bass'),
                                 (self.seg_canvas, 'drum'), (self.seg_canvas_bass, 'bass')):
                try:
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind('<<Drop>>', lambda e, k=kind: self._on_drop(e, k))
                except Exception as exc:
                    _report_error("enabling drag-and-drop (falling back to click-to-browse)", exc)

            # OUTBOUND: dragging a segment rectangle OUT of the timeline copies
            # its current audio to wherever it's dropped (Explorer, another
            # app, etc) - Windows-only feature, uses OLE2 drag-and-drop under
            # the hood via tkdnd. See _on_segment_drag_init.
            for canvas, kind in ((self.seg_canvas, 'drum'), (self.seg_canvas_bass, 'bass')):
                try:
                    canvas.drag_source_register(1, DND_FILES)
                    canvas.dnd_bind('<<DragInitCmd>>',
                                    lambda e, c=canvas, k=kind: self._on_segment_drag_init(e, c, k))
                except Exception as exc:
                    _report_error("enabling segment drag-out", exc)

        # -- segment navigation: move between segments without needing to
        # click the exact timeline rectangle (fiddly on a song with many
        # short segments) - also doubles as a position indicator. ----------
        nav_row = ttk.Frame(self.root); nav_row.pack(fill='x', padx=8, pady=(4, 0))
        self.prev_segment_btn = ttk.Button(nav_row, text="◀ Previous Segment",
                                           command=self._on_prev_segment, state='disabled')
        self.prev_segment_btn.pack(side='left')
        self.segment_nav_label = ttk.Label(nav_row, text="No segments yet",
                                           foreground='gray', anchor='center')
        self.segment_nav_label.pack(side='left', fill='x', expand=True)
        self.next_segment_btn = ttk.Button(nav_row, text="Next Segment ▶",
                                           command=self._on_next_segment, state='disabled')
        self.next_segment_btn.pack(side='right')
        # Ctrl+Arrow rather than a bare arrow key - a bare Left/Right would
        # collide with the normal keyboard behavior of whatever widget
        # currently has focus (a Scale, a Spinbox, the groove Treeview's own
        # row navigation), so this needs to be unambiguous and global.
        self.root.bind('<Control-Right>', lambda e: self._on_next_segment())
        self.root.bind('<Control-Left>', lambda e: self._on_prev_segment())

        ttk.Separator(self.root).pack(fill='x', padx=8, pady=4)

        # -- body holding the groove-swap panel + three phase sections --
        outer = ttk.Frame(self.root); outer.pack(fill='both', expand=True, padx=8)
        self.groove_section = CollapsibleSection(outer, "Find Similar Groove", start_open=False)
        self.phase1_section = CollapsibleSection(outer, "Phase 1 -- Drum Humanize", start_open=PHASE1_SECTION_STARTS_OPEN)
        self.phase2_section = CollapsibleSection(outer, "Phase 2 -- Rush / Drag", start_open=PHASE2_SECTION_STARTS_OPEN)
        self.phase3_section = CollapsibleSection(outer, "Phase 3 -- Bass Sync", start_open=PHASE3_SECTION_STARTS_OPEN)
        self._build_groove_section(self.groove_section.body)
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
        self._set_model_status(self.hum_status_dot, 'ok')
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

    def _load_seg_model_worker(self, path, show_error=True, is_default=False):
        try:
            device = dts.torch.device('cuda' if dts.torch.cuda.is_available() else 'cpu')
            model, cfg = dts.load_model(path, device)
        except Exception as exc:
            msg = _report_error(f"loading segmentation model '{path}'", exc)
            self.root.after(0, lambda: self._on_seg_model_load_failed(msg, show_error))
            return
        self.root.after(0, lambda: self._on_seg_model_loaded(path, model, cfg, is_default))

    def _on_seg_model_loaded(self, path, model, cfg, is_default=False):
        self.seg_model = model
        self.seg_cfg = cfg
        self.seg_checkpoint_path = path
        suffix = " (bundled default)" if is_default else ""
        self.seg_model_label.config(text=f"{os.path.basename(path)}{suffix}", foreground='black')
        self._set_model_status(self.seg_status_dot, 'ok')
        self._set_status(f"Segmentation model loaded: {os.path.basename(path)}")

    def _on_seg_model_load_failed(self, msg, show_error):
        self._set_model_status(self.seg_status_dot, 'error')
        self._set_status("Segmentation model failed to load.")
        if show_error:
            messagebox.showerror("Failed to load model", msg)

    def _on_seg_threshold_drag(self, value_str):
        """Live-update the numeric readout while dragging. Re-segmenting is
        deferred to <ButtonRelease-1> (_on_seg_threshold_release) since it
        reruns the model and shouldn't fire on every pixel of drag."""
        self.seg_threshold_label.config(text=f"{float(value_str):.2f}")

    def _on_seg_threshold_release(self, event=None):
        # Snap to the 0.05 steps sweep_threshold() actually scores, so the
        # value shown is always one the model was validated at.
        snapped = round(self.seg_threshold_var.get() / 0.05) * 0.05
        snapped = min(0.95, max(0.05, snapped))
        self.seg_threshold_var.set(snapped)
        self.seg_threshold_label.config(text=f"{snapped:.2f}")
        if self.seg_model is not None and self.drum_path:
            self.selected_index = None
            self.segments = []
            self._draw_timeline()   # clears + hides both overlays, revealing the drop zones again
            self._set_status(f"Re-segmenting at sensitivity threshold {snapped:.2f}...", busy=True)
            threading.Thread(target=self._segment_worker, args=(self.drum_path, snapped), daemon=True).start()

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
        self._draw_timeline()   # segments (if any) now also overlay the bass drop zone

    def _load_drum_file(self, path):
        if self.seg_model is None:
            messagebox.showwarning("No segmentation model",
                                   "Load a segmentation model first (top of window).")
            return
        try:
            forced_path = force_all_drums_to_temp(path, self.temp_dir)
        except Exception as exc:
            msg = _report_error(f"forcing drum channel on '{path}'", exc)
            messagebox.showerror("Failed to load drum MIDI", msg)
            return
        self.drum_path = forced_path
        self.drum_original_path = path
        self.drum_drop.config(text=os.path.basename(path), foreground='black')
        self.selected_index = None
        self.segments = []
        self._draw_timeline()   # clears + hides both overlays, revealing the drop zones again
        self._update_segment_nav_label()
        self._set_status(f"Segmenting '{os.path.basename(path)}'...", busy=True)
        threading.Thread(target=self._segment_worker, args=(forced_path, self.seg_threshold_var.get()),
                         daemon=True).start()

    def _segment_worker(self, path, threshold):
        try:
            result = dts.compute_segment_boundaries(self.seg_model, self.seg_cfg, path,
                                                     threshold=threshold,
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
        self._update_segment_nav_label()
        if not segs:
            self._set_status("No segments detected.")
            return
        self._set_status(f"{len(segs)} segments detected -- click one to begin.")
        if self.groove_index is not None:
            self._start_groove_batch_search()

    # ---------------------------------------------------- groove similarity --
    def _start_groove_batch_search(self):
        """Eagerly searches similar grooves for EVERY detected segment, right
        after segmentation (or right after a groove index gets loaded, if
        segments already existed) - not just whichever one the user happens to
        select. Results are cached on each segment (seg.groove_results), so
        selecting a segment shows them instantly."""
        drum_path = self.drum_path
        segments = list(self.segments)
        cfg = fsg.Config(**{k: v for k, v in self.groove_index['cfg'].items()
                            if k in fsg.Config.__dataclass_fields__})
        threading.Thread(target=self._groove_batch_search_worker,
                         args=(drum_path, segments, cfg), daemon=True).start()

    def _groove_batch_search_worker(self, drum_path, segments, cfg):
        for i, seg in enumerate(segments):
            if drum_path != self.drum_path or self.groove_index is None:
                return   # a newer file was dropped, or the index changed - abandon
            error = None
            results = None
            try:
                raw_path, _, _ = slice_midi_to_temp(drum_path, seg.start_sec, seg.end_sec,
                                                    self.temp_dir, drums_only=True)
                query_fp = fsg.extract_fingerprint(raw_path, cfg)
                if query_fp is None:
                    raise ValueError("Too few drum notes in this segment to search with.")
                sims = fsg.compute_similarities(query_fp, self.groove_index, cfg)
                order = np.argsort(-sims)
                results = []
                for j in order:
                    results.append({'path': self.groove_index['paths'][j],
                                    'similarity': float(sims[j]),
                                    'tempo': float(self.groove_index['tempo'][j])})
                    if len(results) >= GROOVE_SEARCH_TOP_K:
                        break
            except Exception as exc:
                error = _report_error(f"searching similar grooves for segment {i+1}", exc)
            self.root.after(0, lambda seg=seg, results=results, error=error:
                            self._on_groove_search_done(seg, results, error))

    def _on_groove_search_done(self, seg, results, error):
        # identity check, not value-equality (SegmentSettings is a dataclass -
        # `in`/`==` would compare field VALUES, not "is this the same object
        # still being tracked", which is what staleness actually means here).
        if not any(s is seg for s in self.segments):
            return   # stale - segments were reset/re-segmented since this was queued
        seg.groove_results = results if results is not None else []
        seg.groove_search_error = error
        if self._current_segment() is seg:
            self._refresh_groove_section(seg)

    # ------------------------------------------------------------ timeline --
    def _draw_timeline(self):
        """Segment results overlap the drum/bass drop zones themselves (same
        footprint, via place(in_=...)) instead of taking their own row, so they
        cost no extra vertical space. place_forget() when there's nothing to
        show reveals each drop zone's own "Drop ... here" text again.

        The bass overlay specifically only appears once a bass file is
        actually loaded - segments come from the DRUM file alone, so showing
        colored segment blocks over an empty "Drop bass MIDI here" zone before
        any bass file exists would misleadingly look like something's there."""
        self.segment_click_targets = {}
        overlays = [(self.seg_canvas, self.drum_drop)]
        if self.bass_path:
            overlays.append((self.seg_canvas_bass, self.bass_drop))
        else:
            self.seg_canvas_bass.delete('all')
            self.seg_canvas_bass.place_forget()
        for canvas, _target in overlays:
            canvas.delete('all')
        if not self.segments:
            for canvas, _target in overlays:
                canvas.place_forget()
            return
        for canvas, target in overlays:
            canvas.place(in_=target, x=0, y=0, relwidth=1.0, relheight=1.0)
        self.root.update_idletasks()
        for canvas, target in overlays:
            self._draw_segments_on(canvas, target)

    def _redraw_timeline_canvas(self, canvas, target):
        """The canvas itself tracks the drop zone's size dynamically (place()'s
        relwidth/relheight), but the rectangles drawn INSIDE it are fixed pixel
        coordinates from whenever they were last drawn - they don't rescale on
        their own. Bound to each overlay canvas's <Configure> (fires whenever it
        actually changes size, e.g. the window being resized) so segments always
        fill the full current width instead of leaving empty space after a
        resize."""
        if not self.segments:
            return
        canvas.delete('all')
        self._draw_segments_on(canvas, target)

    def _draw_segments_on(self, canvas, target):
        # DESIGN: read the TARGET's size, not the canvas's own winfo_width/height.
        # place(relwidth/relheight=1.0) sizes the canvas to match target, but right
        # after place() the canvas's own geometry hasn't necessarily been re-queried
        # yet (a one-frame lag), while target has been stably packed since startup -
        # so target's size is the reliable source of truth for how big to draw.
        canvas_w = max(200, target.winfo_width())
        canvas_h = max(24, target.winfo_height())
        total_bars = sum(s.end_bar - s.start_bar for s in self.segments)
        if total_bars <= 0:
            return
        y0, y1 = 2, canvas_h - 2
        mid_y = canvas_h / 2
        is_drum_canvas = (canvas is self.seg_canvas)
        x = 0
        for i, seg in enumerate(self.segments):
            length = seg.end_bar - seg.start_bar
            w = max(SEGMENT_MIN_RECT_WIDTH_PX, round(canvas_w * length / total_bars))
            color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
            selected = (i == self.selected_index)
            rect = canvas.create_rectangle(
                x, y0, x + w, y1, fill=color,
                outline=SELECTED_OUTLINE_COLOR if selected else 'white',
                width=SEGMENT_OUTLINE_WIDTH_SELECTED if selected else SEGMENT_OUTLINE_WIDTH_NORMAL,
                tags=(f'seg{i}',))
            label = f"{seg.start_bar+1}-{seg.end_bar}"
            if w > 28:
                canvas.create_text(x + w / 2, mid_y, text=label, fill='white',
                                   font=('', 8), tags=(f'seg{i}',))
            if seg.is_customized():
                canvas.create_oval(x + w - 10, y0 + 2, x + w - 2, y0 + 10,
                                   fill=CUSTOMIZED_MARKER_COLOR, outline='',
                                   tags=(f'seg{i}',))
            if seg.swapped_groove_path:
                # distinct from the "customized" dot (top-right) - this marks
                # the segment's SOURCE audio itself has been swapped, a bigger
                # change than a settings tweak.
                canvas.create_rectangle(x + 2, y0 + 2, x + 10, y0 + 10,
                                        fill=SWAPPED_GROOVE_MARKER_COLOR, outline='#333',
                                        tags=(f'seg{i}',))
            canvas.tag_bind(f'seg{i}', '<Button-1>', lambda e, idx=i: self._on_segment_selected(idx))
            if is_drum_canvas:
                self.segment_click_targets[i] = {'rect': rect, 'x0': x, 'x1': x + w}
            x += w

    def _on_segment_selected(self, idx):
        self.selected_index = idx
        self._draw_timeline()
        self._restore_segment_to_widgets(self.segments[idx])
        self._refresh_gating()
        seg = self.segments[idx]
        self._refresh_groove_section(seg)
        self._update_segment_nav_label()
        self._set_status(f"Segment {idx+1} selected (measures {seg.start_bar+1}-{seg.end_bar}).")

    def _on_next_segment(self, event=None):
        if not self.segments:
            return
        idx = 0 if self.selected_index is None else min(self.selected_index + 1, len(self.segments) - 1)
        if idx == self.selected_index:
            return
        self._on_segment_selected(idx)

    def _on_prev_segment(self, event=None):
        if not self.segments:
            return
        idx = 0 if self.selected_index is None else max(self.selected_index - 1, 0)
        if idx == self.selected_index:
            return
        self._on_segment_selected(idx)

    def _update_segment_nav_label(self):
        n = len(self.segments)
        if n == 0:
            self.segment_nav_label.config(text="No segments yet")
            self.prev_segment_btn.config(state='disabled')
            self.next_segment_btn.config(state='disabled')
            return
        if self.selected_index is None:
            self.segment_nav_label.config(text=f"{n} segment{'s' if n != 1 else ''} detected - none selected")
            self.prev_segment_btn.config(state='disabled')
            self.next_segment_btn.config(state='normal')
            return
        self.segment_nav_label.config(text=f"Segment {self.selected_index + 1} of {n}")
        self.prev_segment_btn.config(state='normal' if self.selected_index > 0 else 'disabled')
        self.next_segment_btn.config(state='normal' if self.selected_index < n - 1 else 'disabled')

    # ------------------------------------------------------------- gating --
    def _current_segment(self) -> Optional[SegmentSettings]:
        if self.selected_index is None or self.selected_index >= len(self.segments):
            return None
        return self.segments[self.selected_index]

    def _phase1_raw_input(self, seg: SegmentSettings) -> str:
        """The audio Phase 1 (Humanize) should actually run on: the swapped-in
        library groove if one's active for this segment, else a fresh slice of
        the original recording. Always returns an existing file."""
        if seg.swapped_groove_path and os.path.exists(seg.swapped_groove_path):
            return seg.swapped_groove_path
        raw_path, _, _ = slice_midi_to_temp(self.drum_path, seg.start_sec, seg.end_sec,
                                            self.temp_dir, drums_only=True)
        return raw_path

    def _current_segment_drum_path(self, seg: SegmentSettings) -> str:
        """Best-available drum audio for this segment RIGHT NOW: the most-
        processed phase output if one exists, else the swapped-in groove if
        one's active, else a fresh raw slice of the original recording.
        Always returns an existing file - used by the render step and by
        dragging a segment out of the timeline."""
        path = seg.phase2_output_path or seg.phase1_output_path or seg.swapped_groove_path
        if path and os.path.exists(path):
            return path
        raw_path, _, _ = slice_midi_to_temp(self.drum_path, seg.start_sec, seg.end_sec,
                                            self.temp_dir, drums_only=True)
        return raw_path

    def _current_segment_bass_path(self, seg: SegmentSettings) -> Optional[str]:
        """Best-available BASS audio for this segment right now, for dragging
        a segment out of the bass timeline. None if no bass file is loaded."""
        if not self.bass_path:
            return None
        path = seg.phase3_output_path or seg.bass_input_path
        if path and os.path.exists(path):
            return path
        raw_path, _, _ = slice_midi_to_temp(self.bass_path, seg.start_sec, seg.end_sec,
                                            self.temp_dir, drums_only=False, program=33)
        return raw_path

    def _segment_index_at_x(self, canvas, x) -> Optional[int]:
        """Which segment (by index) sits at local x-coordinate x within canvas,
        using the SAME width computation _draw_segments_on uses - kept separate
        from segment_click_targets since that's only tracked for the drum
        canvas, while this needs to work for either overlay canvas."""
        if not self.segments:
            return None
        canvas_w = max(1, canvas.winfo_width())
        total_bars = sum(s.end_bar - s.start_bar for s in self.segments)
        if total_bars <= 0:
            return None
        acc = 0
        for i, seg in enumerate(self.segments):
            length = seg.end_bar - seg.start_bar
            w = max(SEGMENT_MIN_RECT_WIDTH_PX, round(canvas_w * length / total_bars))
            if acc <= x < acc + w:
                return i
            acc += w
        return len(self.segments) - 1 if self.segments and x >= acc else None

    def _on_segment_drag_init(self, event, canvas, kind):
        """<<DragInitCmd>> handler for dragging a segment rectangle out of the
        timeline. Must return (action, types, data) per tkinterdnd2's
        contract, or None to refuse the drag. data is a real file path - tkdnd
        hands it to the OS as an OLE2 CF_HDROP, so dropping it onto Explorer
        (or any app that accepts dropped files) COPIES that file there."""
        idx = self._segment_index_at_x(canvas, event.x_root - canvas.winfo_rootx())
        if idx is None:
            return None
        seg = self.segments[idx]
        try:
            path = self._current_segment_drum_path(seg) if kind == 'drum' else self._current_segment_bass_path(seg)
        except Exception as exc:
            _report_error(f"preparing segment {idx+1} for drag-out", exc)
            return None
        if not path:
            return None
        # forward slashes - Windows accepts them interchangeably, and this
        # sidesteps any risk of a backslash being misread as a Tcl escape
        # character during tkdnd's native OLE handoff.
        return (COPY, DND_FILES, path.replace('\\', '/'))

    def _refresh_gating(self):
        seg = self._current_segment()
        has_seg = seg is not None
        self.groove_section.set_enabled(has_seg and self.groove_index is not None)
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

    # ==================================================== FIND SIMILAR GROOVE
    def _build_groove_section(self, parent):
        """Lets you replace the SELECTED segment's original drum recording
        with a similar groove from the library (see fit_groove_to_segment),
        before running Phase 1 onward. Results are searched eagerly for every
        segment as soon as segmentation completes (see _start_groove_batch_
        search) so switching segments shows them instantly."""
        self.groove_status_label = ttk.Label(parent, text="Select a segment to see similar grooves.",
                                             foreground='gray')
        self.groove_status_label.pack(anchor='w', padx=8, pady=(4, 2))

        list_frame = ttk.Frame(parent); list_frame.pack(fill='x', padx=8, pady=(0, 4))
        self.groove_tree = ttk.Treeview(list_frame, columns=('rank', 'sim', 'file'),
                                        show='headings', selectmode='browse', height=10)
        self.groove_tree.heading('rank', text='#')
        self.groove_tree.column('rank', width=32, anchor='center', stretch=False)
        self.groove_tree.heading('sim', text='Similarity')
        self.groove_tree.column('sim', width=80, anchor='center', stretch=False)
        self.groove_tree.heading('file', text='File')
        self.groove_tree.column('file', width=300, anchor='w')
        vsb = ttk.Scrollbar(list_frame, orient='vertical', command=self.groove_tree.yview)
        self.groove_tree.configure(yscrollcommand=vsb.set)
        self.groove_tree.pack(side='left', fill='x', expand=True)
        vsb.pack(side='right', fill='y')
        self.groove_tree.bind('<<TreeviewSelect>>', self._on_groove_row_select)
        self.groove_tree.bind('<Double-Button-1>', self._on_groove_row_double_click)
        if HAS_DND:
            # OUTBOUND: dragging a row out of the results list copies that
            # library groove file to wherever it's dropped - same mechanism
            # (and same COPY action) as dragging a segment out of the timeline.
            try:
                self.groove_tree.drag_source_register(1, DND_FILES)
                self.groove_tree.dnd_bind('<<DragInitCmd>>', self._on_groove_row_drag_init)
            except Exception as exc:
                _report_error("enabling groove-list drag-out", exc)

        btnrow = ttk.Frame(parent); btnrow.pack(fill='x', padx=8, pady=(0, 6))
        self.groove_audition_btn = ttk.Button(btnrow, text="▶ Audition",
                                              command=self._on_groove_audition, state='disabled')
        self.groove_audition_btn.pack(side='left')
        self._audition_buttons['groove'] = self.groove_audition_btn
        self.groove_use_btn = ttk.Button(btnrow, text="Use this groove",
                                         command=self._on_use_groove, state='disabled')
        self.groove_use_btn.pack(side='left', padx=6)
        self.groove_revert_btn = ttk.Button(btnrow, text="Revert to original",
                                            command=self._on_revert_groove, state='disabled')
        self.groove_revert_btn.pack(side='left')

    def _refresh_groove_section(self, seg: Optional[SegmentSettings]):
        for iid in self.groove_tree.get_children():
            self.groove_tree.delete(iid)
        self.groove_selected_result = None
        self.groove_audition_btn.config(state='disabled')
        self.groove_use_btn.config(state='disabled')
        if seg is None:
            self.groove_status_label.config(text="Select a segment to see similar grooves.",
                                            foreground='gray')
            self.groove_revert_btn.config(state='disabled')
            return
        self.groove_revert_btn.config(state='normal' if seg.swapped_groove_path else 'disabled')
        if seg.swapped_groove_path:
            using = f"Using: {os.path.basename(seg.swapped_groove_source or seg.swapped_groove_path)}"
        else:
            using = "Using: original recording"
        if seg.groove_results is None:
            suffix = "load a groove index to search" if self.groove_index is None else "searching..."
            self.groove_status_label.config(text=f"{using}  -  {suffix}",
                                            foreground='gray' if self.groove_index is None else '#0066cc')
            return
        if seg.groove_search_error:
            self.groove_status_label.config(text=f"{using}  -  search failed (see console).",
                                            foreground='#b00000')
            return
        for i, r in enumerate(seg.groove_results, 1):
            self.groove_tree.insert('', 'end', iid=str(i - 1),
                                    values=(i, f"{r['similarity']:.3f}", os.path.basename(r['path'])))
        n = len(seg.groove_results)
        self.groove_status_label.config(
            text=f"{using}  -  {n} similar groove{'s' if n != 1 else ''} found.", foreground='gray')

    def _on_groove_row_select(self, event=None):
        sel = self.groove_tree.selection()
        seg = self._current_segment()
        if not sel or seg is None or not seg.groove_results:
            self.groove_selected_result = None
            self.groove_audition_btn.config(state='disabled')
            self.groove_use_btn.config(state='disabled')
            return
        self.groove_selected_result = seg.groove_results[int(sel[0])]
        self.groove_audition_btn.config(state='normal')
        self.groove_use_btn.config(state='normal')

    def _on_groove_row_double_click(self, event):
        """Double-click a row to play it; double-click again (or the row
        currently playing) to stop - same toggle _on_groove_audition already
        implements for the Audition button, just reachable straight from the
        list too."""
        row_iid = self.groove_tree.identify_row(event.y)
        if row_iid:
            self.groove_tree.selection_set(row_iid)
            self._on_groove_row_select()
        self._on_groove_audition()

    def _on_groove_row_drag_init(self, event):
        """<<DragInitCmd>> for the results list - drag whichever row is under
        the pointer, regardless of which row (if any) is currently selected,
        matching how file browsers let you drag an unselected row directly."""
        seg = self._current_segment()
        if seg is None or not seg.groove_results:
            return None
        local_y = event.y_root - self.groove_tree.winfo_rooty()
        row_iid = self.groove_tree.identify_row(local_y)
        if not row_iid:
            return None
        try:
            result = seg.groove_results[int(row_iid)]
        except (ValueError, IndexError):
            return None
        path = result['path']
        if not path or not os.path.exists(path):
            return None
        return (COPY, DND_FILES, path.replace('\\', '/'))

    def _on_groove_audition(self):
        """Toggles like the drum/bass/both audition buttons (see
        _toggle_audition) - only one of the four can ever be playing, sharing
        the same play/stop icon state via _set_audition_playing."""
        if self.playing_kind == 'groove':
            if self.player is not None:
                self.player.stop()
            self._set_audition_playing(None)
            self._set_status("Stopped.")
            return
        if self.groove_selected_result is None or self.player is None:
            return
        seg = self._current_segment()
        path = self.groove_selected_result['path']
        target_tempo = 120.0
        if seg is not None and self.drum_path:
            target_tempo = _tempo_at_time(pretty_midi.PrettyMIDI(self.drum_path), seg.start_sec)
        native_tempo = self.groove_selected_result.get('tempo') or 120.0
        tempo_scale = (target_tempo / native_tempo) if native_tempo > 0 else 1.0
        self._set_audition_playing('groove')
        self._set_status(f"Auditioning '{os.path.basename(path)}'...", busy=True)
        threading.Thread(target=self._groove_audition_worker, args=(path, tempo_scale), daemon=True).start()

    def _groove_audition_worker(self, path, tempo_scale):
        try:
            # force_drum_channel=True - this is a pure drum library file being
            # previewed on its own.
            self.player.play(path, tempo_scale=tempo_scale, force_drum_channel=True,
                             on_finished=lambda err: self.root.after(
                                 0, lambda: self._on_audition_playback_finished('groove', err)))
        except Exception as exc:
            msg = _report_error(f"auditioning '{path}'", exc)
            self.root.after(0, lambda: self._on_audition_playback_finished('groove', msg))

    def _on_use_groove(self):
        seg = self._current_segment()
        if seg is None or self.groove_selected_result is None or self.drum_path is None:
            return
        groove_path = self.groove_selected_result['path']
        target_bars = seg.end_bar - seg.start_bar
        target_tempo = _tempo_at_time(pretty_midi.PrettyMIDI(self.drum_path), seg.start_sec)
        self._set_status(f"Fitting '{os.path.basename(groove_path)}' to segment {seg.index+1}...", busy=True)
        threading.Thread(target=self._use_groove_worker,
                         args=(seg, groove_path, target_bars, target_tempo), daemon=True).start()

    def _use_groove_worker(self, seg, groove_path, target_bars, target_tempo):
        try:
            fitted_path = fit_groove_to_segment(groove_path, target_bars, target_tempo, self.temp_dir)
        except Exception as exc:
            msg = _report_error(f"fitting groove for segment {seg.index+1}", exc)
            self.root.after(0, lambda: messagebox.showerror("Could not use this groove", msg))
            return
        self.root.after(0, lambda: self._on_groove_swapped(seg, fitted_path, groove_path))

    def _on_groove_swapped(self, seg: SegmentSettings, fitted_path, source_path):
        seg.swapped_groove_path = fitted_path
        seg.swapped_groove_source = source_path
        # downstream phase outputs were computed on the OLD input - clear them
        # rather than just flagging "stale", since showing them at all now
        # would be actively wrong (different source audio entirely).
        seg.phase1_done = seg.phase2_done = seg.phase3_done = False
        seg.phase1_output_path = seg.phase2_output_path = seg.phase3_output_path = None
        seg.phase1_raw_path = None
        if self._current_segment() is seg:
            self._refresh_groove_section(seg)
        self._refresh_gating()
        self._draw_timeline()
        self._set_status(f"Segment {seg.index+1} now uses '{os.path.basename(source_path)}'.")

    def _on_revert_groove(self):
        seg = self._current_segment()
        if seg is None or not seg.swapped_groove_path:
            return
        seg.swapped_groove_path = None
        seg.swapped_groove_source = None
        seg.phase1_done = seg.phase2_done = seg.phase3_done = False
        seg.phase1_output_path = seg.phase2_output_path = seg.phase3_output_path = None
        seg.phase1_raw_path = None
        self._refresh_groove_section(seg)
        self._refresh_gating()
        self._draw_timeline()
        self._set_status(f"Segment {seg.index+1} reverted to the original recording.")

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
            raw_path = self._phase1_raw_input(seg)
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
                raw_path = self._phase1_raw_input(s)
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
        # phase1/phase1_raw/phase2 are drum-track outputs; phase3 is the bass-sync
        # output - only force the drum channel for the former, or phase3 would
        # play back sounding like drums instead of bass (see MidiPlayer.play()).
        force_drum_channel = (which != 'phase3')
        threading.Thread(target=self._audition_worker, args=(path, force_drum_channel),
                         daemon=True).start()

    def _audition_worker(self, path, force_drum_channel=True):
        try:
            self.player.play(path, force_drum_channel=force_drum_channel,
                             on_finished=lambda err: self.root.after(
                                 0, lambda: self._set_status(f"Playback error: {err}" if err else "Ready.")))
        except Exception as exc:
            msg = _report_error(f"auditioning '{path}'", exc)
            self.root.after(0, lambda: self._set_status(msg))

    def _set_audition_playing(self, kind):
        """Refreshes all three segment-audition buttons' icons: the active one
        (if any) shows the stop icon, the others show the play icon. kind=None
        means nothing is playing - every button shows play."""
        self.playing_kind = kind
        for k, btn in self._audition_buttons.items():
            icon = '■' if k == kind else '▶'   # ■ stop / ▶ play
            label = btn.cget('text').split(' ', 1)[1]     # keep everything after the icon
            btn.config(text=f"{icon} {label}")

    def _on_audition_playback_finished(self, kind, err):
        """Shared on_finished for all three buttons. Guarded on kind so a stale
        callback from a playback that got superseded by a DIFFERENT button (or
        stopped explicitly) can't clobber whichever one is actually active now -
        see _play_worker's `finally: on_finished(error)`, which fires even when
        playback was interrupted by a newer play() call or an explicit stop()."""
        if self.playing_kind == kind:
            self._set_audition_playing(None)
        self._set_status(f"Playback error: {err}" if err else "Ready.")

    def _toggle_audition(self, kind):
        """Bound to all three segment-audition buttons (drum/bass/both). Clicking
        the button for whatever's currently playing stops it; clicking any other
        one starts it - which, via MidiPlayer.play()'s own stop-before-play,
        also stops whatever else was playing (only one of drum/bass/both plays
        at a time, matching there being one player)."""
        if self.playing_kind == kind:
            if self.player is not None:
                self.player.stop()
            self._set_audition_playing(None)
            self._set_status("Stopped.")
            return
        seg = self._current_segment()
        if seg is None:
            messagebox.showinfo("Select a segment", "Select a segment first (click one in "
                                "the timeline above).")
            return
        if self.player is None:
            return
        source_path = self.drum_path if kind in ('drum', 'both') else self.bass_path
        if not source_path:
            messagebox.showinfo("Nothing to play", f"Load a {'drum' if kind == 'both' else kind} "
                                "MIDI file first.")
            return
        self._set_audition_playing(kind)
        self._set_status(f"Auditioning segment {seg.index+1} ({kind})...", busy=True)
        if kind == 'both':
            threading.Thread(target=self._audition_raw_both_worker,
                             args=(self.drum_path, self.bass_path, seg.start_sec, seg.end_sec),
                             daemon=True).start()
        else:
            threading.Thread(target=self._audition_raw_worker,
                             args=(kind, source_path, seg.start_sec, seg.end_sec), daemon=True).start()

    def _audition_raw_both_worker(self, drum_path, bass_path, start_sec, end_sec):
        try:
            path = audition_both_to_temp(drum_path, bass_path, start_sec, end_sec, self.temp_dir)
        except Exception as exc:
            msg = _report_error("building combined drum+bass audition slice", exc)
            self.root.after(0, lambda: self._on_audition_playback_finished('both', msg))
            return
        self.playing_path = path
        try:
            # force_drum_channel=False - this file already carries the correct
            # per-instrument is_drum/program (drum + bass), forcing everything to
            # the drum channel would make the bass part sound like drums too.
            self.player.play(path, force_drum_channel=False, on_finished=lambda err: self.root.after(
                0, lambda: self._on_audition_playback_finished('both', err)))
        except Exception as exc:
            msg = _report_error(f"auditioning '{path}'", exc)
            self.root.after(0, lambda: self._on_audition_playback_finished('both', msg))

    def _audition_raw_worker(self, kind, source_path, start_sec, end_sec):
        """Plays the currently selected segment's RAW (unprocessed) span straight
        from the source file, forced to a drum kit or bass sound respectively -
        regardless of what channel/program the source file itself used - so it's
        always audible as intended even before any phase has run."""
        try:
            if kind == 'drum':
                path, _, _ = slice_midi_to_temp(source_path, start_sec, end_sec,
                                                self.temp_dir, drums_only=True)
            else:
                # program=33 (Electric Bass, finger) - same GM program the render
                # step's own bass_inst uses, so this sounds like the final output.
                path, _, _ = slice_midi_to_temp(source_path, start_sec, end_sec,
                                                self.temp_dir, drums_only=False, program=33)
        except Exception as exc:
            msg = _report_error(f"slicing segment for {kind} audition", exc)
            self.root.after(0, lambda: self._on_audition_playback_finished(kind, msg))
            return
        self.playing_path = path
        try:
            # Only force the drum channel for the drum slice - the bass slice
            # already carries program=33 correctly and must not be overridden.
            self.player.play(path, force_drum_channel=(kind == 'drum'),
                             on_finished=lambda err: self.root.after(
                                 0, lambda: self._on_audition_playback_finished(kind, err)))
        except Exception as exc:
            msg = _report_error(f"auditioning '{path}'", exc)
            self.root.after(0, lambda: self._on_audition_playback_finished(kind, msg))

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
        self._refresh_groove_section(seg)
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
                # phase2 output > phase1 output > swapped-in groove > fresh raw
                # slice of the original recording - see _current_segment_drum_path.
                drum_src_path = self._current_segment_drum_path(seg)
                src = pretty_midi.PrettyMIDI(drum_src_path)
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

    # -------------------------------------------------------------- session --
    # A session file (.dbhproj) is a zip: session.json (segment settings, phase
    # done-flags, groove-swap choices, source file paths) + files/ (whichever
    # phase OUTPUTS and swapped-in grooves already exist, copied out of
    # self.temp_dir - which is wiped on exit, so without this copy a save
    # would silently point at files that no longer exist next launch). RAW
    # slices (phase1_raw_path, bass_input_path) are NOT bundled - they're cheap
    # to re-slice from the source files on load, unlike a Phase 1 output, which
    # took a real model pass (with sampling randomness - re-running Phase 1
    # from saved SETTINGS alone would not reliably reproduce the same result).
    def _on_save_session(self):
        if not self.segments and not self.drum_path:
            messagebox.showinfo("Nothing to save", "Load a drum MIDI file first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save session as", defaultextension=SESSION_FILE_EXT,
            filetypes=[("Drum+Bass Studio session", f"*{SESSION_FILE_EXT}")])
        if not path:
            return
        try:
            self._save_session_to(path)
        except Exception as exc:
            msg = _report_error(f"saving session to '{path}'", exc)
            messagebox.showerror("Save failed", msg)
            return
        self._set_status(f"Session saved -> {os.path.basename(path)}")

    def _save_session_to(self, path):
        data = {
            'version': SESSION_FILE_VERSION,
            'drum_original_path': self.drum_original_path,
            'bass_path': self.bass_path,
            'seg_threshold': self.seg_threshold_var.get(),
            'selected_index': self.selected_index,
            'segments': [],
        }
        file_entries = []   # (arcname, actual path on disk right now)
        for seg in self.segments:
            entry = {
                'index': seg.index, 'start_sec': seg.start_sec, 'end_sec': seg.end_sec,
                'start_bar': seg.start_bar, 'end_bar': seg.end_bar,
                'phase1_settings': seg.phase1_settings, 'phase1_done': seg.phase1_done,
                'phase1_stale': seg.phase1_stale,
                'phase2_settings': seg.phase2_settings, 'phase2_done': seg.phase2_done,
                'phase2_stale': seg.phase2_stale,
                'phase3_settings': seg.phase3_settings, 'phase3_done': seg.phase3_done,
                'phase3_stale': seg.phase3_stale,
                'swapped_groove_source': seg.swapped_groove_source,
                'groove_results': seg.groove_results,
            }
            for field_name, arc_prefix in (('phase1_output_path', 'phase1'),
                                           ('phase2_output_path', 'phase2'),
                                           ('phase3_output_path', 'phase3'),
                                           ('swapped_groove_path', 'swap')):
                src = getattr(seg, field_name)
                if src and os.path.exists(src):
                    arcname = f"files/seg{seg.index}_{arc_prefix}{os.path.splitext(src)[1]}"
                    file_entries.append((arcname, src))
                    entry[field_name] = arcname
                else:
                    entry[field_name] = None
            data['segments'].append(entry)

        tmp_path = path + '.tmp'
        try:
            with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('session.json', json.dumps(data, indent=2))
                for arcname, src in file_entries:
                    zf.write(src, arcname)
            os.replace(tmp_path, path)   # atomic - a reader never sees a half-written project
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _on_load_session(self):
        path = filedialog.askopenfilename(
            title="Load session", filetypes=[("Drum+Bass Studio session", f"*{SESSION_FILE_EXT}"),
                                             ("All files", "*.*")])
        if not path:
            return
        try:
            warnings = self._load_session_from(path)
        except Exception as exc:
            msg = _report_error(f"loading session '{path}'", exc)
            messagebox.showerror("Load failed", msg)
            return
        msg = f"Session loaded ({len(self.segments)} segments)."
        self._set_status(msg + (" See warnings." if warnings else ""))
        if warnings:
            messagebox.showwarning("Session loaded with warnings", "\n".join(warnings))

    def _load_session_from(self, path) -> List[str]:
        extract_dir = os.path.join(self.temp_dir, f"session_{uuid.uuid4().hex[:8]}")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(path, 'r') as zf:
            data = json.loads(zf.read('session.json'))
            zf.extractall(extract_dir)

        warnings = []

        # -- drum file: re-derive the forced temp copy fresh from the ORIGINAL
        # path (the old temp copy is long gone - self.temp_dir is wiped on
        # every exit) --
        drum_original = data.get('drum_original_path')
        self.drum_original_path = drum_original
        if drum_original and os.path.exists(drum_original):
            try:
                self.drum_path = force_all_drums_to_temp(drum_original, self.temp_dir)
                self.drum_drop.config(text=os.path.basename(drum_original), foreground='black')
            except Exception as exc:
                _report_error(f"restoring drum file '{drum_original}'", exc)
                warnings.append(f"Could not reload the drum file: {drum_original}")
                self.drum_path = None
        else:
            self.drum_path = None
            self.drum_drop.config(text=self._drop_text("drum"), foreground='#555')
            if drum_original:
                warnings.append(f"Original drum file not found (moved or deleted?): {drum_original}")

        # -- bass file (never temp-forced, so the saved path is usable directly
        # if it still exists) --
        bass_path = data.get('bass_path')
        if bass_path and os.path.exists(bass_path):
            self.bass_path = bass_path
            self.bass_drop.config(text=os.path.basename(bass_path), foreground='black')
        else:
            self.bass_path = None
            self.bass_drop.config(text=self._drop_text("bass"), foreground='#555')
            if bass_path:
                warnings.append(f"Bass file not found (moved or deleted?): {bass_path}")

        if 'seg_threshold' in data:
            self.seg_threshold_var.set(data['seg_threshold'])
            self.seg_threshold_label.config(text=f"{data['seg_threshold']:.2f}")

        segments = []
        for entry in data.get('segments', []):
            seg = SegmentSettings(index=entry['index'], start_sec=entry['start_sec'],
                                  end_sec=entry['end_sec'], start_bar=entry['start_bar'],
                                  end_bar=entry['end_bar'])
            seg.phase1_settings = entry.get('phase1_settings') or default_phase1_settings()
            seg.phase2_settings = entry.get('phase2_settings') or default_phase2_settings()
            seg.phase3_settings = entry.get('phase3_settings') or default_phase3_settings()
            seg.phase1_stale = entry.get('phase1_stale', False)
            seg.phase2_stale = entry.get('phase2_stale', False)
            seg.phase3_stale = entry.get('phase3_stale', False)
            seg.swapped_groove_source = entry.get('swapped_groove_source')
            seg.groove_results = entry.get('groove_results')

            for field_name in ('phase1_output_path', 'phase2_output_path',
                               'phase3_output_path', 'swapped_groove_path'):
                arcname = entry.get(field_name)
                if arcname:
                    full = os.path.join(extract_dir, *arcname.split('/'))
                    setattr(seg, field_name, full if os.path.exists(full) else None)

            seg.phase1_done = bool(seg.phase1_output_path) and entry.get('phase1_done', False)
            seg.phase2_done = bool(seg.phase2_output_path) and entry.get('phase2_done', False)
            seg.phase3_done = bool(seg.phase3_output_path) and entry.get('phase3_done', False)

            # Cheap to regenerate (a raw slice, no model involved) - restoring
            # these means "Audition raw"/Phase 3's bass sync still work right
            # away instead of only after the phase is re-run.
            if self.drum_path and seg.phase1_done:
                try:
                    seg.phase1_raw_path = self._phase1_raw_input(seg)
                except Exception as exc:
                    _report_error(f"regenerating raw input for segment {seg.index+1}", exc)
            if self.bass_path and seg.phase3_done:
                try:
                    seg.bass_input_path, _, _ = slice_midi_to_temp(
                        self.bass_path, seg.start_sec, seg.end_sec, self.temp_dir, drums_only=False)
                except Exception as exc:
                    _report_error(f"regenerating raw bass input for segment {seg.index+1}", exc)

            segments.append(seg)

        self.segments = segments
        self.selected_index = None
        self._draw_timeline()
        self._update_segment_nav_label()
        self._refresh_gating()
        self._refresh_groove_section(None)

        sel = data.get('selected_index')
        if sel is not None and 0 <= sel < len(self.segments):
            self._on_segment_selected(sel)

        return warnings

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
