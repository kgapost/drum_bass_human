#!/usr/bin/env python3
"""
==============================================================================
 GROOVE FINDER — desktop UI for find_similar_grooves.py  (Windows)
==============================================================================

WHAT THIS IS
------------
A small Tkinter desktop app around find_similar_grooves.py: drop in a query
MIDI, see a ranked list of the most similar grooves in your indexed library,
and audition any of them WITHOUT leaving the app or losing window focus.

AUDITION: HOW PLAYBACK WORKS (no external program, no bundled soundfont)
--------------------------------------------------------------------------
Windows ships a built-in General MIDI software synth ("Microsoft GS Wavetable
Synth") that is always available as a MIDI output device — no install needed.
This app talks to it DIRECTLY over Python's MIDI I/O (via `mido` + the
`python-rtmidi` backend), sending the file's note events to it in real time
on a background thread. That means:
  • No separate program window opens (nothing steals focus).
  • Real General MIDI drum-kit sounds (kick/snare/hats/etc.), not a beep.
  • Audio quality is whatever that built-in synth sounds like — this
    deliberately trades fidelity for staying in-process and simple.
The Audition button toggles to a Stop button while playing; Stop immediately
halts playback and sends an all-notes-off safety message so nothing hangs.

REQUIREMENTS (Windows)
-----------------------
  pip install pretty_midi numpy mido python-rtmidi tkinterdnd2
  (tkinter itself ships with the standard python.org Windows installer)

This file must sit in the SAME FOLDER as find_similar_grooves.py — it imports
that module directly rather than duplicating its logic.

HOW TO USE
----------
  python groove_finder_ui.py

  1. Point "Index" at a cache built with:
       python find_similar_grooves.py --mode index --data_dir "..." --cache cache/groove_index.pkl
  2. Drop a query MIDI file onto the drop zone (or click Browse).
  3. Results appear, sorted by similarity (best first).
  4. Click a row, then Audition to hear it play through Windows' built-in
     synth; click again (now labeled Stop) to stop immediately.

TESTING NOTE: this was built and verified in a headless Linux sandbox, where
real Windows MIDI hardware/synths and real OS drag-and-drop gestures cannot
exist. The playback engine was verified against a mock MIDI output (message
order, channel handling, stop responsiveness, no-stuck-notes), and the GUI
was verified under a virtual display (widgets build, state transitions fire
correctly). Actual audible sound and actual drag-and-drop from Windows
Explorer need to be confirmed on your machine — see the summary for specifics.
"""

import os
import sys
import time
import json
import uuid
import shutil
import tempfile
import threading
import traceback
from typing import Optional, List, Dict, Callable

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

try:
    import mido
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False

# find_similar_grooves.py must be alongside this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import find_similar_grooves as fsg
except ImportError as exc:
    print(f"FATAL: could not import find_similar_grooves.py — make sure it is "
          f"in the same folder as this script. ({exc})")
    sys.exit(1)

# drum_theme_segmentation.py is OPTIONAL — the app works fully without it, just
# without the segment-timeline feature (needs torch, which isn't otherwise required).
try:
    import drum_theme_segmentation as dts
    HAS_SEGMENTATION = True
except ImportError:
    HAS_SEGMENTATION = False



# =============================================================================
# ERROR REPORTING HELPERS  (same pattern as the other project scripts)
# =============================================================================

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
    msg = f"{context}\n→ {type(exc).__name__} at {loc}: {exc}"
    print(f"[ERROR] {msg}")
    return msg


# =============================================================================
# PERSISTENT SETTINGS  (remember the last-used index / model across restarts)
# =============================================================================
# DESIGN: building/pointing the app at an index is meant to be a ONE-TIME setup
# step — the index itself is already saved to disk by build_index(). Without
# this, though, the APP would still forget where that file is every time it's
# relaunched, forcing a manual "Load Index..." every session. This small JSON
# file (kept next to the script) closes that gap: whatever index/model loaded
# successfully is remembered and auto-reloaded next launch, silently skipped
# if the file no longer exists (e.g. moved/deleted) rather than erroring.

def _settings_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'groove_finder_settings.json')


def _load_persisted_settings() -> Dict:
    try:
        with open(_settings_path(), 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_persisted_settings(**kv):
    data = _load_persisted_settings()
    data.update({k: v for k, v in kv.items() if v is not None})
    try:
        with open(_settings_path(), 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        _report_error("saving app settings (last-used index/model won't be "
                      "remembered next launch)", exc)


def _tempo_at_time(pm, t: float) -> float:
    """Return the tempo (bpm) actually in effect at absolute time t, tempo-change
    aware — the LAST tempo change at or before t, not blindly the file's first."""
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


# =============================================================================
# MIDI PLAYBACK ENGINE
# =============================================================================
# DESIGN: the output port is an INJECTED dependency (anything with a .send(msg)
# method works) specifically so this class can be unit-tested with a mock
# recorder instead of real MIDI hardware — see the test suite at the bottom of
# this file. On a real Windows machine, `mido.open_output()` with no argument
# name opens the system default port, which is the built-in GS Wavetable Synth.

class MidiPlayer:
    """Plays one MIDI file at a time on a background thread. Forces all note
    events onto MIDI channel 10 (GM drum channel) so playback always uses the
    synth's drum kit sounds regardless of the source file's original channel —
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
            raise RuntimeError("mido is not installed — run: pip install mido python-rtmidi")
        try:
            names = mido.get_output_names()
        except Exception as exc:
            raise RuntimeError(f"could not list MIDI output ports ({exc}). Is "
                               f"python-rtmidi installed? (pip install python-rtmidi)")
        if not names:
            raise RuntimeError("no MIDI output ports found on this system — "
                               "Windows should always have 'Microsoft GS Wavetable "
                               "Synth'; check Windows Sound settings / MIDI devices.")
        # DESIGN: prefer the built-in Windows synth by name if it's present, rather
        # than trusting whichever port happens to enumerate first — a machine with
        # other MIDI hardware/software installed could otherwise route audition
        # to something unexpected (or silent).
        preferred = next((n for n in names if 'gs wavetable' in n.lower()), None)
        return mido.open_output(preferred or names[0])

    def _ensure_port(self):
        if self._outport is None:
            self._outport = self._outport_factory()
        return self._outport

    def play(self, path: str, on_finished: Optional[Callable[[Optional[str]], None]] = None,
             tempo_scale: float = 1.0):
        """Start playback in the background. on_finished(error_message_or_None)
        is called (from the playback thread) when playback ends, whether by
        completing naturally, being stopped, or erroring.
        tempo_scale: uniformly speeds up (>1) or slows down (<1) the WHOLE
        performance's timing — e.g. 1.5 plays 50% faster. Used to retime a
        library groove to the query's tempo when auditioning it; leave at 1.0
        (default) to play a file at its own native tempo, unchanged."""
        if self.is_playing:
            self.stop()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
        if tempo_scale is None or tempo_scale <= 0:
            tempo_scale = 1.0   # defensive: a bad ratio must never divide-by-zero or reverse time
        self._stop_event.clear()
        self.is_playing = True
        self._thread = threading.Thread(target=self._play_worker,
                                        args=(path, on_finished, tempo_scale), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _play_worker(self, path: str, on_finished, tempo_scale: float = 1.0):
        error = None
        try:
            port = self._ensure_port()
            midi = mido.MidiFile(path)
            # DESIGN: mido.MidiFile.play() sleeps INSIDE its generator between
            # messages, so the stop flag would only be checked once per message —
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
                # uniformly (not the file's own embedded tempo value itself) — this
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
                if hasattr(msg, 'channel'):
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
# UI APPLICATION
# =============================================================================
# DESIGN: every find_similar_grooves.py CLI argument is exposed as a live
# control here — the two INDEX-time args (data_dir/cache/num_workers/min_notes/
# 5 velocity floors) live in a "Build New Index" panel; the QUERY-time args
# (top_k/exclude_same_family/7 weights) live in a "Query Settings" panel. Both
# panels sit in a Settings dialog reachable from the main window, since ~15
# controls would clutter the simple main view. Query settings are read LIVE
# (Tk variables) every time a search runs, so you can tweak a weight and hit
# "Search Again" without re-dropping the query file.
#
# Query computation, index loading/building, and playback start/stop ALL run
# on background threads and hand results back via `root.after(0, ...)` — the
# standard safe pattern for touching Tkinter widgets from a worker thread —
# so nothing in this app can freeze the window.

RESULTS_LIMIT_DEFAULT = 10

# Qualitative palette for the segment timeline — distinguishable, cycles if there
# are more segments than colors.
SEGMENT_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2',
                  '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']


class GrooveFinderApp:
    def __init__(self, root):
        self.root = root
        root.title("Groove Finder")
        root.geometry("780x560")
        root.minsize(580, 420)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.index: Optional[Dict] = None
        self.index_path: Optional[str] = None
        self.results: List[Dict] = []
        self.player = MidiPlayer()
        self.playing_path: Optional[str] = None
        self.playing_source: Optional[str] = None   # 'result' or 'segment'
        self.last_query_path: Optional[str] = None
        self.settings_win: Optional[tk.Toplevel] = None

        # ── Segmentation state ──────────────────────────────────────────────────
        self.seg_model = None
        self.seg_cfg = None
        self.seg_checkpoint_path: Optional[str] = None
        self.current_full_query_path: Optional[str] = None   # the ORIGINAL dropped
        # file — distinct from last_query_path, which may point at a segment's
        # sliced temp file. Lets the "Whole File" button always get back to it.
        self.whole_file_tempo: Optional[float] = None   # tempo at t=0 of that file
        self.segments: List[Dict] = []
        self.active_segment_index: Optional[int] = None
        self.active_segment_temp_path: Optional[str] = None
        self.temp_dir = tempfile.mkdtemp(prefix='groove_finder_')

        # ── Query-time controls: every Config query-weight field + top_k +
        # exclude_same_family. Live Tk variables — read fresh on every search. ──
        _d = fsg.Config()
        self.var_top_k               = tk.IntVar(value=RESULTS_LIMIT_DEFAULT)
        self.var_exclude_same_family = tk.BooleanVar(value=False)
        self.var_weight_rhythm         = tk.DoubleVar(value=_d.weight_rhythm)
        self.var_weight_velocity       = tk.DoubleVar(value=_d.weight_velocity)
        self.var_weight_density        = tk.DoubleVar(value=_d.weight_density)
        self.var_weight_tempo          = tk.DoubleVar(value=_d.weight_tempo)
        self.var_weight_hihat_pattern  = tk.DoubleVar(value=_d.weight_hihat_pattern)
        self.var_weight_tom_pattern    = tk.DoubleVar(value=_d.weight_tom_pattern)
        self.var_weight_cymbal_pattern = tk.DoubleVar(value=_d.weight_cymbal_pattern)

        # ── Index-build controls: every build_index / index-time Config field. ──
        self.var_build_data_dir = tk.StringVar(value="")
        self.var_build_cache    = tk.StringVar(value="cache/groove_index.pkl")
        self.var_build_workers  = tk.IntVar(value=8)
        self.var_build_min_notes = tk.IntVar(value=_d.min_notes)
        self.var_build_min_vel_kick    = tk.IntVar(value=_d.min_velocity_kick)
        self.var_build_min_vel_snare   = tk.IntVar(value=_d.min_velocity_snare)
        self.var_build_min_vel_hihat   = tk.IntVar(value=_d.min_velocity_hihat)
        self.var_build_min_vel_toms    = tk.IntVar(value=_d.min_velocity_toms)
        self.var_build_min_vel_cymbals = tk.IntVar(value=_d.min_velocity_cymbals)

        # ── Segmentation-time controls ──────────────────────────────────────────
        self.var_seg_threshold = tk.DoubleVar(value=0.5)
        self.var_seg_overlap   = tk.DoubleVar(value=0.25)

        self._build_widgets()
        self._set_status("Load an index, then drop a query MIDI file.", busy=False)
        self._auto_load_persisted_settings()

    def _auto_load_persisted_settings(self):
        saved = _load_persisted_settings()
        idx_path = saved.get('index_path')
        if idx_path and os.path.exists(idx_path):
            self._set_status(f"Loading last-used index ({os.path.basename(idx_path)})...", busy=True)
            threading.Thread(target=self._load_index_worker, args=(idx_path,), daemon=True).start()
        if HAS_SEGMENTATION:
            seg_path = saved.get('seg_checkpoint_path')
            if seg_path and os.path.exists(seg_path):
                threading.Thread(target=self._load_seg_model_worker, args=(seg_path,), daemon=True).start()

    # ---- main window layout ------------------------------------------------
    def _build_widgets(self):
        top = ttk.Frame(self.root)
        top.pack(fill='x', padx=8, pady=(8, 4))
        ttk.Label(top, text="Index:").pack(side='left')
        self.index_label = ttk.Label(top, text="(none loaded)", foreground='gray')
        self.index_label.pack(side='left', padx=6)
        ttk.Button(top, text="Settings...", command=self._open_settings).pack(side='right')
        ttk.Button(top, text="Load Index...", command=self._on_load_index).pack(side='right', padx=(0, 6))

        self.drop_zone = tk.Label(
            self.root, text=self._drop_zone_text(), relief='groove', bd=2,
            height=3, bg='#f5f5f5', fg='#555', cursor='hand2', justify='center')
        self.drop_zone.pack(fill='x', padx=8, pady=4)
        self.drop_zone.bind('<Button-1>', self._on_browse_query)
        self.dnd_active = False
        if HAS_DND:
            # DESIGN: a successful `import tkinterdnd2` only proves the Python
            # package is installed — the underlying tkdnd Tcl extension can still
            # fail to register on some systems (version/platform quirks). Drag-
            # and-drop is a nicety with a working click-to-browse fallback, so a
            # failure here must degrade gracefully, never crash the whole app.
            try:
                self.drop_zone.drop_target_register(DND_FILES)
                self.drop_zone.dnd_bind('<<Drop>>', self._on_drop)
                self.dnd_active = True
            except Exception as exc:
                _report_error("enabling drag-and-drop (falling back to click-to-browse)", exc)
                self.drop_zone.config(text="Click to browse for a query MIDI file\n"
                                           "(drag-and-drop unavailable on this system)")

        # ── Segment timeline: hidden (0 height) until segments exist, so it
        # never clutters the window when no segmentation model is loaded. ──────
        seg_frame = ttk.Frame(self.root)
        seg_frame.pack(fill='x', padx=8, pady=(0, 4))
        seg_header = ttk.Frame(seg_frame)
        seg_header.pack(fill='x')
        self.seg_label = ttk.Label(seg_header, text="", foreground='gray')
        self.seg_label.pack(side='left')
        self.segment_audition_btn = ttk.Button(seg_header, text="▶ Audition Segment",
                                               command=self._on_segment_audition_clicked,
                                               state='disabled')
        self.segment_audition_btn.pack(side='right', padx=(6, 0))
        self.whole_file_btn = ttk.Button(seg_header, text="⬜ Whole File",
                                         command=self._on_whole_file_clicked, state='disabled')
        self.whole_file_btn.pack(side='right')
        self.seg_canvas = tk.Canvas(seg_frame, height=0, bg='#e8e8e8', highlightthickness=0)
        self.seg_canvas.pack(fill='x', pady=(2, 0))
        self.segment_click_targets: Dict[int, Dict] = {}   # canvas item id -> segment dict

        tree_frame = ttk.Frame(self.root)
        tree_frame.pack(fill='both', expand=True, padx=8, pady=4)
        columns = ('rank', 'sim', 'file')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', selectmode='browse')
        self.tree.heading('rank', text='#')
        self.tree.column('rank', width=40, anchor='center', stretch=False)
        self.tree.heading('sim', text='Similarity')
        self.tree.column('sim', width=90, anchor='center', stretch=False)
        self.tree.heading('file', text='File')
        self.tree.column('file', width=560, anchor='w')
        vsb = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        self.tree.bind('<<TreeviewSelect>>', self._on_row_select)

        bottom = ttk.Frame(self.root)
        bottom.pack(fill='x', padx=8, pady=(4, 8))
        self.audition_btn = ttk.Button(bottom, text="▶ Audition",
                                       command=self._on_audition_clicked, state='disabled')
        self.audition_btn.pack(side='left')
        self.save_btn = ttk.Button(bottom, text="💾 Save...",
                                   command=self._on_save_clicked, state='disabled')
        self.save_btn.pack(side='left', padx=(6, 0))
        self.research_btn = ttk.Button(bottom, text="🔄 Search Again",
                                       command=self._on_search_again, state='disabled')
        self.research_btn.pack(side='left', padx=(6, 0))
        self.status_label = ttk.Label(bottom, text="", foreground='gray')
        self.status_label.pack(side='left', padx=10)

    def _drop_zone_text(self):
        if HAS_DND:
            return "Drop a query MIDI file here\n(or click to browse)"
        return "Click to browse for a query MIDI file\n(install tkinterdnd2 for drag-and-drop)"

    # ---- settings dialog ----------------------------------------------------
    def _open_settings(self):
        if self.settings_win is not None and self.settings_win.winfo_exists():
            self.settings_win.lift()
            return
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("480x560")
        win.transient(self.root)
        self.settings_win = win

        nb = ttk.Notebook(win)
        nb.pack(fill='both', expand=True, padx=8, pady=8)

        query_tab = ttk.Frame(nb)
        build_tab = ttk.Frame(nb)
        nb.add(query_tab, text="Query Settings")
        nb.add(build_tab, text="Build New Index")
        self._build_query_settings_tab(query_tab)
        self._build_index_settings_tab(build_tab)

        if HAS_SEGMENTATION:
            seg_tab = ttk.Frame(nb)
            nb.add(seg_tab, text="Segmentation")
            self._build_segmentation_settings_tab(seg_tab)
        else:
            hint = ttk.Frame(nb)
            nb.add(hint, text="Segmentation")
            ttk.Label(hint, text="drum_theme_segmentation.py (and torch) were not "
                     "found — segment-timeline is disabled.\nEverything else works "
                     "normally.", foreground='gray', wraplength=400,
                     justify='left').pack(padx=12, pady=12)

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 8))

    def _build_segmentation_settings_tab(self, parent):
        pad = dict(padx=8, pady=4)
        row = ttk.Frame(parent); row.pack(fill='x', **pad)
        ttk.Label(row, text="Model:", width=10, anchor='w').pack(side='left')
        self.seg_model_label = ttk.Label(row, text="(none loaded)", foreground='gray')
        self.seg_model_label.pack(side='left', padx=6)
        ttk.Button(row, text="Load Segmentation Model...",
                  command=self._on_load_seg_model).pack(side='right')

        ttk.Separator(parent).pack(fill='x', padx=8, pady=8)
        row = ttk.Frame(parent); row.pack(fill='x', **pad)
        ttk.Label(row, text="Threshold:", width=16, anchor='w').pack(side='left')
        ttk.Spinbox(row, textvariable=self.var_seg_threshold, from_=0.05, to=0.95,
                   increment=0.05, width=8, format='%.2f').pack(side='left')
        ttk.Label(row, text="probability to call a measure a theme start",
                 foreground='gray').pack(side='left', padx=6)

        row = ttk.Frame(parent); row.pack(fill='x', **pad)
        ttk.Label(row, text="Context overlap:", width=16, anchor='w').pack(side='left')
        ttk.Spinbox(row, textvariable=self.var_seg_overlap, from_=0.0, to=0.9,
                   increment=0.05, width=8, format='%.2f').pack(side='left')
        ttk.Label(row, text="chunk overlap for long files",
                 foreground='gray').pack(side='left', padx=6)

        ttk.Separator(parent).pack(fill='x', padx=8, pady=8)
        ttk.Label(parent, text="A segment timeline appears above the results list "
                 "after you drop a query MIDI, once a model is loaded here. Click "
                 "a segment to search for grooves similar to JUST that section.",
                 foreground='gray', wraplength=400, justify='left').pack(padx=8, pady=4)

    def _on_load_seg_model(self):
        path = filedialog.askopenfilename(
            title="Select segmentation model checkpoint",
            filetypes=[("Checkpoint", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        self._set_status("Loading segmentation model...", busy=True)
        threading.Thread(target=self._load_seg_model_worker, args=(path,), daemon=True).start()

    def _load_seg_model_worker(self, path):
        try:
            device = dts.torch.device('cuda' if dts.torch.cuda.is_available()
                                      else 'mps' if dts.torch.backends.mps.is_available() else 'cpu')
            model, cfg = dts.load_model(path, device)
        except Exception as exc:
            msg = _report_error(f"loading segmentation model '{path}'", exc)
            self.root.after(0, lambda: self._on_seg_model_load_error(msg))
            return
        self.root.after(0, lambda: self._on_seg_model_loaded(path, model, cfg))

    def _on_seg_model_loaded(self, path, model, cfg):
        self.seg_model = model
        self.seg_cfg = cfg
        self.seg_checkpoint_path = path
        if hasattr(self, 'seg_model_label'):
            self.seg_model_label.config(text=os.path.basename(path), foreground='black')
        self._set_status(f"Segmentation model loaded ({os.path.basename(path)}).", busy=False)
        _save_persisted_settings(seg_checkpoint_path=path)
        # if a query is already showing, (re-)segment it now
        if self.current_full_query_path:
            self._start_segmentation(self.current_full_query_path)

    def _on_seg_model_load_error(self, msg):
        messagebox.showerror("Failed to load segmentation model", msg)
        self._set_status("Failed to load segmentation model.", busy=False)

    def _build_query_settings_tab(self, parent):
        pad = dict(padx=8, pady=4)

        def int_row(label, var, frm, to, help_text=None):
            row = ttk.Frame(parent); row.pack(fill='x', **pad)
            ttk.Label(row, text=label, width=22, anchor='w').pack(side='left')
            ttk.Spinbox(row, textvariable=var, from_=frm, to=to, width=8).pack(side='left')
            if help_text:
                ttk.Label(row, text=help_text, foreground='gray').pack(side='left', padx=6)

        def weight_row(label, var, help_text=None):
            row = ttk.Frame(parent); row.pack(fill='x', **pad)
            ttk.Label(row, text=label, width=22, anchor='w').pack(side='left')
            ttk.Spinbox(row, textvariable=var, from_=0.0, to=1.0, increment=0.05,
                       width=8, format='%.2f').pack(side='left')
            if help_text:
                ttk.Label(row, text=help_text, foreground='gray').pack(side='left', padx=6)

        ttk.Label(parent, text="Result count", font=('', 9, 'bold')).pack(anchor='w', padx=8, pady=(8, 0))
        int_row("Top K results:", self.var_top_k, 1, 200)
        row = ttk.Frame(parent); row.pack(fill='x', **pad)
        ttk.Checkbutton(row, text="Exclude same-family filenames (e.g. \"Fill 1\" vs \"Fill 14\")",
                        variable=self.var_exclude_same_family).pack(side='left')

        ttk.Separator(parent).pack(fill='x', padx=8, pady=8)
        ttk.Label(parent, text="Core similarity weights", font=('', 9, 'bold')).pack(anchor='w', padx=8)
        weight_row("Rhythm:", self.var_weight_rhythm, "pattern shape")
        weight_row("Velocity:", self.var_weight_velocity, "dynamics profile")
        weight_row("Density:", self.var_weight_density, "notes/bar closeness")
        weight_row("Tempo:", self.var_weight_tempo, "BPM closeness")

        ttk.Separator(parent).pack(fill='x', padx=8, pady=8)
        ttk.Label(parent, text="Optional articulation-flattened patterns (off = 0.0)",
                 font=('', 9, 'bold')).pack(anchor='w', padx=8)
        weight_row("Hi-hat pattern:", self.var_weight_hihat_pattern, "any articulation")
        weight_row("Tom pattern:", self.var_weight_tom_pattern, "floor vs rack")
        weight_row("Cymbal pattern:", self.var_weight_cymbal_pattern, "any cymbal")

        ttk.Separator(parent).pack(fill='x', padx=8, pady=8)
        ttk.Button(parent, text="Search Again with these settings",
                  command=self._on_search_again).pack(padx=8, pady=4, anchor='w')

    def _build_index_settings_tab(self, parent):
        pad = dict(padx=8, pady=4)

        row = ttk.Frame(parent); row.pack(fill='x', **pad)
        ttk.Label(row, text="MIDI library folder:", width=18, anchor='w').pack(side='left')
        ttk.Entry(row, textvariable=self.var_build_data_dir).pack(side='left', fill='x', expand=True)
        ttk.Button(row, text="Browse...", command=self._on_browse_data_dir).pack(side='left', padx=(4, 0))

        row = ttk.Frame(parent); row.pack(fill='x', **pad)
        ttk.Label(row, text="Save index to:", width=18, anchor='w').pack(side='left')
        ttk.Entry(row, textvariable=self.var_build_cache).pack(side='left', fill='x', expand=True)
        ttk.Button(row, text="Browse...", command=self._on_browse_cache_out).pack(side='left', padx=(4, 0))

        def int_row(label, var, frm, to):
            r = ttk.Frame(parent); r.pack(fill='x', **pad)
            ttk.Label(r, text=label, width=22, anchor='w').pack(side='left')
            ttk.Spinbox(r, textvariable=var, from_=frm, to=to, width=8).pack(side='left')

        ttk.Separator(parent).pack(fill='x', padx=8, pady=8)
        int_row("Worker processes:", self.var_build_workers, 1, 32)
        int_row("Min notes per file:", self.var_build_min_notes, 0, 200)

        ttk.Separator(parent).pack(fill='x', padx=8, pady=8)
        ttk.Label(parent, text="Ignore notes below this velocity (0 = no filtering)",
                 font=('', 9, 'bold')).pack(anchor='w', padx=8)
        int_row("Kick:", self.var_build_min_vel_kick, 0, 127)
        int_row("Snare:", self.var_build_min_vel_snare, 0, 127)
        int_row("Hi-hat:", self.var_build_min_vel_hihat, 0, 127)
        int_row("Toms:", self.var_build_min_vel_toms, 0, 127)
        int_row("Cymbals:", self.var_build_min_vel_cymbals, 0, 127)

        ttk.Separator(parent).pack(fill='x', padx=8, pady=8)
        self.build_btn = ttk.Button(parent, text="Build Index", command=self._on_build_index)
        self.build_btn.pack(padx=8, pady=4, anchor='w')

    def _on_browse_data_dir(self):
        path = filedialog.askdirectory(title="Select your MIDI library folder")
        if path:
            self.var_build_data_dir.set(path)

    def _on_browse_cache_out(self):
        path = filedialog.asksaveasfilename(title="Save index cache as", defaultextension=".pkl",
                                            filetypes=[("Index cache", "*.pkl")])
        if path:
            self.var_build_cache.set(path)

    def _on_build_index(self):
        data_dir = self.var_build_data_dir.get().strip()
        cache_path = self.var_build_cache.get().strip()
        if not data_dir:
            messagebox.showwarning("Missing folder", "Choose a MIDI library folder first.")
            return
        if not cache_path:
            messagebox.showwarning("Missing path", "Choose where to save the index first.")
            return
        cfg = fsg.Config(
            min_notes=self.var_build_min_notes.get(),
            min_velocity_kick=self.var_build_min_vel_kick.get(),
            min_velocity_snare=self.var_build_min_vel_snare.get(),
            min_velocity_hihat=self.var_build_min_vel_hihat.get(),
            min_velocity_toms=self.var_build_min_vel_toms.get(),
            min_velocity_cymbals=self.var_build_min_vel_cymbals.get(),
        )
        self.build_btn.config(state='disabled')
        self._set_status(f"Building index from '{data_dir}'...", busy=True)
        threading.Thread(target=self._build_index_worker,
                         args=(data_dir, cache_path, cfg, self.var_build_workers.get()),
                         daemon=True).start()

    def _build_index_worker(self, data_dir, cache_path, cfg, num_workers):
        try:
            fsg.build_index(data_dir, cache_path, cfg, num_workers=num_workers)
        except Exception as exc:
            msg = _report_error(f"building index from '{data_dir}'", exc)
            self.root.after(0, lambda: self._on_build_index_error(msg))
            return
        self.root.after(0, lambda: self._on_build_index_done(cache_path))

    def _on_build_index_done(self, cache_path):
        self.build_btn.config(state='normal')
        self._set_status(f"Index built → {cache_path}", busy=False)
        if messagebox.askyesno("Index built", f"Index built successfully:\n{cache_path}\n\n"
                               f"Load it now?"):
            self._set_status("Loading index...", busy=True)
            threading.Thread(target=self._load_index_worker, args=(cache_path,), daemon=True).start()

    def _on_build_index_error(self, msg):
        self.build_btn.config(state='normal')
        messagebox.showerror("Failed to build index", msg)
        self._set_status("Failed to build index.", busy=False)

    # ---- index loading ----------------------------------------------------
    def _on_load_index(self):
        path = filedialog.askopenfilename(
            title="Select groove index cache",
            filetypes=[("Index cache", "*.pkl"), ("All files", "*.*")])
        if not path:
            return
        self._set_status("Loading index...", busy=True)
        threading.Thread(target=self._load_index_worker, args=(path,), daemon=True).start()

    def _load_index_worker(self, path):
        try:
            index = fsg.load_index(path)
        except Exception as exc:
            msg = _report_error(f"loading index '{path}'", exc)
            self.root.after(0, lambda: self._on_index_load_error(msg))
            return
        self.root.after(0, lambda: self._on_index_loaded(path, index))

    def _on_index_loaded(self, path, index):
        self.index = index
        self.index_path = path
        n = len(index.get('paths', []))
        self.index_label.config(text=f"{os.path.basename(path)}  ({n} files)", foreground='black')
        self._set_status(f"Index loaded ({n} files). Drop a query MIDI to search.", busy=False)
        _save_persisted_settings(index_path=path)

    def _on_index_load_error(self, msg):
        messagebox.showerror("Failed to load index", msg)
        self._set_status("Failed to load index.", busy=False)

    # ---- query input --------------------------------------------------------
    def _on_browse_query(self, event=None):
        path = filedialog.askopenfilename(
            title="Select a query MIDI file",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")])
        if path:
            self._run_query(path)

    def _on_drop(self, event):
        raw = getattr(event, 'data', None)
        if not raw:
            return
        paths = self.root.tk.splitlist(raw)
        if paths:
            self._run_query(paths[0])

    def _on_search_again(self):
        if self.last_query_path is None:
            messagebox.showinfo("No query yet", "Drop or browse a query MIDI file first.")
            return
        # re-run with the SAME target (whole file or whichever segment was last
        # searched) — never re-triggers segmentation or resets the timeline
        label = None
        if self.active_segment_index is not None and self.active_segment_index < len(self.segments):
            s = self.segments[self.active_segment_index]
            label = f"segment {self.active_segment_index + 1} (measures {s['start_bar']+1}-{s['end_bar']})"
        self._run_query(self.last_query_path, label=label, is_new_file=False)

    def _run_query(self, query_path, label: Optional[str] = None, is_new_file: bool = True):
        if self.index is None:
            messagebox.showwarning("No index loaded", "Load an index first (Load Index... button).")
            return
        self.last_query_path = query_path
        display = label or os.path.basename(query_path)
        self.research_btn.config(state='normal')
        self._clear_results()
        self._set_status(f"Searching for grooves similar to '{display}'...", busy=True)
        # snapshot every live setting NOW (on the main thread) so the worker
        # thread never touches Tk variables directly (not thread-safe)
        settings = {
            'top_k': self.var_top_k.get(),
            'exclude_same_family': self.var_exclude_same_family.get(),
            'weight_rhythm': self.var_weight_rhythm.get(),
            'weight_velocity': self.var_weight_velocity.get(),
            'weight_density': self.var_weight_density.get(),
            'weight_tempo': self.var_weight_tempo.get(),
            'weight_hihat_pattern': self.var_weight_hihat_pattern.get(),
            'weight_tom_pattern': self.var_weight_tom_pattern.get(),
            'weight_cymbal_pattern': self.var_weight_cymbal_pattern.get(),
        }
        threading.Thread(target=self._query_worker, args=(query_path, settings, display), daemon=True).start()

        if is_new_file:
            self.current_full_query_path = query_path
            self.whole_file_btn.config(state='disabled')
            self._clear_segments()
            # cache the tempo at the start of the file — used to retime RESULT
            # audition to the query's tempo when no segment is selected.
            self.whole_file_tempo = None
            if HAS_PRETTY_MIDI:
                try:
                    self.whole_file_tempo = _tempo_at_time(pretty_midi.PrettyMIDI(query_path), 0.0)
                except Exception as exc:
                    _report_error(f"reading tempo from '{query_path}'", exc)
            if HAS_SEGMENTATION and self.seg_model is not None:
                self._start_segmentation(query_path)

    def _query_worker(self, query_path, settings, display):
        try:
            cfg = fsg.Config(**{k: v for k, v in self.index['cfg'].items()
                                if k in fsg.Config.__dataclass_fields__})
            # apply the LIVE query-time overrides (weights/top_k/exclude) on top
            # of the index-baked cfg — same override pattern query_similar() uses
            for field in ('weight_rhythm', 'weight_velocity', 'weight_density',
                         'weight_tempo', 'weight_hihat_pattern', 'weight_tom_pattern',
                         'weight_cymbal_pattern'):
                setattr(cfg, field, settings[field])

            query_fp = fsg.extract_fingerprint(query_path, cfg)
            if query_fp is None:
                raise ValueError("Could not extract a fingerprint — parse failure "
                                 "or too few drum notes in this file/section.")
            sims = fsg.compute_similarities(query_fp, self.index, cfg)
            order = np.argsort(-sims)
            query_abs = os.path.abspath(query_path)
            query_family = fsg.normalize_basename(query_path)
            results = []
            for i in order:
                p = self.index['paths'][i]
                if os.path.abspath(p) == query_abs:
                    continue
                if settings['exclude_same_family'] and self.index['meta'][i]['family'] == query_family:
                    continue
                results.append({'path': p, 'similarity': float(sims[i]),
                                'tempo': float(self.index['tempo'][i])})
                if len(results) >= settings['top_k']:
                    break
        except Exception as exc:
            msg = _report_error(f"searching for '{query_path}'", exc)
            self.root.after(0, lambda: self._on_query_error(msg))
            return
        self.root.after(0, lambda: self._on_query_done(display, results))

    def _on_query_done(self, display, results):
        self.results = results
        for i, r in enumerate(results, 1):
            self.tree.insert('', 'end', iid=str(i - 1),
                             values=(i, f"{r['similarity']:.3f}", os.path.basename(r['path'])))
        if results:
            self._set_status(f"{len(results)} matches for '{display}'.", busy=False)
        else:
            self._set_status(f"No matches found for '{display}'.", busy=False)

    def _on_query_error(self, msg):
        messagebox.showerror("Search failed", msg)
        self._set_status("Search failed.", busy=False)

    def _clear_results(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self.results = []
        self._stop_playback()
        self.audition_btn.config(state='disabled', text="▶ Audition")
        self.save_btn.config(state='disabled')
        self.segment_audition_btn.config(text="▶ Audition Segment")

    # ---- segmentation --------------------------------------------------------
    def _clear_segments(self):
        self.segments = []
        self.active_segment_index = None
        self.active_segment_temp_path = None
        self.segment_click_targets = {}
        self.segment_audition_btn.config(state='disabled', text="▶ Audition Segment")
        self.seg_canvas.delete('all')
        self.seg_canvas.config(height=0)
        self.seg_label.config(text="")

    def _start_segmentation(self, query_path):
        self.seg_label.config(text="Segmenting...", foreground='#0066cc')
        settings = {'threshold': self.var_seg_threshold.get(),
                   'context_overlap': self.var_seg_overlap.get()}
        threading.Thread(target=self._segment_worker, args=(query_path, settings), daemon=True).start()

    def _segment_worker(self, query_path, settings):
        try:
            result = dts.compute_segment_boundaries(
                self.seg_model, self.seg_cfg, query_path,
                threshold=settings['threshold'], context_overlap=settings['context_overlap'])
            starts = result['starts']
            total_measures = result['total_measures']
            midi = result['midi']
            # DESIGN: tempo-change-AWARE conversion (handles a query file with a
            # mid-file tempo change correctly) — NOT a naive constant sec_per_bar,
            # which would silently misplace every boundary after the change.
            segments = []
            for i, start_bar in enumerate(starts):
                end_bar = starts[i + 1] if i + 1 < len(starts) else total_measures
                if end_bar <= start_bar:
                    continue
                start_sec = dts.bar_to_seconds(start_bar, midi, self.seg_cfg)
                segments.append({
                    'start_bar': start_bar, 'end_bar': end_bar,
                    'start_sec': start_sec,
                    'end_sec': dts.bar_to_seconds(end_bar, midi, self.seg_cfg),
                    # tempo ACTUALLY active at this segment's start (not blindly the
                    # file's first tempo) — this is what result-groove audition gets
                    # rescaled to match, so it sounds right even after a mid-file
                    # tempo change in the query itself.
                    'tempo': _tempo_at_time(midi, start_sec),
                })
        except Exception as exc:
            msg = _report_error(f"segmenting '{query_path}'", exc)
            self.root.after(0, lambda: self._on_segmentation_error(msg))
            return
        self.root.after(0, lambda: self._on_segmentation_done(query_path, segments))

    def _on_segmentation_error(self, msg):
        self.seg_label.config(text="Segmentation failed (see console).", foreground='#b00000')
        print(f"[ERROR] {msg}")

    def _on_segmentation_done(self, query_path, segments):
        if query_path != self.current_full_query_path:
            return   # a newer file was dropped before this finished; discard
        self.segments = segments
        self.active_segment_index = None
        if not segments:
            self.seg_label.config(text="No segments detected.", foreground='gray')
            return
        self.seg_label.config(text=f"{len(segments)} segments detected "
                                    f"(click one to search just that section):",
                              foreground='gray')
        self._draw_segment_timeline()

    def _draw_segment_timeline(self):
        self.seg_canvas.delete('all')
        self.segment_click_targets = {}
        if not self.segments:
            self.seg_canvas.config(height=0)
            return
        self.seg_canvas.config(height=42)
        self.root.update_idletasks()
        canvas_w = max(200, self.seg_canvas.winfo_width())
        total_bars = sum(s['end_bar'] - s['start_bar'] for s in self.segments)
        if total_bars <= 0:
            return
        x = 0
        for i, seg in enumerate(self.segments):
            length = seg['end_bar'] - seg['start_bar']
            w = max(2, round(canvas_w * length / total_bars))
            color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
            rect = self.seg_canvas.create_rectangle(
                x, 2, x + w, 40, fill=color, outline='white', width=2, tags=(f'seg{i}',))
            label = f"{seg['start_bar']+1}-{seg['end_bar']}"
            if w > 28:
                self.seg_canvas.create_text(x + w / 2, 21, text=label, fill='white',
                                            font=('', 8), tags=(f'seg{i}',))
            self.seg_canvas.tag_bind(f'seg{i}', '<Button-1>',
                                     lambda e, s=seg, idx=i: self._on_segment_clicked(s, idx))
            self.segment_click_targets[i] = {'rect': rect, 'x0': x, 'x1': x + w}
            x += w

    def _on_segment_clicked(self, segment, idx):
        if self.current_full_query_path is None:
            return
        self.active_segment_index = idx
        # highlight: thicker white-ish outline on the clicked segment, reset others
        for i, info in self.segment_click_targets.items():
            self.seg_canvas.itemconfig(info['rect'],
                                       width=4 if i == idx else 2,
                                       outline='#222' if i == idx else 'white')
        self.whole_file_btn.config(state='normal')
        try:
            temp_path = self._slice_segment_to_temp(
                self.current_full_query_path, segment['start_sec'], segment['end_sec'])
        except Exception as exc:
            msg = _report_error(f"slicing segment {idx+1} from "
                                f"'{self.current_full_query_path}'", exc)
            messagebox.showerror("Could not slice segment", msg)
            return
        self.active_segment_temp_path = temp_path
        self.segment_audition_btn.config(state='normal')
        label = f"segment {idx + 1} (measures {segment['start_bar']+1}-{segment['end_bar']})"
        self._run_query(temp_path, label=label, is_new_file=False)

    def _on_whole_file_clicked(self):
        if self.current_full_query_path is None:
            return
        self.active_segment_index = None
        self.active_segment_temp_path = None
        self.segment_audition_btn.config(state='disabled')
        for info in self.segment_click_targets.values():
            self.seg_canvas.itemconfig(info['rect'], width=2, outline='white')
        self.whole_file_btn.config(state='disabled')
        self._run_query(self.current_full_query_path, is_new_file=False)

    def _slice_segment_to_temp(self, source_path, start_sec, end_sec):
        """Write a NEW small MIDI file containing only the notes that fall inside
        [start_sec, end_sec) of source_path, with times shifted to start at 0, so
        it can be fingerprinted and compared exactly like any other groove file."""
        if not HAS_PRETTY_MIDI:
            raise RuntimeError("pretty_midi is required to slice segments.")
        src = pretty_midi.PrettyMIDI(source_path)
        # DESIGN: use the tempo actually ACTIVE at this segment's start time, not
        # blindly the file's first tempo — a file that changes tempo partway
        # through would otherwise tag every later segment with the wrong tempo,
        # misleading the --weight_tempo similarity comparison for that search.
        tempo = _tempo_at_time(src, start_sec)
        out = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        inst = pretty_midi.Instrument(program=0, is_drum=True)
        for orig in src.instruments:
            if not orig.is_drum:
                continue
            for n in orig.notes:
                if start_sec <= n.start < end_sec:
                    inst.notes.append(pretty_midi.Note(
                        velocity=n.velocity, pitch=n.pitch,
                        start=n.start - start_sec,
                        end=max(n.start - start_sec + 0.01, n.end - start_sec)))
        if not inst.notes:
            raise ValueError("This segment contains no drum notes to search with.")
        out.instruments.append(inst)
        path = os.path.join(self.temp_dir, f"segment_{uuid.uuid4().hex[:8]}.mid")
        out.write(path)
        return path

    # ---- selection / audition ----------------------------------------------
    def _on_row_select(self, event=None):
        sel = self.tree.selection()
        if not sel:
            self.audition_btn.config(state='disabled')
            self.save_btn.config(state='disabled')
            return
        self.audition_btn.config(state='normal')
        self.save_btn.config(state='normal')
        idx = int(sel[0])
        selected_path = self.results[idx]['path']
        # If a RESULT is already playing and the user picked a DIFFERENT row,
        # seamlessly switch playback to the new one (per spec, no extra click).
        # A currently-playing SEGMENT audition is left alone — only another
        # result-row selection or the segment button itself should affect it.
        if (self.playing_source == 'result' and self.playing_path is not None
                and selected_path != self.playing_path):
            self._start_playback(selected_path, 'result')

    def _on_audition_clicked(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        path = self.results[idx]['path']
        if self.playing_source == 'result' and self.playing_path is not None:
            self._stop_playback()
        else:
            self._start_playback(path, 'result')

    def _on_save_clicked(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        result_path = self.results[idx]['path']

        # DESIGN: filename = {query}-{segment_no}-{similar_groove}.mid — the query
        # name always comes from the ORIGINAL dropped file (current_full_query_path),
        # never a segment's generated temp-file name, since that's what "$query_
        # filename$" means to a user. segment_no is 1-indexed to match how segments
        # are already labelled elsewhere in this app ("segment 2 (measures ...)");
        # a whole-file search (no active segment) uses "whole" instead of a number.
        query_source = self.current_full_query_path or self.last_query_path
        query_base = os.path.splitext(os.path.basename(query_source))[0] if query_source else "query"
        if self.active_segment_index is not None:
            segment_label = str(self.active_segment_index + 1)
        else:
            segment_label = "whole"
        result_base = os.path.splitext(os.path.basename(result_path))[0]
        suggested_name = f"{query_base}-{segment_label}-{result_base}.mid"

        dest = filedialog.asksaveasfilename(
            title="Save similar groove as", initialfile=suggested_name,
            defaultextension=".mid", filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")])
        if not dest:
            return
        try:
            shutil.copy2(result_path, dest)
        except Exception as exc:
            msg = _report_error(f"saving '{result_path}' to '{dest}'", exc)
            messagebox.showerror("Save failed", msg)
            return
        self._set_status(f"Saved → {os.path.basename(dest)}", busy=False)

    def _on_segment_audition_clicked(self):
        if self.active_segment_temp_path is None:
            return
        if self.playing_source == 'segment' and self.playing_path is not None:
            self._stop_playback()
        else:
            self._start_playback(self.active_segment_temp_path, 'segment')

    def _current_query_tempo(self) -> Optional[float]:
        """Tempo (bpm) that RESULT-groove audition should be retimed to match —
        the active segment's own tempo if one is selected, else the whole query
        file's tempo at its start. None if unknown (no query loaded yet)."""
        if self.active_segment_index is not None and self.active_segment_index < len(self.segments):
            t = self.segments[self.active_segment_index].get('tempo')
            if t:
                return t
        return self.whole_file_tempo

    def _start_playback(self, path, source):
        # Both audition buttons share ONE MidiPlayer (only one output port, only
        # one thing can really be sounding at a time) — starting either kind of
        # playback stops whatever the other one was doing and resets ITS button.
        self.playing_path = path
        self.playing_source = source
        if source == 'result':
            self.audition_btn.config(text="■ Stop")
            self.segment_audition_btn.config(text="▶ Audition Segment")
        else:
            self.segment_audition_btn.config(text="■ Stop")
            self.audition_btn.config(text="▶ Audition")

        # RETEMPO a result groove to the query's (segment's or whole-file's) tempo
        # — a segment/whole-file audition is already at the query's own tempo, so
        # it never needs rescaling. A missing tempo on either side (e.g. tempo
        # couldn't be read) safely falls back to 1.0 = play at native tempo.
        tempo_scale = 1.0
        status_suffix = ""
        if source == 'result':
            target_tempo = self._current_query_tempo()
            result_tempo = next((r.get('tempo') for r in self.results if r['path'] == path), None)
            if target_tempo and result_tempo and result_tempo > 0:
                tempo_scale = target_tempo / result_tempo
                status_suffix = (f" at {target_tempo:.0f} bpm "
                                 f"(rescaled from its native {result_tempo:.0f} bpm)")
        self._set_status(f"Playing '{os.path.basename(path)}'{status_suffix}...", busy=False)
        threading.Thread(target=self._start_playback_worker,
                         args=(path, source, tempo_scale), daemon=True).start()

    def _start_playback_worker(self, path, source, tempo_scale=1.0):
        try:
            self.player.play(path, tempo_scale=tempo_scale, on_finished=lambda err: self.root.after(
                0, lambda: self._on_playback_finished(path, source, err)))
        except Exception as exc:
            msg = _report_error(f"starting playback of '{path}'", exc)
            self.root.after(0, lambda: self._on_playback_finished(path, source, msg))

    def _stop_playback(self):
        self.player.stop()   # non-blocking request; UI resets via _on_playback_finished

    def _on_playback_finished(self, path, source, error):
        # Guard against a stale callback from a track we've already switched
        # AWAY from (its on_finished still fires when interrupted).
        if self.playing_path != path or self.playing_source != source:
            return
        self.playing_path = None
        self.playing_source = None
        self.audition_btn.config(text="▶ Audition")
        self.segment_audition_btn.config(text="▶ Audition Segment")
        if error:
            self._set_status(f"Playback error: {error}", busy=False)
        else:
            self._set_status("Ready.", busy=False)

    # ---- misc ----------------------------------------------------------------
    def _set_status(self, text, busy=False):
        self.status_label.config(text=text, foreground=('#0066cc' if busy else 'gray'))

    def _on_close(self):
        try:
            self.player.close()
        except Exception:
            pass
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
        self.root.destroy()


def main():
    if not HAS_MIDO:
        print("mido is required for audition playback: pip install mido python-rtmidi")
    root = None
    if HAS_DND:
        try:
            root = TkinterDnD.Tk()
        except Exception as exc:
            _report_error("initializing drag-and-drop root window (falling back "
                          "to a plain window; click-to-browse still works)", exc)
            root = None
    if root is None:
        root = tk.Tk()
        if not HAS_DND:
            print("(tkinterdnd2 not installed — drag-and-drop disabled, "
                  "click-to-browse still works: pip install tkinterdnd2)")
    GrooveFinderApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
