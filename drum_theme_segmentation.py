#!/usr/bin/env python3
"""
==============================================================================
 DRUM MIDI THEME TEMPORAL SEGMENTATION  (single-file: dataset + train + infer)
==============================================================================

WHAT THIS PROGRAM DOES
-----------------------
Given a long drum MIDI file (e.g. a full song assembled from several distinct
grooves/sections), predict WHERE the theme/section boundaries are - i.e. which
measures start a genuinely new pattern versus continue/repeat the current one.

There is no hand-labeled boundary data. Instead, this tool SYNTHESIZES its own
ground truth: it takes short single-theme MIDI files (loops) from your library,
concatenates many of them together in long, randomized sequences with random
repeats, and - because *it* did the concatenating - knows exactly where every
true boundary is. That is the (input, label) training pair. A model then
learns to detect the same kind of boundary in a real, unlabeled file.

KEY RULES (as specified)
-------------------------
  • Eligible "theme" files: filename does NOT contain "song" (case-insensitive)
    AND the file is at most 10 measures long AND its very first note lands
    exactly on beat 1 of bar 1 (a clean downbeat start - see DESIGN note below).
  • Repeating the SAME theme back-to-back is NOT a boundary - only a genuine
    change of theme is labeled as one.
  • Filenames that look like variations of each other (e.g. "Blues Fast Fill 1"
    vs "Blues Fast Fill 14") are treated as the same "family" and never used as
    two different themes in one training sample - only one member of a family
    is picked per sample, so near-duplicate takes never masquerade as a real,
    learnable boundary.
  • Each training sample concatenates 30–40 distinct themes (configurable), each
    repeated a random 1–10 times, with the constraint that at least one theme in
    the sample is NOT repeated (count=1) and at least one IS repeated (count>1).
  • The model predicts a per-NOTE boundary probability, but a segment can only
    be judged to START on a note that sits at the very start of a measure
    (grid_step == 0). infer_segments() enforces this explicitly.

DESIGN NOTE - why "clean downbeat start" is required for a theme to be eligible
---------------------------------------------------------------------------------
Because only grid-step-0 notes are ever allowed to carry a boundary label, a
theme whose first note is NOT on the downbeat (a pickup/anacrusis) would have
no note to attach its true boundary to - an unlabelable positive example. Such
files are excluded from theme selection so every boundary the model is trained
on has a real note carrying its label. This is a real, stated limitation: a
song built with off-downbeat pickup sections in your own inference data may be
harder for the model to catch, precisely because it was never shown one.

HOW IT IS USED - THREE MODES
------------------------------
  1) dataset : scan a MIDI library, filter eligible themes, synthesize the
               concatenated+repeated training sequences, cache them.
  2) train   : train the segmentation transformer on that cache.
  3) infer   : run a trained model on a real MIDI file and report predicted
               theme-boundary measures (optionally writing MIDI markers).

  pip install torch pretty_midi numpy tqdm

  python drum_theme_segmentation.py --mode dataset --data_dir "/path/to/MIDI" \
         --cache cache/segments.pkl --num_samples 300

  python drum_theme_segmentation.py --mode train --cache cache/segments.pkl \
         --run_name seg_v1 --epochs 40

  python drum_theme_segmentation.py --mode infer --checkpoint checkpoints/seg_v1/best.pt \
         --input my_song.mid --threshold 0.5

Search for "DESIGN:" to find inline rationale for specific decisions.
"""

import os
import sys
import re
import glob
import json
import time
import pickle
import random
import difflib
import argparse
import traceback
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False
    print("Warning: pretty_midi not installed - MIDI I/O disabled. "
          "Install with: pip install pretty_midi")

try:
    import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# REPRODUCIBILITY - single seed for ALL random number generators
# =============================================================================
GLOBAL_SEED = 42

def seed_everything(seed: int = GLOBAL_SEED, verbose: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if verbose:
        print(f"[seed] All RNGs seeded with {seed} (random, numpy, torch/cuda).")


def _worker_init_fn(worker_id: int):
    worker_seed = (GLOBAL_SEED + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


seed_everything(GLOBAL_SEED, verbose=False)   # seed at IMPORT time too (library use)


# =============================================================================
# ERROR REPORTING HELPERS  (same pattern as the humanizer project)
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


def _report_error(context: str, exc: BaseException, fatal: bool = False):
    loc = _error_location(exc)
    print(f"\n[ERROR] {context}")
    print(f"        -> {type(exc).__name__} at {loc}: {exc}")
    if fatal:
        print("        Full traceback:")
        traceback.print_exc()


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # ── Musical grid (simplifying assumption: fixed 4/4, unlike the humanizer) ──
    ticks_per_beat:     int = 480
    grid_resolution:    int = 4       # subdivisions per beat -> 16th-note grid
    beats_per_bar:      int = 4
    max_bars:           int = 4096    # long concatenated sequences need headroom

    # ── Theme eligibility filters ────────────────────────────────────────────
    exclude_substring:   str   = "song"   # case-insensitive filename exclusion
    max_theme_measures:  int   = 10       # themes longer than this are excluded
    min_notes_per_theme: int   = 4        # too-sparse files are excluded
    # DESIGN: the PRIMARY, reliable de-duplication is exact-match after stripping a
    # trailing take-number ("Blues Fast Fill 1"/"...14" both -> "blues fast fill" -
    # handles the user's stated example with zero ambiguity). This ratio is only a
    # SECONDARY, coarser safety net for near-duplicate spellings the strip misses
    # ("funk verse"/"funk verses"). Generic string similarity has no threshold that
    # is right for every naming convention - e.g. "Groove00"/"Groove01" (genuinely
    # different themes) can score as similarly "close" as true near-duplicates. Kept
    # conservative (high) by default so it rarely merges themes that are actually
    # different; lower it via --family_similarity if your library has closer misses.
    family_similarity:   float = 0.95     # difflib ratio ≥ this ⇒ treat as same family

    # ── Synthetic dataset construction ───────────────────────────────────────
    min_themes_per_sample: int = 30
    max_themes_per_sample: int = 40
    min_repeat:             int = 1
    max_repeat:              int = 10

    # ── Model input vocabulary ────────────────────────────────────────────────
    num_instruments:   int = 13
    velocity_bins:      int = 32
    max_ioi_steps:      int = 64

    # ── Model architecture ────────────────────────────────────────────────────
    d_model:          int = 128
    nhead:            int = 4
    num_layers:       int = 3
    dim_feedforward:  int = 256
    dropout:          float = 0.1
    max_seq_len:      int = 512

    # ── Training ───────────────────────────────────────────────────────────────
    batch_size:       int = 16
    lr:               float = 3e-4
    weight_decay:     float = 0.01
    max_epochs:       int = 40
    warmup_pct:       float = 0.1
    grad_clip:        float = 1.0
    val_split:        float = 0.1
    early_stop_patience: int = 8
    num_workers:      int = 2
    pos_weight:       float = 8.0    # class-imbalance weight for the rare positive class

    def __post_init__(self):
        self.ticks_per_grid     = self.ticks_per_beat // self.grid_resolution
        self.grid_steps_per_bar = self.grid_resolution * self.beats_per_bar
        self.max_position       = self.max_bars * self.grid_steps_per_bar + 8


# =============================================================================
# MINIMAL GM DRUM MAP  (self-contained - no dependency on other project files)
# =============================================================================
GM_DRUM_MAP = {
    35: 0, 36: 0,                          # kick
    38: 1, 40: 1, 37: 1, 39: 1,            # snare / e-snare / side stick / clap
    42: 2, 44: 2,                          # closed hi-hat / pedal hi-hat
    46: 3,                                 # open hi-hat
    41: 4, 43: 4,                          # low tom
    45: 5, 47: 5,                          # mid tom
    48: 6, 50: 6,                          # high tom
    49: 7, 57: 7,                          # crash
    51: 8, 59: 8,                          # ride
    53: 9,                                 # ride bell
    52: 10, 55: 10,                        # china / splash
    56: 11, 58: 11, 60: 11, 62: 11,        # perc
}
INSTRUMENT_TO_GM = {0: 36, 1: 38, 2: 42, 3: 46, 4: 41, 5: 45, 6: 48,
                    7: 49, 8: 51, 9: 53, 10: 52, 11: 56, 12: 38}
DRUM_CLASS_NAMES = {
    0: "Kick", 1: "Snare", 2: "HH-Closed", 3: "HH-Open", 4: "LowTom",
    5: "MidTom", 6: "HighTom", 7: "Crash", 8: "Ride", 9: "RideBell",
    10: "China/Splash", 11: "Perc", 12: "Other",
}


# =============================================================================
# NOTE EVENT + MIDI LOADING
# =============================================================================

@dataclass
class NoteEvent:
    """One drum hit, quantized to a fixed 16th-note / 4-4 grid."""
    instrument: int
    bar:        int
    grid_step:  int      # 0..grid_steps_per_bar-1 (position within the bar)
    velocity:   int       # raw MIDI velocity 0-127
    raw_tick:   int
    raw_pitch:  int
    boundary:   int = 0   # TRAINING LABEL ONLY: 1 = this note starts a new theme


def _load_midi_notes_impl(path: str, cfg: Config) -> Optional[List[NoteEvent]]:
    midi = pretty_midi.PrettyMIDI(path)
    notes = [n for t in midi.instruments if t.is_drum for n in t.notes]
    if not notes:
        return None
    notes.sort(key=lambda n: n.start)
    bar_ticks = cfg.ticks_per_beat * cfg.beats_per_bar
    events: List[NoteEvent] = []
    for n in notes:
        inst = GM_DRUM_MAP.get(n.pitch, -1)
        if inst < 0:
            continue
        # DESIGN: tempo-change-AWARE conversion via pretty_midi's own time_to_tick()
        # - see the matching fix (and its verification) in find_similar_grooves.py's
        # _extract_fingerprint_impl for why a naive `time * single_tempo` formula
        # silently misplaces every note after a mid-file tempo change.
        beats_elapsed = midi.time_to_tick(n.start) / midi.resolution
        tick = int(round(beats_elapsed * cfg.ticks_per_beat))
        grid_tick = round(tick / cfg.ticks_per_grid) * cfg.ticks_per_grid
        bar = grid_tick // bar_ticks
        step = (grid_tick % bar_ticks) // cfg.ticks_per_grid
        events.append(NoteEvent(instrument=inst, bar=int(bar), grid_step=int(step),
                                velocity=int(n.velocity), raw_tick=tick, raw_pitch=int(n.pitch)))
    return events or None


def load_midi_notes(path: str, cfg: Config) -> Optional[List[NoteEvent]]:
    """Load a MIDI file's drum notes onto the fixed grid. Returns None (and
    reports why) on any parse failure - callers treat None as 'skip this file'."""
    if not HAS_PRETTY_MIDI:
        raise ImportError("pretty_midi required for MIDI I/O")
    try:
        return _load_midi_notes_impl(path, cfg)
    except Exception as exc:
        _report_error(f"parsing MIDI file '{os.path.basename(path)}'", exc)
        return None


def measure_count(events: List[NoteEvent]) -> int:
    if not events:
        return 0
    return max(e.bar for e in events) + 1


# =============================================================================
# FILENAME "FAMILY" GROUPING
# =============================================================================
# DESIGN: files like "Blues Fast Fill 1.mid" and "Blues Fast Fill 14.mid" are
# almost always different TAKES of the same underlying theme, not two genuinely
# different themes. If we concatenated both into one training sample as if they
# were distinct, the "boundary" between them would be musically near-undetectable
# (label noise) - and worse, it would teach the model that near-identical audio
# can still be a boundary, which is exactly wrong. So: normalize each filename
# to a "family" key by stripping numbers/extensions/separators, and NEVER pick
# two files from the same (or a very similar) family as separate themes in one
# training sample.

def normalize_basename(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0].lower()
    name = re.sub(r'[_\-]+', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = re.sub(r'\s*\d+\s*$', '', name).strip()   # strip a trailing take/number
    return name


def _too_similar(a: str, b: str, ratio: float) -> bool:
    if a == b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= ratio


def scan_theme_files(data_dir: str, cfg: Config) -> Tuple[Dict[str, List[Tuple[str, List[NoteEvent]]]], Dict]:
    """
    Scan data_dir for eligible "theme" MIDI files and group them into families.
    A file is eligible only if ALL of:
      • its filename does not contain cfg.exclude_substring (case-insensitive)
      • it parses successfully and has ≥ cfg.min_notes_per_theme notes
      • its length is ≤ cfg.max_theme_measures measures
      • its very first note lands exactly on bar 0 / grid_step 0 (a clean
        downbeat start - see the module DESIGN note for why this is required)
    Returns (families, stats) where families maps a normalized name -> list of
    (path, events) tuples, and stats reports how many files were excluded and why.
    """
    paths = []
    for ext in ('mid', 'midi', 'MID', 'MIDI'):
        paths.extend(glob.glob(os.path.join(data_dir, '**', f'*.{ext}'), recursive=True))
    paths = sorted(set(paths))

    stats = {'found': len(paths), 'excluded_name': 0, 'excluded_parse': 0,
             'excluded_too_few_notes': 0, 'excluded_too_long': 0,
             'excluded_no_downbeat': 0, 'kept': 0}
    families: Dict[str, List[Tuple[str, List[NoteEvent]]]] = {}
    excl = cfg.exclude_substring.lower()

    for p in paths:
        base = os.path.basename(p)
        if excl and excl in base.lower():
            stats['excluded_name'] += 1
            continue
        events = load_midi_notes(p, cfg)
        if events is None:
            stats['excluded_parse'] += 1
            continue
        if len(events) < cfg.min_notes_per_theme:
            stats['excluded_too_few_notes'] += 1
            continue
        events.sort(key=lambda e: (e.bar, e.grid_step, e.instrument))
        nmeasures = measure_count(events)
        if nmeasures > cfg.max_theme_measures:
            stats['excluded_too_long'] += 1
            continue
        if events[0].bar != 0 or events[0].grid_step != 0:
            stats['excluded_no_downbeat'] += 1
            continue
        stats['kept'] += 1
        fam = normalize_basename(p)
        families.setdefault(fam, []).append((p, events))

    print(f"── Theme scan ─────────────────────────────────────────")
    print(f"  Files found:                  {stats['found']}")
    print(f"  Excluded (filename match):    {stats['excluded_name']} "
          f"(contains '{cfg.exclude_substring}')")
    print(f"  Excluded (parse failed):      {stats['excluded_parse']}")
    print(f"  Excluded (too few notes):     {stats['excluded_too_few_notes']}")
    print(f"  Excluded (> {cfg.max_theme_measures} measures):        {stats['excluded_too_long']}")
    print(f"  Excluded (no downbeat start): {stats['excluded_no_downbeat']}")
    print(f"  Eligible theme files:         {stats['kept']}  "
          f"(grouped into {len(families)} name-families)")
    print(f"──────────────────────────────────────────────────────\n")
    return families, stats


def events_to_arrays(events: List[NoteEvent], cfg: Config) -> Dict[str, np.ndarray]:
    n = len(events)
    spb = cfg.grid_steps_per_bar
    out = {
        'instruments':      np.zeros(n, dtype=np.int32),
        'velocities':       np.zeros(n, dtype=np.int32),
        'grid_steps':       np.zeros(n, dtype=np.int32),
        'positions':        np.zeros(n, dtype=np.int32),
        'iois':             np.zeros(n, dtype=np.int32),
        'is_measure_start': np.zeros(n, dtype=np.int32),
        'boundary':         np.zeros(n, dtype=np.float32),
    }
    prev_pos = None
    vscale = 128.0 / cfg.velocity_bins
    for i, e in enumerate(events):
        out['instruments'][i] = e.instrument
        out['velocities'][i]  = min(cfg.velocity_bins - 1, int(e.velocity / vscale))
        out['grid_steps'][i]  = e.grid_step
        gpos = e.bar * spb + e.grid_step
        out['positions'][i]   = gpos
        out['iois'][i]        = 0 if prev_pos is None else int(np.clip(gpos - prev_pos, 0, cfg.max_ioi_steps))
        prev_pos = gpos
        out['is_measure_start'][i] = 1 if e.grid_step == 0 else 0
        out['boundary'][i]    = float(e.boundary)
    return out


# =============================================================================
# SYNTHETIC DATASET CONSTRUCTION
# =============================================================================
# DESIGN: this is the whole trick that makes the task learnable without any real
# labeled data. We pick K distinct, dissimilarly-named themes, shuffle their
# order, assign each a random repeat count, and concatenate them back-to-back
# with bar indices re-based so everything stays grid-aligned. Because WE did the
# concatenating, we know exactly which note is the true first note of each new
# theme's FIRST occurrence in its block - that note (and only that note) gets
# boundary=1. Repeats of the same theme within its own block get boundary=0.

def _pick_family_keys(families: Dict, k: int, rng: random.Random, similarity: float) -> List[str]:
    keys = list(families.keys())
    rng.shuffle(keys)
    chosen: List[str] = []
    for key in keys:
        if len(chosen) >= k:
            break
        if any(_too_similar(key, c, similarity) for c in chosen):
            continue
        chosen.append(key)
    return chosen


def build_one_sample(families: Dict, cfg: Config, rng: random.Random) -> Optional[List[NoteEvent]]:
    k = rng.randint(cfg.min_themes_per_sample, cfg.max_themes_per_sample)
    fam_keys = _pick_family_keys(families, k, rng, cfg.family_similarity)
    if len(fam_keys) < 2:
        return None   # not enough distinct, dissimilar themes available

    chosen = [rng.choice(families[fk]) for fk in fam_keys]   # [(path, events), ...]
    rng.shuffle(chosen)

    repeats = [rng.randint(cfg.min_repeat, cfg.max_repeat) for _ in chosen]
    # ENFORCE: at least one theme NOT repeated (count==1) and at least one repeated (count>1)
    if all(r == 1 for r in repeats):
        repeats[rng.randrange(len(repeats))] = rng.randint(2, cfg.max_repeat)
    if all(r > 1 for r in repeats):
        repeats[rng.randrange(len(repeats))] = 1

    out_events: List[NoteEvent] = []
    bar_cursor = 0
    for (path, events), r in zip(chosen, repeats):
        n_measures = measure_count(events)
        if n_measures <= 0:
            continue
        for rep_i in range(r):
            for e in events:
                is_first_note_of_block = (rep_i == 0 and e.bar == 0 and e.grid_step == 0)
                out_events.append(NoteEvent(
                    instrument=e.instrument, bar=e.bar + bar_cursor, grid_step=e.grid_step,
                    velocity=e.velocity, raw_tick=e.raw_tick, raw_pitch=e.raw_pitch,
                    boundary=1 if is_first_note_of_block else 0,
                ))
            bar_cursor += n_measures
        if bar_cursor >= cfg.max_bars:
            break   # safety: don't overflow the position-embedding range
    out_events.sort(key=lambda e: (e.bar, e.grid_step, e.instrument))
    return out_events or None


def build_training_samples(families: Dict, cfg: Config, num_samples: int,
                           seed: int = GLOBAL_SEED, verbose_examples: int = 2) -> List[Dict]:
    rng = random.Random(seed)
    samples = []
    shown = 0
    for i in range(num_samples):
        events = build_one_sample(families, cfg, rng)
        if events is None:
            continue
        n_boundary_bars = len(set(e.bar for e in events if e.boundary == 1))
        n_bars = measure_count(events)
        arr = events_to_arrays(events, cfg)
        samples.append({'arrays': arr, 'length': len(events),
                        'n_bars': n_bars, 'n_boundaries': n_boundary_bars})
        if shown < verbose_examples:
            print(f"  sample {i}: {len(events)} notes, {n_bars} bars, "
                  f"{n_boundary_bars} true theme boundaries")
            shown += 1
    print(f"\nBuilt {len(samples)}/{num_samples} training samples "
          f"(some may be skipped if too few dissimilar theme families exist).")
    return samples


def build_cache(data_dir: str, cache_path: str, cfg: Config, num_samples: int):
    families, stats = scan_theme_files(data_dir, cfg)
    if len(families) < 2:
        print(f"ERROR: only {len(families)} usable theme family(ies) found in "
              f"'{data_dir}'. Need at least 2 distinct, dissimilarly-named themes.")
        sys.exit(1)
    samples = build_training_samples(families, cfg, num_samples)
    if not samples:
        print("ERROR: no training samples could be built (library too small/uniform).")
        sys.exit(1)
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'cfg': asdict(cfg), 'samples': samples,
                    'scan_stats': stats, 'num_families': len(families)}, f, protocol=4)
    print(f"Cache saved -> {cache_path}")
    return samples


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class SegDataset(Dataset):
    """
    Windows each long synthetic sample down to cfg.max_seq_len notes per training
    step (a random crop each call - different context each epoch). Labels are
    per-note and window-crop-invariant (a note's boundary status doesn't depend on
    where the window starts), so cropping never corrupts the target.
    """
    def __init__(self, samples: List[Dict], cfg: Config):
        self.samples = samples
        self.cfg = cfg

    def __len__(self):
        return len(self.samples)

    def _window(self, arr, start, length):
        return {k: v[start:start + length] for k, v in arr.items()}

    def _pad(self, arr, target_len):
        n = len(arr['instruments'])
        pad = target_len - n
        if pad <= 0:
            return arr
        out = {}
        for k, v in arr.items():
            fill = -1 if k == 'instruments' else 0
            out[k] = np.concatenate([v, np.full(pad, fill, dtype=v.dtype)])
        return out

    def __getitem__(self, idx):
        arr = self.samples[idx]['arrays']
        L = self.cfg.max_seq_len
        n = len(arr['instruments'])
        if n > L:
            start = random.randint(0, n - L)
            arr = self._window(arr, start, L)
        arr = self._pad(arr, L)

        pad_mask = torch.tensor(arr['instruments'] < 0, dtype=torch.bool)
        ni = self.cfg.num_instruments
        clip = lambda a: np.clip(a, 0, ni - 1)
        return {
            'instruments':      torch.tensor(clip(arr['instruments']), dtype=torch.long),
            'velocities':       torch.tensor(np.clip(arr['velocities'], 0, self.cfg.velocity_bins - 1), dtype=torch.long),
            'grid_steps':       torch.tensor(np.clip(arr['grid_steps'], 0, self.cfg.grid_steps_per_bar - 1), dtype=torch.long),
            'positions':        torch.tensor(np.clip(arr['positions'], 0, self.cfg.max_position - 1), dtype=torch.long),
            'iois':             torch.tensor(np.clip(arr['iois'], 0, self.cfg.max_ioi_steps), dtype=torch.long),
            'is_measure_start': torch.tensor(arr['is_measure_start'], dtype=torch.long),
            'tgt_boundary':     torch.tensor(arr['boundary'], dtype=torch.float32),
            'pad_mask':         pad_mask,
        }


def make_loaders(samples: List[Dict], cfg: Config):
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * cfg.val_split))
    val, train = samples[:n_val], samples[n_val:]
    print(f"Train samples: {len(train)}  Val samples: {len(val)}")
    g = torch.Generator(); g.manual_seed(GLOBAL_SEED)
    tl = DataLoader(SegDataset(train, cfg), batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                    persistent_workers=(cfg.num_workers > 0),
                    worker_init_fn=_worker_init_fn if cfg.num_workers > 0 else None,
                    generator=g)
    vl = DataLoader(SegDataset(val, cfg), batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True,
                    persistent_workers=(cfg.num_workers > 0),
                    worker_init_fn=_worker_init_fn if cfg.num_workers > 0 else None)
    return tl, vl


# =============================================================================
# MODEL
# =============================================================================
# DESIGN: kept deliberately SIMPLER than the humanizer's feature set, per request -
# no per-instrument IOI, no flam detection (irrelevant to segmentation). Inputs are
# just: instrument, velocity, position-in-measure, a simple global IOI, an explicit
# "is this a measure-start note" flag, and absolute sequence position.

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, positions):
        return self.pe[positions.clamp(max=self.pe.size(0) - 1)]


class NoteEmbedding(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.instrument_emb      = nn.Embedding(cfg.num_instruments + 1, d, padding_idx=0)
        self.velocity_emb        = nn.Embedding(cfg.velocity_bins, d)
        self.grid_step_emb       = nn.Embedding(cfg.grid_steps_per_bar, d)
        self.ioi_emb              = nn.Embedding(cfg.max_ioi_steps + 1, d)
        self.measure_start_emb   = nn.Embedding(2, d)
        self.pos_emb              = SinusoidalPositionEmbedding(d, cfg.max_position + 1)
        self.proj                 = nn.Linear(d * 5, d)
        self.norm                 = nn.LayerNorm(d)

    def forward(self, instruments, velocities, grid_steps, iois, is_measure_start, positions):
        ie  = self.instrument_emb(instruments + 1)
        ve  = self.velocity_emb(velocities)
        ge  = self.grid_step_emb(grid_steps)
        oe  = self.ioi_emb(iois.clamp(max=self.ioi_emb.num_embeddings - 1))
        mse = self.measure_start_emb(is_measure_start.clamp(0, 1))
        pe  = self.pos_emb(positions)
        x = self.proj(torch.cat([ie, ve, ge, oe, mse], dim=-1))
        return self.norm(x + pe)


class SegmentationTransformer(nn.Module):
    """Encoder-only transformer: per-note boundary-probability tagging, one
    forward pass (same "labelling not generation" logic as the humanizer)."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = NoteEmbedding(cfg)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.num_layers)
        self.head = nn.Linear(cfg.d_model, 1)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch):
        x = self.embed(batch['instruments'], batch['velocities'], batch['grid_steps'],
                       batch['iois'], batch['is_measure_start'], batch['positions'])
        x = self.encoder(x, src_key_padding_mask=batch['pad_mask'])
        return self.head(x).squeeze(-1)          # (B,T) raw logits

    @torch.no_grad()
    def infer_probs(self, batch):
        self.eval()
        return torch.sigmoid(self.forward(batch))


# =============================================================================
# LOSS
# =============================================================================
# DESIGN: supervise ONLY measure-start notes (grid_step==0). Non-measure-start
# notes can NEVER be a true boundary by construction, so including them in the
# loss would just be trivial always-0 supervision - wasted signal that also makes
# the already-severe class imbalance worse. The model still SEES every note (full
# context matters for deciding whether the next downbeat is a boundary); we only
# score it at the positions the task actually cares about.

def compute_loss(logits: torch.Tensor, batch: Dict, cfg: Config):
    valid = (~batch['pad_mask']) & (batch['is_measure_start'] == 1)
    if valid.sum() == 0:
        z = logits.sum() * 0.0
        return z, {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'n_pos': 0, 'n_valid': 0}
    lv = logits[valid]
    tv = batch['tgt_boundary'][valid]
    pos_weight = torch.tensor(cfg.pos_weight, device=logits.device, dtype=torch.float32)
    loss = F.binary_cross_entropy_with_logits(lv, tv, pos_weight=pos_weight)
    with torch.no_grad():
        pred = (torch.sigmoid(lv) >= 0.5).float()
        tp = ((pred == 1) & (tv == 1)).sum().item()
        fp = ((pred == 1) & (tv == 0)).sum().item()
        fn = ((pred == 0) & (tv == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return loss, {'precision': precision, 'recall': recall, 'f1': f1,
                  'n_pos': int(tv.sum().item()), 'n_valid': int(valid.sum().item())}


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    tot_loss = 0.0
    tp = fp = fn = 0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        loss, parts = compute_loss(logits, batch, cfg)
        tot_loss += loss.item(); n += 1
        valid = (~batch['pad_mask']) & (batch['is_measure_start'] == 1)
        if valid.sum() == 0:
            continue
        pred = (torch.sigmoid(logits[valid]) >= 0.5).float()
        tv = batch['tgt_boundary'][valid]
        tp += ((pred == 1) & (tv == 1)).sum().item()
        fp += ((pred == 1) & (tv == 0)).sum().item()
        fn += ((pred == 0) & (tv == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'loss': tot_loss / max(n, 1), 'precision': precision, 'recall': recall, 'f1': f1}


# =============================================================================
# TRAINING
# =============================================================================

def train(cfg: Config, samples: List[Dict], run_name: str, resume: Optional[str] = None):
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    ckpt_dir = os.path.join('checkpoints', run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, 'config.json'), 'w') as f:
        json.dump(asdict(cfg), f, indent=2)

    train_loader, val_loader = make_loaders(samples, cfg)
    model = SegmentationTransformer(cfg).to(device)
    print(f"Parameters: {count_params(model):,}")

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader)) * cfg.max_epochs
    sched = OneCycleLR(opt, max_lr=cfg.lr, total_steps=total_steps, pct_start=cfg.warmup_pct)

    start_epoch = 0
    best_f1 = -1.0
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler']); start_epoch = ck['epoch'] + 1
        best_f1 = ck.get('best_f1', -1.0)
        print(f"Resumed from {resume} at epoch {start_epoch}")

    bad = 0
    num_batches = max(1, len(train_loader))
    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        run_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
          try:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad()
            logits = model(batch)
            loss, parts = compute_loss(logits, batch, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step()
            run_loss += loss.item()
            if (step + 1) % 10 == 0 or step == num_batches - 1:
                print(f"\r  Epoch {epoch+1}/{cfg.max_epochs}  batch {step+1}/{num_batches}  "
                      f"loss={loss.item():.4f}  P={parts['precision']:.2f} R={parts['recall']:.2f} "
                      f"F1={parts['f1']:.2f}  lr={sched.get_last_lr()[0]:.2e}   ",
                      end='', flush=True)
          except RuntimeError as exc:
            if 'out of memory' in str(exc).lower():
                _report_error(f"training ran out of memory at epoch {epoch+1} "
                              f"batch {step+1}/{num_batches} - reduce --batch_size", exc, fatal=True)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                _report_error(f"training step failed at epoch {epoch+1} "
                              f"batch {step+1}/{num_batches}", exc, fatal=True)
            raise
        print()

        val = evaluate(model, val_loader, device, cfg)
        nb = num_batches
        print(f"Epoch {epoch:03d}  train_loss={run_loss/nb:.4f}  "
              f"val_loss={val['loss']:.4f}  val_P={val['precision']:.3f}  "
              f"val_R={val['recall']:.3f}  val_F1={val['f1']:.3f}  {time.time()-t0:.0f}s")

        ck = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
              'scheduler': sched.state_dict(), 'best_f1': best_f1, 'config': asdict(cfg)}
        try:
            torch.save(ck, os.path.join(ckpt_dir, 'last.pt'))
            if val['f1'] > best_f1:
                best_f1 = val['f1']; ck['best_f1'] = best_f1
                torch.save(ck, os.path.join(ckpt_dir, 'best.pt'))
                print(f"  ✓ new best (val_F1={best_f1:.4f})"); bad = 0
            else:
                bad += 1
                if bad >= cfg.early_stop_patience:
                    print(f"Early stopping at epoch {epoch}."); break
        except Exception as exc:
            _report_error(f"saving checkpoint for epoch {epoch+1} to '{ckpt_dir}'", exc, fatal=True)

    print(f"\nDone. Best val F1: {best_f1:.4f}  ->  {ckpt_dir}/best.pt")
    return {'best_f1': best_f1, 'ckpt_dir': ckpt_dir, 'best_ckpt': os.path.join(ckpt_dir, 'best.pt')}


# =============================================================================
# INFERENCE
# =============================================================================

def load_model(checkpoint: str, device):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: '{checkpoint}'.")
    try:
        ck = torch.load(checkpoint, map_location=device)
        cfg = Config(**{k: v for k, v in ck['config'].items()
                        if k in Config.__dataclass_fields__ and Config.__dataclass_fields__[k].init})
        model = SegmentationTransformer(cfg).to(device)
        model.load_state_dict(ck['model']); model.eval()
    except Exception as exc:
        _report_error(f"loading checkpoint '{checkpoint}' (corrupt file?)", exc, fatal=True)
        raise
    print(f"Loaded {checkpoint}  (d_model={cfg.d_model}, layers={cfg.num_layers}, "
          f"{count_params(model):,} params, best_f1={ck.get('best_f1', float('nan')):.4f})")
    return model, cfg


def infer_segments(events: List[NoteEvent], probs: np.ndarray, cfg: Config,
                   threshold: float = 0.5, always_include_start: bool = True) -> List[int]:
    """
    STANDALONE thresholding function, as requested: given per-note boundary
    probabilities (aligned 1:1 with `events`), decide which measures are predicted
    theme starts. A segment can ONLY start on a note at grid_step==0 - this is
    enforced explicitly here, not left to the model. Notes NOT at grid_step==0 are
    never candidates, no matter their probability.

    Multiple instruments can legitimately hit simultaneously on the downbeat of
    one bar, giving several candidate notes for the same bar; we take the MAX
    probability among them, since if ANY of the simultaneous hits carries strong
    boundary signal that's enough to call the whole bar a boundary.

    Returns a sorted list of bar (measure) indices predicted to start a new theme.
    """
    if len(events) != len(probs):
        raise ValueError(f"events ({len(events)}) and probs ({len(probs)}) length mismatch")
    bar_prob: Dict[int, float] = {}
    for e, p in zip(events, probs):
        if e.grid_step == 0:                      # ONLY measure-start notes are candidates
            bar_prob[e.bar] = max(bar_prob.get(e.bar, 0.0), float(p))
    starts = sorted(bar for bar, p in bar_prob.items() if p >= threshold)
    if always_include_start and events:
        first_bar = min(e.bar for e in events)
        if first_bar not in starts:
            starts = sorted(starts + [first_bar])
    return starts


def _chunk_batch(arr, cfg, device):
    t = lambda a, dt=torch.long: torch.tensor(a, dtype=dt).unsqueeze(0).to(device)
    return {
        'instruments':      t(np.clip(arr['instruments'], 0, cfg.num_instruments - 1)),
        'velocities':       t(np.clip(arr['velocities'], 0, cfg.velocity_bins - 1)),
        'grid_steps':       t(np.clip(arr['grid_steps'], 0, cfg.grid_steps_per_bar - 1)),
        'positions':        t(np.clip(arr['positions'], 0, cfg.max_position - 1)),
        'iois':             t(np.clip(arr['iois'], 0, cfg.max_ioi_steps)),
        'is_measure_start': t(arr['is_measure_start']),
        'pad_mask':         torch.zeros(1, len(arr['instruments']), dtype=torch.bool, device=device),
    }


def bar_to_seconds(bar: int, midi, cfg: Config) -> float:
    """
    Convert a bar index to its start time in seconds, tempo-change-AWARE - uses
    the file's own pretty_midi.tick_to_time(), not a naive constant sec_per_bar.
    A file with a mid-file tempo change would otherwise report (and slice/segment)
    every boundary after the change at the WRONG time, compounding the same class
    of error fixed in _load_midi_notes_impl's tick conversion, just in reverse.
    """
    bar_ticks_internal = bar * cfg.ticks_per_beat * cfg.beats_per_bar   # our internal ticks
    beats = bar_ticks_internal / cfg.ticks_per_beat
    native_tick = int(round(beats * midi.resolution))
    return float(midi.tick_to_time(native_tick))


def compute_segment_boundaries(model, cfg: Config, input_path: str, threshold: float = 0.5,
                               context_overlap: float = 0.25) -> Dict:
    """
    Core of the inference pipeline, factored out so a caller that already has the
    model loaded in memory (e.g. a UI that shouldn't reload a checkpoint from disk
    on every request) can reuse it directly. Returns a dict with: 'starts' (bar
    indices where a new theme begins), 'tempo' (bpm, a single representative value
    - for PRECISE bar->seconds conversion in a file with tempo changes, use
    bar_to_seconds(bar, result['midi'], cfg) instead of tempo*beats_per_bar math),
    'total_measures', 'events' (the loaded NoteEvent list), 'probs' (per-note
    boundary probabilities, aligned 1:1 with 'events'), 'midi' (the loaded
    pretty_midi.PrettyMIDI object, for tempo-aware conversions downstream).
    """
    device = next(model.parameters()).device
    events = load_midi_notes(input_path, cfg)
    if not events:
        raise ValueError(f"No drum events found in {input_path}")
    events.sort(key=lambda e: (e.bar, e.grid_step, e.instrument))

    arr = events_to_arrays(events, cfg)
    N = len(events)
    L = cfg.max_seq_len
    overlap = max(1, int(L * context_overlap))

    probs_sum = np.zeros(N); counts = np.zeros(N)
    start = 0
    while start < N:
        end = min(start + L, N)
        chunk = {k: v[start:end] for k, v in arr.items()}
        b = _chunk_batch(chunk, cfg, device)
        p = model.infer_probs(b).squeeze(0).cpu().numpy()[:end - start]
        w = np.ones(end - start)
        ramp = min(overlap, (end - start) // 4)
        if ramp > 0 and start > 0:
            w[:ramp] = np.linspace(0, 1, ramp)
        if ramp > 0 and end < N:
            w[-ramp:] = np.linspace(1, 0, ramp)
        probs_sum[start:end] += p * w
        counts[start:end] += w
        if end >= N:
            break
        start += L - overlap
    counts = np.maximum(counts, 1)
    probs = probs_sum / counts

    starts = infer_segments(events, probs, cfg, threshold=threshold)

    pm = pretty_midi.PrettyMIDI(input_path)
    tempo = 120.0
    try:
        tc = pm.get_tempo_changes()
        if len(tc[1]) > 0:
            tempo = float(tc[1][0])
    except Exception:
        pass

    return {'starts': starts, 'tempo': tempo, 'total_measures': measure_count(events),
            'events': events, 'probs': probs, 'midi': pm}


def segment_file(checkpoint: str, input_path: str, threshold: float = 0.5,
                 context_overlap: float = 0.25, output_midi: Optional[str] = None,
                 output_json: Optional[str] = None):
    """
    Full inference pipeline: load a real MIDI file, run the model in overlapping
    chunks (for files longer than max_seq_len), blend per-note probabilities in
    the overlap region, then call infer_segments(). Optionally writes MIDI Marker
    events at each detected boundary, and/or a JSON report.
    """
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model, cfg = load_model(checkpoint, device)

    result = compute_segment_boundaries(model, cfg, input_path, threshold=threshold,
                                        context_overlap=context_overlap)
    starts, tempo, midi = result['starts'], result['tempo'], result['midi']
    print(f"Loaded {len(result['events'])} events from {input_path} "
          f"({result['total_measures']} measures)")
    # tempo-change-AWARE seconds for each boundary (not a naive constant sec_per_bar)
    starts_sec = [bar_to_seconds(b, midi, cfg) for b in starts]

    print(f"\n── Predicted theme boundaries (threshold={threshold}) ─────────────")
    for bar, sec in zip(starts, starts_sec):
        print(f"  measure {bar + 1:4d}   (~{sec:7.2f}s)")
    print(f"  -> {len(starts)} segments detected across {result['total_measures']} measures")
    print(f"──────────────────────────────────────────────────────────────────\n")

    if output_json:
        report = {'input': input_path, 'threshold': threshold, 'tempo_bpm': tempo,
                  'total_measures': result['total_measures'],
                  'segment_start_measures': starts,
                  'segment_start_seconds': [round(s, 3) for s in starts_sec]}
        with open(output_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report written -> {output_json}")

    if output_midi:
        try:
            # DESIGN: this pretty_midi version has no Marker class; pretty_midi.Lyric
            # has the same (text, time) shape and is the standard substitute for
            # embedding readable text-events at specific timestamps in a MIDI file.
            pm = pretty_midi.PrettyMIDI(input_path)
            pm.lyrics = [pretty_midi.Lyric(f"Theme_{i+1}", sec)
                        for i, sec in enumerate(starts_sec)]
            pm.write(output_midi)
            print(f"MIDI with theme markers (as lyric/text events) written -> {output_midi}")
        except Exception as exc:
            _report_error(f"writing marker MIDI to '{output_midi}'", exc)

    return starts


# =============================================================================
# CLI
# =============================================================================

def main():
    seed_everything(GLOBAL_SEED)
    p = argparse.ArgumentParser(description="Drum MIDI theme temporal segmentation "
                                            "(dataset + train + infer, single file)")
    p.add_argument('--mode', required=True, choices=['dataset', 'train', 'infer'])
    p.add_argument('--run_name', default='seg_run')

    # dataset mode
    p.add_argument('--data_dir', default='./midi_collection')
    p.add_argument('--cache', default='cache/segments.pkl')
    p.add_argument('--num_samples', type=int, default=300,
                   help='DATASET: how many synthetic concatenated training sequences to build.')
    p.add_argument('--exclude_substring', default=None,
                   help="DATASET: filename substring that excludes a file as a theme "
                        "(case-insensitive, default 'song').")
    p.add_argument('--max_theme_measures', type=int, default=None,
                   help='DATASET: themes longer than this (in measures) are excluded (default 10).')
    p.add_argument('--min_themes_per_sample', type=int, default=None,
                   help='DATASET: min distinct themes concatenated per training sample (default 30).')
    p.add_argument('--max_themes_per_sample', type=int, default=None,
                   help='DATASET: max distinct themes concatenated per training sample (default 40).')
    p.add_argument('--max_repeat', type=int, default=None,
                   help='DATASET: max times a theme may repeat back-to-back (default 10).')
    p.add_argument('--family_similarity', type=float, default=None,
                   help='DATASET: filename-similarity ratio (0-1) above which two themes '
                        'are treated as the same family and never both used in one sample '
                        '(default 0.95 - see DESIGN note in the source for the tradeoff).')

    # train mode
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--d_model', type=int, default=None)
    p.add_argument('--num_layers', type=int, default=None)
    p.add_argument('--max_seq_len', type=int, default=None)
    p.add_argument('--pos_weight', type=float, default=None,
                   help='TRAIN: class-imbalance weight for the rare positive (boundary) '
                        'class in the loss (default 8.0). Raise if precision is very high '
                        'but recall is very low (model too conservative); lower if the '
                        'reverse. The dataset-build step prints the true imbalance ratio '
                        'as a reference.')
    p.add_argument('--num_workers', type=int, default=None)
    p.add_argument('--resume', default=None)

    # infer mode
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--input', default=None)
    p.add_argument('--threshold', type=float, default=0.5,
                   help='INFER: probability threshold for calling a measure a theme start.')
    p.add_argument('--context_overlap', type=float, default=0.25,
                   help='INFER: chunk overlap fraction for files longer than max_seq_len.')
    p.add_argument('--output_json', default=None, help='INFER: write a JSON report here.')
    p.add_argument('--output_midi', default=None,
                   help='INFER: write a copy of the input MIDI with Marker events at each '
                        'detected boundary.')

    args = p.parse_args()
    cfg = Config()

    def apply_overrides(c):
        if args.exclude_substring is not None: c.exclude_substring = args.exclude_substring
        if args.max_theme_measures is not None: c.max_theme_measures = args.max_theme_measures
        if args.min_themes_per_sample is not None: c.min_themes_per_sample = args.min_themes_per_sample
        if args.max_themes_per_sample is not None: c.max_themes_per_sample = args.max_themes_per_sample
        if args.max_repeat is not None: c.max_repeat = args.max_repeat
        if args.family_similarity is not None: c.family_similarity = args.family_similarity
        if args.epochs is not None: c.max_epochs = args.epochs
        if args.batch_size is not None: c.batch_size = args.batch_size
        if args.lr is not None: c.lr = args.lr
        if args.d_model is not None: c.d_model = args.d_model
        if args.num_layers is not None: c.num_layers = args.num_layers
        if args.max_seq_len is not None: c.max_seq_len = args.max_seq_len
        if args.pos_weight is not None: c.pos_weight = args.pos_weight
        if args.num_workers is not None: c.num_workers = args.num_workers
        return c
    cfg = apply_overrides(cfg)

    if args.mode == 'dataset':
        build_cache(args.data_dir, args.cache, cfg, args.num_samples)

    elif args.mode == 'train':
        if not os.path.exists(args.cache):
            print(f"Cache not found: {args.cache}. Run --mode dataset first.")
            sys.exit(1)
        with open(args.cache, 'rb') as f:
            blob = pickle.load(f)
        samples = blob['samples']
        saved = blob.get('cfg', {})
        cfg = Config(**{k: v for k, v in saved.items()
                       if k in Config.__dataclass_fields__ and Config.__dataclass_fields__[k].init})
        cfg = apply_overrides(cfg)
        train(cfg, samples, args.run_name, resume=args.resume)

    elif args.mode == 'infer':
        if not args.checkpoint or not args.input:
            p.error("--checkpoint and --input are required for infer mode")
        segment_file(args.checkpoint, args.input, threshold=args.threshold,
                    context_overlap=args.context_overlap,
                    output_midi=args.output_midi, output_json=args.output_json)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted] Stopped by user (Ctrl-C).")
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as exc:
        _report_error("fatal error (see traceback below)", exc, fatal=True)
        sys.exit(1)
