#!/usr/bin/env python3
"""
==============================================================================
 FIND SIMILAR DRUM MIDI GROOVES  (single-file: index + query)
==============================================================================

WHAT THIS PROGRAM DOES
-----------------------
Given a QUERY drum MIDI file, ranks every file in your training MIDI library by
how similar it "feels" rhythmically and dynamically - the same idea as Superior
Drummer 3's "Tap2Find" (drag-a-MIDI-in mode): drop in a groove you like, get
back a ranked list of the closest matches from your library.

NOTE ON RELATION TO TAP2FIND: Toontrack does not publish Tap2Find's internal
algorithm, so this is NOT a reproduction of it - it's an independent, from-
scratch similarity method built on standard, explainable signal-processing
ideas (rhythmic-pattern histograms + dynamics profile + tempo/density), with
every step visible and tunable. Treat it as an equivalent TOOL for the same
JOB, not a clone of SD3's proprietary matching.

HOW SIMILARITY IS COMPUTED
----------------------------
Each MIDI file is reduced to a compact "fingerprint" with four core parts:
  1) RHYTHM   - for each drum voice (kick/snare/hats/toms/...), a histogram of
                WHERE in the bar it tends to land (fraction of that voice's
                hits falling on each 16th-note grid position, folded onto ONE
                representative bar via modulo - see DESIGN note below).
  2) VELOCITY - for each drum voice, its average loudness (0-1), capturing the
                dynamic/energy profile (a soft brushed groove vs. a hard-hit one).
  3) DENSITY  - overall notes-per-bar, a simple complexity signal.
  4) TEMPO    - the file's BPM.
Two files' similarity is a WEIGHTED BLEND of cosine similarity on (1) and (2)
and closeness on (3) and (4) - weights are CLI-tunable so you can decide how
much tempo or raw density should matter versus the rhythmic pattern itself.

OPTIONAL ARTICULATION-FLATTENED COMPONENTS (off by default, weight 0.0)
-------------------------------------------------------------------------
The core RHYTHM component above compares each of the 13 drum-voice classes
SEPARATELY, so it naturally penalizes an articulation swap even when the
underlying pattern is identical - e.g. the same 8th-note pattern played on
closed vs. open hi-hat looks "different" to it, because those are different
rows in the histogram. Three optional components restore pattern-only
matching for exactly the cases where articulation is usually the least
important part of the idea:
  --weight_hihat_pattern   hi-hat, ALL articulations (closed/open/pedal)
                           flattened into ONE row before comparing.
  --weight_tom_pattern     toms, flattened to TWO functional groups - floor
                           tom vs. rack toms - instead of three raw pitches.
  --weight_cymbal_pattern  cymbals (crash/ride/ride-bell/china/splash), ALL
                           flattened into ONE row before comparing.
Each defaults to 0.0 (not considered at all - identical to the original
behavior). Give one a positive weight to have "does the hi-hat pattern feel
the same, regardless of articulation" (etc.) factored into the blend.

OPTIONAL PER-INSTRUMENT VELOCITY FLOORS (index-time, off by default)
------------------------------------------------------------------------
--min_velocity_kick / _snare / _hihat / _toms / _cymbals (0-127, default 0)
exclude notes in that instrument group below the given velocity from the
fingerprint ENTIRELY - they never enter the rhythm histogram, the velocity
profile, or any of the flattened-pattern components. A ghost-note snare hit
and a full backbeat hit are rhythmically different in KIND, not just
loudness, so filtering lets you compare the "real" pattern without quiet
fill/ghost notes diluting it. These are baked into the index at build time
(same as --min_notes) - change them and rebuild with --mode index to apply.

DESIGN NOTE - why "fold onto one representative bar" instead of comparing
whole files directly
---------------------------------------------------------------------------
A 2-bar loop repeated 8 times and a 4-bar phrase played once should be able to
match each other if they have the same FEEL - file length and tempo shouldn't
matter to "does this feel similar." Folding every note's position modulo the
bar length (averaged across however many bars the file has) makes the
fingerprint invariant to file length and repeat count, and comparing histograms
instead of raw event sequences makes it tempo-invariant too (a loop played at
100 BPM and the "same feel" loop at 140 BPM still produce the same histogram).
This does trade away sensitivity to LONGER-scale structure (a build across 8
bars) - it captures "typical feel," not "exact performance," which is the right
tradeoff for "find me things that feel like this."

HOW IT IS USED - TWO MODES
------------------------------
  1) index : scan a MIDI library, extract a fingerprint for every file, cache it.
  2) query : given a query MIDI (in or outside the library), rank the cached
             library by similarity and print/save the top matches.

  pip install pretty_midi numpy

  python find_similar_grooves.py --mode index --data_dir "/path/to/MIDI" \
         --cache cache/groove_index.pkl

  python find_similar_grooves.py --mode query --cache cache/groove_index.pkl \
         --query "/path/to/some_groove.mid" --top_k 15

Search for "DESIGN:" to find inline rationale for specific decisions.
"""

import os
import sys
import re
import glob
import json
import time
import pickle
import difflib
import argparse
import traceback
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False
    print("Warning: pretty_midi not installed - MIDI I/O disabled. "
          "Install with: pip install pretty_midi")


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
    ticks_per_beat:  int = 480
    grid_resolution: int = 4       # subdivisions per beat -> 16th-note grid
    beats_per_bar:   int = 4
    min_notes:       int = 4       # files with fewer drum notes are skipped

    # similarity blend weights (auto-normalized to sum to 1 at query time)
    weight_rhythm:   float = 0.5
    weight_velocity: float = 0.2
    weight_density:  float = 0.15
    weight_tempo:    float = 0.15
    # OPTIONAL extra components - articulation-flattened per-instrument-GROUP
    # patterns. Default 0.0 = not considered at all (matches existing behavior
    # exactly). Set a positive weight to factor a group's pattern into the blend
    # REGARDLESS of which specific articulation was used within that group - see
    # the module DESIGN note on why this differs from the main rhythm component.
    weight_hihat_pattern:  float = 0.0    # closed+open+pedal hi-hat, flattened to 1
    weight_tom_pattern:    float = 0.0    # all toms, flattened to {floor, rack} = 2
    weight_cymbal_pattern: float = 0.0    # crash+ride+bell+china/splash, flattened to 1
    # falloff scales for the two scalar features (bigger = more forgiving)
    density_scale:   float = 8.0    # notes/bar difference that "feels different"
    tempo_scale:     float = 25.0   # BPM difference that "feels different"

    # ── Per-instrument-group velocity floors (INDEX-time; baked into the finger-
    # print, so changing these requires rebuilding the index - same as min_notes).
    # DESIGN: a ghost-note snare hit and a full backbeat hit are rhythmically
    # different in KIND, not just loudness - lumping them together can dilute what
    # "the pattern" actually is. Each floor is 0-127; a note in that group with
    # velocity BELOW its floor is dropped entirely before it's counted into the
    # rhythm histogram, velocity profile, or any of the flattened-pattern
    # components. Default 0 = no filtering (identical to current behavior).
    min_velocity_kick:    int = 0
    min_velocity_snare:   int = 0
    min_velocity_hihat:   int = 0
    min_velocity_toms:    int = 0
    min_velocity_cymbals: int = 0

    def __post_init__(self):
        self.ticks_per_grid     = self.ticks_per_beat // self.grid_resolution
        self.grid_steps_per_bar = self.grid_resolution * self.beats_per_bar


# =============================================================================
# MINIMAL GM DRUM MAP  (self-contained - no dependency on other project files)
# =============================================================================
GM_DRUM_MAP = {
    35: 0, 36: 0,
    38: 1, 40: 1, 37: 1, 39: 1,
    42: 2, 44: 2,
    46: 3,
    41: 4, 43: 4,
    45: 5, 47: 5,
    48: 6, 50: 6,
    49: 7, 57: 7,
    51: 8, 59: 8,
    53: 9,
    52: 10, 55: 10,
    56: 11, 58: 11, 60: 11, 62: 11,
}
NUM_INSTRUMENTS = 13
DRUM_CLASS_NAMES = {
    0: "Kick", 1: "Snare", 2: "HH-Closed", 3: "HH-Open", 4: "LowTom",
    5: "MidTom", 6: "HighTom", 7: "Crash", 8: "Ride", 9: "RideBell",
    10: "China/Splash", 11: "Perc", 12: "Other",
}

# ── Articulation-flattening groups for the optional pattern components ──────────
# DESIGN: "does the hi-hat pattern feel the same" should be TRUE even if one
# groove plays it all-closed and another opens the hat for an accent - that's
# the same rhythmic idea on the same physical instrument, just a different
# articulation choice. The main RHYTHM component (per the module docstring)
# already compares each of the 13 classes separately, so it naturally penalizes
# an articulation swap even when the underlying pattern is identical. These
# three optional components restore that invariance for exactly the cases
# where "which articulation" is usually the LEAST important part of the idea -
# hi-hats, toms (functionally: floor vs rack), and cymbal choice.
HIHAT_CLASSES   = (2, 3)          # HH-Closed, HH-Open (pedal is already folded into 2)
FLOOR_TOM_CLASS = (4,)            # both floor-tom pitches already share class 4
RACK_TOM_CLASSES = (5, 6)         # mid + high toms = the rack toms
CYMBAL_CLASSES  = (7, 8, 9, 10)   # Crash, Ride, RideBell, China/Splash

# ── Per-instrument-group velocity-floor lookup ──────────────────────────────────
# Maps each drum class -> which Config field holds its minimum velocity (or None
# if that class has no dedicated floor - perc/other pass through unfiltered since
# only these five groups were asked for).
_VELOCITY_FLOOR_FIELD = {
    0: 'min_velocity_kick',
    1: 'min_velocity_snare',
    2: 'min_velocity_hihat', 3: 'min_velocity_hihat',
    4: 'min_velocity_toms', 5: 'min_velocity_toms', 6: 'min_velocity_toms',
    7: 'min_velocity_cymbals', 8: 'min_velocity_cymbals',
    9: 'min_velocity_cymbals', 10: 'min_velocity_cymbals',
}


def _passes_velocity_floor(inst: int, velocity: int, cfg: Config) -> bool:
    field = _VELOCITY_FLOOR_FIELD.get(inst)
    if field is None:
        return True   # no floor defined for this class (perc/other) -> always keep
    return velocity >= getattr(cfg, field)


# =============================================================================
# FINGERPRINT EXTRACTION
# =============================================================================

def _extract_fingerprint_impl(path: str, cfg: Config) -> Optional[Dict]:
    midi = pretty_midi.PrettyMIDI(path)
    notes = [n for t in midi.instruments if t.is_drum for n in t.notes]
    if len(notes) < cfg.min_notes:
        return None
    tempo = 120.0
    try:
        tc = midi.get_tempo_changes()
        if len(tc[1]) > 0:
            tempo = float(tc[1][0])
    except Exception:
        pass

    spb = cfg.grid_steps_per_bar
    bar_ticks = cfg.ticks_per_beat * cfg.beats_per_bar
    # rhythm[inst] = counts per grid position (0..spb-1), folded onto ONE bar
    rhythm_counts = np.zeros((NUM_INSTRUMENTS, spb), dtype=np.float64)
    vel_sum = np.zeros(NUM_INSTRUMENTS, dtype=np.float64)
    vel_n   = np.zeros(NUM_INSTRUMENTS, dtype=np.float64)
    max_bar = 0
    used_notes = 0
    for n in notes:
        inst = GM_DRUM_MAP.get(n.pitch, -1)
        if inst < 0:
            continue
        if not _passes_velocity_floor(inst, n.velocity, cfg):
            continue   # below this instrument-group's floor -> excluded entirely
        # DESIGN: convert via pretty_midi's own tempo-change-AWARE time_to_tick(),
        # not a naive `time * tempo`. A file with a mid-file tempo change would
        # otherwise place every note after the change at the WRONG grid position
        # (verified: a note exactly 1 beat after a 120->240bpm change landed at
        # tick 480 under the naive formula vs. its true tick 720 - a full 2-step
        # grid error). time_to_tick() returns ticks in the FILE's own resolution,
        # so we rescale to our fixed internal ticks_per_beat via a beats-elapsed
        # intermediate (resolution-independent).
        beats_elapsed = midi.time_to_tick(n.start) / midi.resolution
        tick = int(round(beats_elapsed * cfg.ticks_per_beat))
        grid_tick = round(tick / cfg.ticks_per_grid) * cfg.ticks_per_grid
        bar = grid_tick // bar_ticks
        step = (grid_tick % bar_ticks) // cfg.ticks_per_grid   # 0..spb-1, folded
        rhythm_counts[inst, int(step)] += 1
        vel_sum[inst] += n.velocity
        vel_n[inst] += 1
        max_bar = max(max_bar, int(bar))
        used_notes += 1
    if used_notes < cfg.min_notes:
        return None
    n_bars = max_bar + 1

    # RHYTHM block: normalize each instrument's histogram to sum to 1 (a
    # "where in the bar does this voice typically land" distribution). An
    # instrument that never appears gets an all-zero row (naturally penalizes
    # matches with files that use a very different subset of the kit).
    row_sums = rhythm_counts.sum(axis=1, keepdims=True)
    rhythm_norm = np.divide(rhythm_counts, row_sums, out=np.zeros_like(rhythm_counts),
                            where=row_sums > 0)

    # OPTIONAL flattened-articulation pattern components. Sum RAW counts across
    # each group's classes first, THEN normalize the combined row to sum to 1 -
    # summing already-normalized rows would double-count unevenly depending on
    # how many hits each articulation individually had.
    def _flatten(counts_2d, class_ids):
        combined = counts_2d[list(class_ids)].sum(axis=0)         # (spb,)
        s = combined.sum()
        return (combined / s) if s > 0 else combined

    hihat_pattern  = _flatten(rhythm_counts, HIHAT_CLASSES).astype(np.float32)     # (spb,)
    cymbal_pattern = _flatten(rhythm_counts, CYMBAL_CLASSES).astype(np.float32)    # (spb,)
    tom_pattern = np.stack([
        _flatten(rhythm_counts, FLOOR_TOM_CLASS),
        _flatten(rhythm_counts, RACK_TOM_CLASSES),
    ]).astype(np.float32)                                                          # (2, spb)

    # VELOCITY block: mean velocity per instrument, normalized to [0,1]. Silent
    # (unused) instruments get 0 - again a natural, meaningful penalty.
    vel_mean = np.divide(vel_sum, vel_n, out=np.zeros_like(vel_sum), where=vel_n > 0) / 127.0

    density = used_notes / max(1, n_bars)   # notes per bar

    return {
        'rhythm': rhythm_norm.astype(np.float32),      # (NUM_INSTRUMENTS, spb)
        'velocity': vel_mean.astype(np.float32),        # (NUM_INSTRUMENTS,)
        'hihat_pattern': hihat_pattern,                  # (spb,)
        'tom_pattern': tom_pattern,                      # (2, spb) - [floor, rack]
        'cymbal_pattern': cymbal_pattern,                # (spb,)
        'density': float(density),
        'tempo': float(tempo),
        'n_bars': int(n_bars),
        'n_notes': int(used_notes),
        'instruments_used': sorted(set(GM_DRUM_MAP.get(n.pitch, -1) for n in notes) - {-1}),
    }


def extract_fingerprint(path: str, cfg: Config) -> Optional[Dict]:
    """Load a MIDI file and reduce it to a similarity fingerprint. Returns None
    (and reports why) on any parse failure or if too sparse to be meaningful."""
    if not HAS_PRETTY_MIDI:
        raise ImportError("pretty_midi required for MIDI I/O")
    try:
        return _extract_fingerprint_impl(path, cfg)
    except Exception as exc:
        _report_error(f"extracting fingerprint from '{os.path.basename(path)}'", exc)
        return None


# =============================================================================
# FILENAME NORMALIZATION  (optional de-duplication of near-identical takes)
# =============================================================================

def normalize_basename(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0].lower()
    name = re.sub(r'[_\-]+', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = re.sub(r'\s*\d+\s*$', '', name).strip()
    return name


# =============================================================================
# INDEX BUILDING
# =============================================================================

def _process_one_file(args) -> Optional[Tuple[str, Dict]]:
    path, cfg = args
    fp = extract_fingerprint(path, cfg)
    if fp is None:
        return None
    return (path, fp)


def build_index(data_dir: str, cache_path: str, cfg: Config, num_workers: int = 8):
    paths = []
    for ext in ('mid', 'midi', 'MID', 'MIDI'):
        paths.extend(glob.glob(os.path.join(data_dir, '**', f'*.{ext}'), recursive=True))
    paths = sorted(set(paths))
    print(f"Found {len(paths)} MIDI files in {data_dir}")
    floors = [(name, v) for name, v in [
        ('kick', cfg.min_velocity_kick), ('snare', cfg.min_velocity_snare),
        ('hihat', cfg.min_velocity_hihat), ('toms', cfg.min_velocity_toms),
        ('cymbals', cfg.min_velocity_cymbals)] if v > 0]
    if floors:
        print(f"Velocity floors active: " + ", ".join(f"{n}≥{v}" for n, v in floors) +
              " (notes below these are excluded from the fingerprint entirely)")

    fingerprints: Dict[str, Dict] = {}
    failed = 0
    work = [(p, cfg) for p in paths]
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_process_one_file, w) for w in work]
        done = 0
        t0 = time.time()
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            if r is not None:
                fingerprints[r[0]] = r[1]
            else:
                failed += 1
            if done % 200 == 0 or done == len(futures):
                print(f"\r  processed {done}/{len(futures)}  "
                      f"({len(fingerprints)} indexed, {failed} skipped)  "
                      f"{time.time()-t0:.0f}s", end='', flush=True)
    print()
    if not fingerprints:
        print("ERROR: no usable MIDI files found - nothing to index.")
        sys.exit(1)

    # stack into arrays for fast vectorized similarity search at query time
    idx_paths = list(fingerprints.keys())
    N = len(idx_paths)
    spb = cfg.grid_steps_per_bar
    rhythm_mat   = np.zeros((N, NUM_INSTRUMENTS, spb), dtype=np.float32)
    velocity_mat = np.zeros((N, NUM_INSTRUMENTS), dtype=np.float32)
    hihat_mat    = np.zeros((N, spb), dtype=np.float32)
    tom_mat      = np.zeros((N, 2, spb), dtype=np.float32)
    cymbal_mat   = np.zeros((N, spb), dtype=np.float32)
    density_vec  = np.zeros(N, dtype=np.float32)
    tempo_vec    = np.zeros(N, dtype=np.float32)
    meta = []
    for i, p in enumerate(idx_paths):
        fp = fingerprints[p]
        rhythm_mat[i] = fp['rhythm']
        velocity_mat[i] = fp['velocity']
        hihat_mat[i] = fp['hihat_pattern']
        tom_mat[i] = fp['tom_pattern']
        cymbal_mat[i] = fp['cymbal_pattern']
        density_vec[i] = fp['density']
        tempo_vec[i] = fp['tempo']
        meta.append({'n_bars': fp['n_bars'], 'n_notes': fp['n_notes'],
                     'instruments_used': fp['instruments_used'],
                     'family': normalize_basename(p)})

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'cfg': asdict(cfg), 'paths': idx_paths, 'meta': meta,
                    'rhythm': rhythm_mat, 'velocity': velocity_mat,
                    'hihat_pattern': hihat_mat, 'tom_pattern': tom_mat,
                    'cymbal_pattern': cymbal_mat,
                    'density': density_vec, 'tempo': tempo_vec}, f, protocol=4)
    print(f"Indexed {N} files ({failed} skipped: parse errors or too sparse) -> {cache_path}")
    return idx_paths


def load_index(cache_path: str) -> Dict:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Index not found: '{cache_path}'. Run --mode index first.")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# SIMILARITY SEARCH
# =============================================================================

def _cosine_sim_batch(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """query_vec: (D,)  mat: (N,D)  ->  (N,) cosine similarities, safe for zero rows."""
    qn = np.linalg.norm(query_vec)
    mn = np.linalg.norm(mat, axis=1)
    dots = mat @ query_vec
    denom = qn * mn
    return np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)


def compute_similarities(query_fp: Dict, index: Dict, cfg: Config) -> np.ndarray:
    """Returns an (N,) array of blended similarity scores in [0,1] (higher=closer),
    one per file in the index, using the weighted components described in the
    module docstring. The three articulation-flattened components (hi-hat, tom,
    cymbal pattern) are OPTIONAL: only computed/included when their weight is
    nonzero, so old indexes built before this feature existed keep working as
    long as you don't request them."""
    N = index['rhythm'].shape[0]

    # 1) RHYTHM: flatten each file's (instrument x grid) histogram and cosine-compare
    rhythm_flat_idx = index['rhythm'].reshape(N, -1)
    rhythm_flat_q   = query_fp['rhythm'].reshape(-1)
    sim_rhythm = _cosine_sim_batch(rhythm_flat_q, rhythm_flat_idx)

    # 2) VELOCITY: per-instrument mean-velocity vector, cosine-compared
    sim_velocity = _cosine_sim_batch(query_fp['velocity'], index['velocity'])

    # 3) DENSITY: Gaussian falloff on notes/bar difference
    dd = index['density'] - query_fp['density']
    sim_density = np.exp(-0.5 * (dd / cfg.density_scale) ** 2)

    # 4) TEMPO: Gaussian falloff on BPM difference
    td = index['tempo'] - query_fp['tempo']
    sim_tempo = np.exp(-0.5 * (td / cfg.tempo_scale) ** 2)

    weights = [cfg.weight_rhythm, cfg.weight_velocity, cfg.weight_density, cfg.weight_tempo]
    sims = [sim_rhythm, sim_velocity, sim_density, sim_tempo]

    # 5-7) OPTIONAL articulation-flattened components - only touched if weighted > 0
    for name, w, key in [('hi-hat', cfg.weight_hihat_pattern, 'hihat_pattern'),
                         ('tom', cfg.weight_tom_pattern, 'tom_pattern'),
                         ('cymbal', cfg.weight_cymbal_pattern, 'cymbal_pattern')]:
        if w <= 0:
            continue
        if key not in index:
            raise ValueError(f"--weight_{key} was requested but this index was built "
                             f"before that feature existed and has no '{key}' data. "
                             f"Rebuild the index with --mode index to use it.")
        idx_mat = index[key].reshape(N, -1)
        q_vec = query_fp[key].reshape(-1)
        sims.append(_cosine_sim_batch(q_vec, idx_mat))
        weights.append(w)

    w = np.array(weights)
    w = w / max(1e-9, w.sum())    # auto-normalize so weights need not sum to 1
    blended = sum(wi * si for wi, si in zip(w, sims))
    return blended



def query_similar(query_path: str, cache_path: str, top_k: int = 10,
                  exclude_same_family: bool = False,
                  weight_overrides: Optional[Dict[str, float]] = None) -> List[Dict]:
    index = load_index(cache_path)
    cfg = Config(**{k: v for k, v in index['cfg'].items()
                    if k in Config.__dataclass_fields__})
    if weight_overrides:
        for k, v in weight_overrides.items():
            if v is not None:
                setattr(cfg, k, v)

    query_fp = extract_fingerprint(query_path, cfg)
    if query_fp is None:
        print(f"ERROR: could not extract a fingerprint from '{query_path}' "
              f"(parse failure or too few notes).")
        sys.exit(1)

    sims = compute_similarities(query_fp, index, cfg)

    query_abs = os.path.abspath(query_path)
    query_family = normalize_basename(query_path)
    order = np.argsort(-sims)   # descending

    results = []
    for i in order:
        p = index['paths'][i]
        if os.path.abspath(p) == query_abs:
            continue   # never match the query against itself if it's IN the index
        if exclude_same_family and index['meta'][i]['family'] == query_family:
            continue
        results.append({'path': p, 'similarity': float(sims[i]), **index['meta'][i]})
        if len(results) >= top_k:
            break

    print(f"\nQuery: {query_path}")
    print(f"  {query_fp['n_notes']} notes, {query_fp['n_bars']} bars, "
          f"tempo≈{query_fp['tempo']:.0f} bpm, density≈{query_fp['density']:.1f} notes/bar")
    print(f"  weights -> rhythm={cfg.weight_rhythm:.2f} velocity={cfg.weight_velocity:.2f} "
          f"density={cfg.weight_density:.2f} tempo={cfg.weight_tempo:.2f}")
    extra = []
    if cfg.weight_hihat_pattern > 0:  extra.append(f"hihat_pattern={cfg.weight_hihat_pattern:.2f}")
    if cfg.weight_tom_pattern > 0:    extra.append(f"tom_pattern={cfg.weight_tom_pattern:.2f}")
    if cfg.weight_cymbal_pattern > 0: extra.append(f"cymbal_pattern={cfg.weight_cymbal_pattern:.2f}")
    if extra:
        print(f"  + optional  -> {' '.join(extra)}")
    print(f"\n── Top {len(results)} most similar grooves ─────────────────────────")
    for rank, r in enumerate(results, 1):
        insts = ",".join(DRUM_CLASS_NAMES[i] for i in r['instruments_used'][:4])
        print(f"  {rank:2d}. sim={r['similarity']:.3f}  {os.path.basename(r['path']):40}  "
              f"({r['n_bars']}bar, {insts}...)")
    print("──────────────────────────────────────────────────────────────────\n")
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Find similar drum MIDI grooves "
                                            "(index + query, single file)")
    p.add_argument('--mode', required=True, choices=['index', 'query'])

    # index mode
    p.add_argument('--data_dir', default='./midi_collection')
    p.add_argument('--cache', default='cache/groove_index.pkl')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--min_notes', type=int, default=None,
                   help='INDEX: skip files with fewer drum notes than this (default 4).')
    p.add_argument('--min_velocity_kick', type=int, default=None,
                   help='INDEX: ignore kick notes below this velocity (0-127, default 0 '
                        '= no filtering). Baked into the index - rebuild to change.')
    p.add_argument('--min_velocity_snare', type=int, default=None,
                   help='INDEX: ignore snare notes below this velocity (0-127, default 0). '
                        'Useful for excluding ghost notes from the pattern comparison.')
    p.add_argument('--min_velocity_hihat', type=int, default=None,
                   help='INDEX: ignore hi-hat notes (any articulation) below this velocity '
                        '(0-127, default 0).')
    p.add_argument('--min_velocity_toms', type=int, default=None,
                   help='INDEX: ignore tom notes (any tom) below this velocity (0-127, '
                        'default 0).')
    p.add_argument('--min_velocity_cymbals', type=int, default=None,
                   help='INDEX: ignore cymbal notes (any cymbal) below this velocity '
                        '(0-127, default 0).')

    # query mode
    p.add_argument('--query', default=None, help='QUERY: path to the query MIDI file.')
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--exclude_same_family', action='store_true',
                   help='QUERY: exclude results whose filename looks like a near-'
                        'duplicate take of the query (e.g. "Fill 1" vs "Fill 14").')
    p.add_argument('--weight_rhythm', type=float, default=None,
                   help='QUERY: contribution of rhythmic-pattern similarity (default 0.5).')
    p.add_argument('--weight_velocity', type=float, default=None,
                   help='QUERY: contribution of dynamics/velocity-profile similarity (default 0.2).')
    p.add_argument('--weight_density', type=float, default=None,
                   help='QUERY: contribution of notes-per-bar closeness (default 0.15).')
    p.add_argument('--weight_tempo', type=float, default=None,
                   help='QUERY: contribution of tempo closeness (default 0.15). Set to 0 '
                        'to match purely on feel regardless of tempo.')
    p.add_argument('--weight_hihat_pattern', type=float, default=None,
                   help='QUERY: OPTIONAL - consider the hi-hat pattern with all '
                        'articulations (closed/open/pedal) flattened into one, so a '
                        'matching rhythm counts even if the articulation differs. '
                        'Default 0.0 = not considered at all.')
    p.add_argument('--weight_tom_pattern', type=float, default=None,
                   help='QUERY: OPTIONAL - consider the tom pattern, flattened to two '
                        'functional groups (floor tom, rack toms) rather than each tom '
                        'pitch separately. Default 0.0 = not considered at all.')
    p.add_argument('--weight_cymbal_pattern', type=float, default=None,
                   help='QUERY: OPTIONAL - consider the cymbal pattern with crash/ride/'
                        'ride-bell/china/splash all flattened into one, so a matching '
                        'accent rhythm counts even if the cymbal choice differs. '
                        'Default 0.0 = not considered at all.')
    p.add_argument('--output_json', default=None, help='QUERY: write results here as JSON.')

    args = p.parse_args()

    if args.mode == 'index':
        cfg = Config()
        if args.min_notes is not None:
            cfg.min_notes = args.min_notes
        if args.min_velocity_kick is not None: cfg.min_velocity_kick = args.min_velocity_kick
        if args.min_velocity_snare is not None: cfg.min_velocity_snare = args.min_velocity_snare
        if args.min_velocity_hihat is not None: cfg.min_velocity_hihat = args.min_velocity_hihat
        if args.min_velocity_toms is not None: cfg.min_velocity_toms = args.min_velocity_toms
        if args.min_velocity_cymbals is not None: cfg.min_velocity_cymbals = args.min_velocity_cymbals
        build_index(args.data_dir, args.cache, cfg, num_workers=args.num_workers)

    elif args.mode == 'query':
        if not args.query:
            p.error("--query is required for query mode")
        overrides = {'weight_rhythm': args.weight_rhythm, 'weight_velocity': args.weight_velocity,
                    'weight_density': args.weight_density, 'weight_tempo': args.weight_tempo,
                    'weight_hihat_pattern': args.weight_hihat_pattern,
                    'weight_tom_pattern': args.weight_tom_pattern,
                    'weight_cymbal_pattern': args.weight_cymbal_pattern}
        results = query_similar(args.query, args.cache, top_k=args.top_k,
                                exclude_same_family=args.exclude_same_family,
                                weight_overrides=overrides)
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results written -> {args.output_json}")


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
