"""
==============================================================================
 MIDI DRUM HUMANIZATION TRANSFORMER  (v3 - encoder-only)
==============================================================================

Learns how a HUMAN drummer plays - the micro-timing pushes/pulls and the
velocity dynamics - from a corpus of human-performed MIDI drums (built for the
Superior Drummer 3 groove library) and then applies that "feel" to stiff,
quantized, programmed MIDI. It learns entirely from data: there are NO
hand-written "make it swing" rules (with ONE optional, explicitly-labelled
physical-plausibility exception, see --fast_hit_cap below).

It NEVER changes which notes you played. Every hit keeps its exact original
pitch, so your specific kick/snare/tom/cymbal choices always survive untouched.
The model only adjusts each hit's VELOCITY and TIMING.

HOW IT WORKS
----------------------------------------------
Humanization is sequence LABELLING, not generation: the hits are fixed and known,
and each input hit maps to exactly one (velocity, timing-offset) output. So the
model is an ENCODER-ONLY transformer that tags every hit in ONE forward pass
(no autoregression). Training is self-supervised: a human groove is the TARGET,
and a de-humanized (quantized + velocity-coarsened) copy is the INPUT. The model
learns to turn stiff back into human. Faithful reconstruction re-uses each hit's
exact original pitch so notes are never altered.

HOW IT IS USED - THREE MODES
----------------------------
  1) cache : parse a folder of MIDI files once into a fast training cache.
  2) train : train a model on that cache (or on --synthetic data for a smoke test).
  3) infer : humanize a single MIDI file with a trained checkpoint.

  # install deps
  pip install torch pretty_midi numpy tqdm

  # 1. build a cache from your library (once)
  python drum_humanizer_v3.py --mode cache --data_dir "/path/to/SD3/MIDI" \
         --cache cache/samples.pkl

  # 2. train (default 'base' model; use --model_size deep for more capacity)
  python drum_humanizer_v3.py --mode train --cache cache/samples.pkl \
         --run_name sd3_v1 --epochs 100

  # smoke-test with zero real files:
  python drum_humanizer_v3.py --mode train --synthetic --epochs 3 --run_name smoke

  # 3. humanize a loop (architecture is read from the checkpoint automatically)
  python drum_humanizer_v3.py --mode infer --checkpoint checkpoints/sd3_v1/best.pt \
         --input my_loop.mid --output my_loop_human.mid --strength 0.85

==============================================================================
 COMMAND-LINE ARGUMENTS
==============================================================================
GLOBAL
  --mode {cache,train,infer,grid_search}   which stage to run (required)
  --run_name NAME              checkpoint/log subfolder under checkpoints/

CACHE MODE (build training data from MIDI)
  --data_dir DIR               folder of .mid/.midi files (searched recursively)
  --cache PATH                 output cache .pkl path
  --no_split_songs             don't split long files into sections (default: split)
  --section_bars N             section length in bars when splitting (default 16)
  --hop_bars N                 hop between sections; <section_bars = overlap (default 8)
  --no_quality_filter          keep flat/robotic sections (default: filter them out)
  --min_velocity_std X         flat-dynamics reject threshold, vel std (default 8)
  --min_velocity_range X       flat-dynamics reject threshold, vel range (default 24)
  --min_offset_std X           flat-timing reject threshold, ticks std (default 4)
  --min_offset_range X         flat-timing reject threshold, ticks range (default 12)

TRAIN MODE
  --cache PATH                 training cache from cache mode
  --synthetic                  train on generated data instead (smoke test)
  --synthetic_n N              how many synthetic grooves to generate
  --epochs N                   max epochs (early-stopping may end sooner)
  --batch_size N               batch size (lower it if you run out of GPU memory)
  --lr X                       learning rate
  --model_size NAME            tiny|small|base|deep|deeper|huge (capacity/depth)
  --d_model N / --num_layers N / --dropout X   fine-grained architecture overrides
  --target_mode {classification,regression}    prediction paradigm (see below)
  --velocity_bins {128|64|32}  velocity resolution (128 = lossless, default)
  --vel_soft_sigma X           ordinal-loss width for velocity (default 1.0)
  --off_soft_sigma X           ordinal-loss width for timing (default 1.0)
  --bar_rotation               enable phrase-start augmentation (see history #16)
  --bar_rotation_prob X        chance of rotating a given sample (default 0.5)
  --num_workers N              dataloader workers
  --resume PATH                resume from a checkpoint

GRID_SEARCH MODE (takes the TRAIN MODE args as its baseline, then sweeps 3 axes)
  The leaderboard is printed AND written to checkpoints/grid_results.json, so it
  survives the terminal - it holds every ranked run (val_loss, the winning combo,
  each best.pt path), the runs that were NOT comparable and why, and the final
  full-data retrain. Written once before the auto final train and again after it,
  so an OOM in that last long pass can't cost you the ranking. Read it with e.g.
    jq '.leaderboard[:5]' checkpoints/grid_results.json
  Per-epoch curves live in checkpoints/<run>/log.jsonl; TensorBoard: --logdir checkpoints

INFER MODE
  --checkpoint PATH            trained model (best.pt); architecture read from it
  --input FILE                 MIDI to humanize
  --output FILE                where to write the humanized MIDI
  --strength X                 how far to move dry->model: 0=none, 1=model, >1=exaggerate
  --strength_velocity X        override --strength for velocity only
  --strength_timing X          override --strength for timing only
  --temperature_vel X / --temperature_off X    sampling temperature per head
  --top_k N                    sampling top-k (shared)
  --top_k_velocity N / --top_k_timing N         per-head top-k overrides
  --vel_decode {expected,sample,argmax}         velocity decode strategy
  --off_decode {expected,sample,argmax}         timing decode strategy
  --intensity X                target energy/genre cue (0..1 or a 1..127 velocity)
  --context_overlap X          chunk overlap fraction for more context (default 0.33)
  --preserve_grid_distance X   keep baked-in timing on already-off-grid notes (0=off)
  --preserve_floor X           min timing-strength for the most off-grid notes
  --preserve_velocity_dynamics X   keep already-programmed dynamics (0=off)
  --preserve_velocity_floor X  min velocity-strength for the most expressive notes
  --fast_hit_cap               apply a blast-beat velocity ceiling (a RULE, off by default)
  --fast_hit_ceiling X         the ceiling for maximally-fast hits (default 85)

TWO PREDICTION PARADIGMS  (--target_mode)
  classification (default): binned heads + ordinal soft-target loss; keeps the full
      output distribution so sampling gives real variety.
  regression: continuous scalar heads; infinite resolution but predicts only the
      mean, so it can sound timid on multi-modal grooves. A/B both on your data.

==============================================================================
 HISTORY OF OPTIMIZATIONS / ADJUSTMENTS (one line each, in the order requested)
==============================================================================
  1.  Encoder-only rewrite: humanization is per-hit labelling, so predict every
      hit's velocity+timing in ONE forward pass instead of autoregressively.
  2.  Lossless 128-bin velocity with ordinal soft-target loss, so Superior Drummer's
      layer-switching Δ2–3 velocity nuances survive training.
  3.  Separate velocity vs timing controls (strength, temperature, decode, top-k,
      soft-sigma) so dynamics and feel can be tuned independently.
  4.  --strength extrapolation beyond 1.0, so you can exaggerate the model's
      humanization past what it predicts, with safety clamps.
  5.  Removed the note-changing / cymbal-swap feature entirely, because altering
      which notes you played is too dangerous - pitch is now always preserved.
  6.  Progressive --model_size presets (tiny->huge) so you can chase a drummer's
      long-range intuition with more depth, with dropout scaled up to fight overfit.
  7.  Grid-distance preservation: notes already off-grid (already humanized) get
      less model timing-influence, so baked-in feel in the input is protected.
  8.  Average-preserving de-humanization + an explicit --intensity signal, so the
      input's overall loudness reads as an intensity/genre cue you can also set.
  9.  Time-signature awareness: read the real meter(s) and each note's metric
      position, so 6/8 accents are learned differently from 4/4.
  10. Larger temporal window (--context_overlap) so every note is decided with a
      full measure of context on each side; long files handled section by section.
  11. Song-splitting in the cache builder, so whole-song files become coherent
      verse/chorus/bridge-sized training samples instead of one giant crop.
  12. Training-data quality filter: reject flat/robotic sections that would teach
      the model nothing (or teach it that "human" means "mechanical").
  13. Per-instrument IOI + explicit metric-strength feature, so repetitive same-
      instrument runs are legible and on-beats can be learned stronger than off-beats.
  14. Tempo fed as an input feature, so micro-timing feel can scale with speed.
  15. Flam / grace-note detection and preservation, so fast same-instrument pairs
      survive quantization instead of collapsing onto one grid step.
  16. Optional --fast_hit_cap: a speed-scaled velocity ceiling (e.g. blast-beat
      32nd snares ~85) as an explicit physical-plausibility RULE, off by default.
  17. Bar-rotation augmentation: drop the first 2/4/8 bars (for grooves ≥4/≥8/≥16
      bars) so the model doesn't over-index on phrase openings; input+target stay
      aligned so the humanization target is never corrupted.
  18. Richer training progress output: dataset provenance (original files -> split
      section-samples -> augmented) plus an in-place per-batch counter every 10 batches.

Search the source for "DESIGN:" to find inline rationale for any decision above.
"""

import os
import re
import sys
import glob
import math
import json
import time
import pickle
import random
import shutil
import argparse
import warnings
import traceback
import multiprocessing
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

try:
    from torch.amp import GradScaler, autocast
    HAS_AMP = True
except Exception:
    HAS_AMP = False

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False
    print("Warning: pretty_midi not installed - real MIDI I/O disabled. "
          "Install with: pip install pretty_midi")

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

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
    """
    Seed every RNG source this program touches, so runs are reproducible:
      • Python's stdlib `random` (augmentation choices, shuffling, windowing)
      • NumPy's global RNG (jitter, masks, synthetic-data generation)
      • PyTorch CPU + CUDA (weight init, dropout, any GPU randomness)
    Called once at startup (top of main()), and again before each individual run in
    grid_search() so every combo is reproducible independent of sweep order. Also
    used to derive per-worker seeds for DataLoader workers (see _worker_init_fn) so
    multi-process loading doesn't silently reintroduce nondeterminism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if verbose:
        print(f"[seed] {seed}")


def _worker_init_fn(worker_id: int):
    """
    DataLoader worker seeding. Each worker is a separate process - without this,
    workers can inherit correlated or platform-dependent RNG state instead of a
    reproducible one. Derive a distinct-but-deterministic seed per worker from the
    single global seed, so the whole pipeline stays reproducible under multiprocessing.
    """
    worker_seed = (GLOBAL_SEED + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Seed everything at IMPORT time, not just inside main(). This covers using the
# module as a library (`import drum_humanizer_v3 as D`) without going through the
# CLI - e.g. calling D.train(...)/D.generate_synthetic_events(...) directly in a
# notebook or another script. main() still re-seeds explicitly for clarity when run
# as a program; this line just guarantees the same reproducibility for library use.
seed_everything(GLOBAL_SEED, verbose=False)


# =============================================================================
# ERROR REPORTING HELPERS
# =============================================================================
# These make failures legible: instead of a raw traceback, they print the exact
# file:line where the error occurred plus a human-readable description of what the
# program was trying to do. Used at the crucial points (MIDI parsing, training
# step, checkpoint I/O, inference, top-level dispatch). They deliberately do NOT
# swallow errors silently - they explain, then either skip one item or re-raise.

def _error_location(exc: BaseException) -> str:
    """Return 'filename:line in function' for the DEEPEST frame of an exception -
    i.e. the exact line that actually raised, not where it was caught."""
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
    """Print a descriptive, location-pinpointed error message.
    context: what the program was doing (e.g. 'training step epoch 3 batch 210').
    fatal:   if True, also print the full traceback for debugging."""
    loc = _error_location(exc)
    print(f"\n[ERROR] {context}")
    print(f"        -> {type(exc).__name__} at {loc}: {exc}")
    if fatal:
        print("        Full traceback:")
        traceback.print_exc()


# =============================================================================
# CONFIGURATION
# =============================================================================

# ── Model size presets (progressively deeper) ──────────────────────────────────
# DESIGN: depth is the main lever for capturing higher-order musical intuition -
# "this is a build, lean into it" needs the net to relate distant hits across the
# whole phrase, which more layers do better. But deeper ≠ automatically smarter:
# with a fixed dataset a bigger model can overfit, so each preset also raises
# dropout as capacity grows, and early-stopping (already on) guards the rest.
# Params grow roughly linearly in layers and quadratically in d_model.
MODEL_PRESETS = {
    # name       d_model  nhead  num_layers  dim_ff  dropout   (~params @128 vel-bins)
    "small":  dict(d_model=192, nhead=6,  num_layers=4,  dim_feedforward=768,  dropout=0.10),   # ~2.3M
    "base":   dict(d_model=256, nhead=8,  num_layers=6,  dim_feedforward=1024, dropout=0.10),   # ~5.4M (default)
    "deep":   dict(d_model=320, nhead=8,  num_layers=10, dim_feedforward=1280, dropout=0.15),   # ~13M
    "deeper": dict(d_model=384, nhead=8,  num_layers=14, dim_feedforward=1536, dropout=0.20),   # ~26M
    "very_deep": dict(d_model=384, nhead=8, num_layers=20, dim_feedforward=1536, dropout=0.25),  # ~37M
    "huge":   dict(d_model=512, nhead=8,  num_layers=18, dim_feedforward=2048, dropout=0.20),   # ~59M
}
DEFAULT_PRESET = "base"

def apply_model_preset(cfg: "Config", name: str) -> "Config":
    """Overwrite the model-shape fields on cfg from a named preset."""
    if name not in MODEL_PRESETS:
        raise ValueError(f"Unknown model size '{name}'. Choose from: {list(MODEL_PRESETS)}")
    for k, v in MODEL_PRESETS[name].items():
        setattr(cfg, k, v)
    cfg.model_size = name
    return cfg


@dataclass
class Config:
    # ── Tokenization / musical grid ────────────────────────────────────────────
    ticks_per_beat:   int = 480     # PPQ used internally
    grid_resolution:  int = 16      # sub-divisions per beat (1/64 grid)
    beats_per_bar:    int = 4       # fallback / default meter when a file has none
    max_bars:         int = 64
    # Time-signature awareness. The internal grid stays fixed & fine; on top of it we
    # feed the model (a) each note's metric position WITHIN its real measure and
    # (b) a categorical time-signature id, so 6/8 accents are learned differently
    # from 4/4. beat_in_bar is quantized to this many slots per measure:
    beat_slots:       int = 48      # metric-position resolution within a measure
    num_metric_levels: int = 6      # subdivision-strength classes: downbeat..finer/off
    # Tempo conditioning: micro-timing feel scales with tempo (a +15ms push is lazy at
    # 80bpm, sloppy at 180bpm). We feed normalized tempo so the model can tighten/loosen
    # timing with speed. Range for normalization -> [0,1]:
    tempo_min_bpm:    float = 40.0
    tempo_max_bpm:    float = 240.0
    # Flam / grace-note detection: two same-instrument hits closer than this (in ms)
    # are a flam/drag/ruff - the micro-gap that DEFINES them would otherwise be
    # quantized away. We flag the grace hit and preserve the gap through reconstruction.
    flam_window_ms:   float = 35.0  # same-instrument hits within this = flam
    flam_gap_bins:    int = 12      # quantize the flam gap (0..flam_window_ms) into N bins
    # DESIGN: 128 bins = 1 MIDI velocity per bin = LOSSLESS. High-quality multi-
    # layer VSTis (e.g. SD3) can switch sample layer on a Δ2–3 velocity change, so
    # we keep full resolution. Set 64 (Δ2) or 32 (Δ4) only if you want coarser feel.
    velocity_bins:    int = 128     # velocity resolution (128=lossless, 64=Δ2, 32=Δ4)
    num_instruments:  int = 13      # drum voice classes (see GM_DRUM_MAP)
    max_ioi_steps:    int = 64      # inter-onset interval capped at N grid steps

    # ── Training-data quality filter ────────────────────────────────────────────
    # A flat, mechanical MIDI teaches the model nothing about humanization (or worse,
    # teaches that "human" == "robotic"). We measure each candidate sample's velocity
    # spread and timing spread; a sample is REJECTED only if it is flat on BOTH - i.e.
    # essentially a quantized robotic pattern. If it has real expression in either
    # dimension it is still a useful target for that dimension, so we keep it.
    quality_filter:      bool  = True
    min_velocity_std:    float = 8.0    # MIDI-velocity units; below -> "flat dynamics"
    min_velocity_range:  float = 24.0   # max-min MIDI velocity; below -> "flat dynamics"
    min_offset_std:      float = 4.0    # ticks; below -> "flat/quantized timing"
    min_offset_range:    float = 12.0   # max-min offset ticks; below -> "flat timing"

    # ── Bar-rotation augmentation (training only) ───────────────────────────────
    # A model that only ever sees files starting at bar 1 can over-index on phrase
    # OPENINGS. Bar rotation randomly drops the first few bars so phrases begin at
    # varied points. SAFE because the result is still a real performance, just
    # starting later; input and target rotate together, and per-note feel features
    # (metric position, IOI, velocity, offset) are invariant to whole-bar shifts.
    bar_rotation:        bool  = True    # ON by default (per user preference)
    bar_rotation_prob:   float = 0.6     # chance of applying it to a given sample
    bar_rotation_max:    int   = 2       # max lead bars to drop (only if ≥4 bars total)

    # ── "Smarter network" options (#3,#4,#6,#7,#12) - all toggleable ─────────────
    # #4 Relative positional encoding: replace the absolute learned position embedding
    #    with a relative-position attention bias (T5-style buckets), so the model keys
    #    off musical DISTANCE between hits rather than absolute index -> generalises to
    #    unseen phrase lengths/positions. Default ON (near-sure win, cheap).
    rel_pos_encoding:    bool  = True
    rel_pos_buckets:     int   = 32      # number of relative-distance buckets (each side)
    # #3 Autoregressive-over-beats timing bias: let the predicted offset of earlier
    #    beats nudge later ones, so timing has momentum (push/drift/recover) instead of
    #    independent per-note noise. Implemented as a lightweight causal offset-smoothing
    #    pass over beat-ordered notes at INFERENCE (train stays parallel). Default OFF.
    ar_timing:           bool  = False
    ar_timing_weight:    float = 0.3     # how strongly a note inherits prior beats' drift
    # #12 Per-instrument feel profiles: a learned per-voice bias added to each head, so
    #    the model can hold distinct "personalities" (hats rush, kick sits back). Cheap.
    per_instrument_feel: bool  = True
    # #6 Distribution-matching (regression only): add a spread (log-variance) output so
    #    the regression head predicts a DISTRIBUTION, not just the mean -> less timid.
    #    Trained with Gaussian NLL. Ignored in classification mode. Default OFF.
    distribution_match:  bool  = False
    # #7 Correlation-aware loss: reward getting the RELATIONSHIP between metric strength
    #    and velocity right (on-beat stronger), not just per-note accuracy. Small weight.
    correlation_loss:    bool  = False
    correlation_weight:  float = 0.1

    # ── Velocity loss/decoding (matters once bins are fine-grained) ─────────────
    # DESIGN: with 128 fine bins, flat cross-entropy treats bin 100 vs 101 as
    # "just as wrong" as 100 vs 20 - wasteful. We add a soft ordinal target
    # (Gaussian around the true bin) so near-misses are penalised less. Width is
    # in BINS. ~1.0 is gentle; set 0 to fall back to plain one-hot cross-entropy.
    vel_soft_sigma:   float = 1.0
    off_soft_sigma:   float = 1.0   # timing offset is ordinal too
    # Inference velocity decoding: "expected" = softmax-weighted mean over bins
    # (smooth, natural curves; only sensible with fine bins). "sample" = temperature
    # /top-k sampling. "argmax" = deterministic peak.
    vel_decode:       str = "expected"
    off_decode:       str = "sample"

    # ── Prediction paradigm (borrowed idea, made optional) ──────────────────────
    # "classification": binned heads + ordinal soft-target loss (default). Keeps a
    #     full output DISTRIBUTION, so sampling gives real humanization variety.
    # "regression":     continuous scalar heads + SmoothL1. Infinite resolution
    #     (no bins at all), but predicts only the conditional MEAN -> can sound
    #     timid on multi-modal grooves. Offered so you can A/B on real SD3 data.
    target_mode:      str = "classification"   # "classification" | "regression"

    # ── Model ──────────────────────────────────────────────────────────────────
    model_size:       str = DEFAULT_PRESET   # named preset (see MODEL_PRESETS)
    d_model:          int = 256
    nhead:            int = 8
    num_layers:       int = 6       # DESIGN: encoder-only, single stack
    dim_feedforward:  int = 1024
    dropout:          float = 0.1
    # DESIGN: EVERY sample is padded to this length, so it is a compute knob, not just a
    # ceiling - and attention is O(n^2), so an oversized value is brutally expensive.
    # Measured on a real 166k-sample library: median sample is 38 notes, p95 is 176, and
    # only 0.10% exceed 1024. At the old 1024 default that made 93.9% of every batch pure
    # padding - 16x wasted work in the linear layers and 123x in attention. 256 covers
    # 98.5% of samples uncropped; the rest are randomly cropped in training and chunked
    # with overlap at inference, so this is a speed win, not a quality loss. Raise it only
    # if your library is genuinely long-form (the cache build prints the note counts).
    max_seq_len:      int = 256     # max hits per sample (see DESIGN above before raising)

    # ── Training ───────────────────────────────────────────────────────────────
    batch_size:       int = 32
    lr:               float = 3e-4
    weight_decay:     float = 0.01
    max_epochs:       int = 100
    grad_clip:        float = 1.0
    warmup_pct:       float = 0.05
    label_smoothing:  float = 0.05
    vel_loss_weight:  float = 1.0
    off_loss_weight:  float = 2.0   # DESIGN: timing is harder -> more gradient
    val_split:        float = 0.05
    num_workers:      int = 4
    early_stop_patience: int = 20
    print_every:      int = 15      # batch-progress print interval (in-place, see train())

    # ── Derived ────────────────────────────────────────────────────────────────
    ticks_per_grid:     int = field(init=False)
    max_offset_ticks:   int = field(init=False)
    offset_bins:        int = field(init=False)
    grid_steps_per_bar: int = field(init=False)
    max_position:       int = field(init=False)

    def __post_init__(self):
        self.ticks_per_grid     = self.ticks_per_beat // self.grid_resolution
        self.max_offset_ticks   = self.ticks_per_grid          # ±1 grid cell
        self.offset_bins        = 2 * self.max_offset_ticks + 1
        self.grid_steps_per_bar = self.grid_resolution * self.beats_per_bar
        self.max_position       = self.grid_steps_per_bar * self.max_bars


# =============================================================================
# DRUM MAPPING  (13 expressive voice classes)
# =============================================================================
# DESIGN: fine-grained (separate open/closed HH, ride bell, china, ghost) because
# those articulation choices ARE part of feel. NOTE: the *class* is only used as a
# model INPUT feature - reconstruction uses each hit's ORIGINAL pitch, so real
# cymbal selection (e.g. which specific crash) is preserved exactly.

GM_DRUM_MAP = {
    35: 0, 36: 0,                      # Kick
    38: 1, 40: 1,                      # Snare
    37: 12,                            # Side stick / cross-stick (expressive)
    42: 2, 44: 2,                      # Closed hi-hat
    46: 3,                             # Open hi-hat
    41: 4, 43: 4,                      # Low tom
    45: 5, 47: 5,                      # Mid tom
    48: 6, 50: 6,                      # High tom
    49: 7, 57: 7,                      # Crash
    51: 8, 59: 8,                      # Ride
    53: 9,                             # Ride bell
    52: 10, 55: 10,                    # China / splash
    56: 11, 58: 11, 60: 11, 62: 11,   # Cowbell / misc perc
    39: 12,                            # Ghost / clap-ish
}

INSTRUMENT_TO_GM = {
    0: 36, 1: 38, 2: 42, 3: 46, 4: 41, 5: 45, 6: 48,
    7: 49, 8: 51, 9: 53, 10: 52, 11: 56, 12: 37,
}

DRUM_CLASS_NAMES = {
    0: "Kick", 1: "Snare", 2: "HH-Closed", 3: "HH-Open", 4: "LowTom",
    5: "MidTom", 6: "HighTom", 7: "Crash", 8: "Ride", 9: "RideBell",
    10: "China", 11: "Perc", 12: "Ghost/Rim",
}

# ── Time-signature vocabulary ──────────────────────────────────────────────────
# Maps (numerator, denominator) -> categorical id. The model embeds this so it can
# apply meter-specific accent/feel - the "one-and-two-and" of 4/4 vs the compound
# "ONE-two-three-FOUR-five-six" lilt of 6/8. Unknown meters fall back to "other".
TIME_SIGNATURES = [
    (4, 4), (3, 4), (2, 4), (6, 8), (12, 8), (9, 8), (5, 4), (7, 8), (6, 4), (2, 2),
]
TIMESIG_TO_ID = {ts: i + 1 for i, ts in enumerate(TIME_SIGNATURES)}  # 0 reserved = unknown/other
NUM_TIMESIGS = len(TIME_SIGNATURES) + 1  # +1 for the "other" bucket at id 0

def timesig_id(num: int, den: int) -> int:
    return TIMESIG_TO_ID.get((int(num), int(den)), 0)

def beats_per_bar_of(num: int, den: int) -> float:
    """Quarter-note beats per bar for a given time signature (for bar-length math)."""
    return float(num) * (4.0 / float(den))


# =============================================================================
# DRUM EVENT + TOKENIZER
# =============================================================================

@dataclass
class DrumEvent:
    """One drum hit in musical terms. raw_pitch preserves exact articulation."""
    instrument:   int    # class 0..num_instruments-1 (model input feature)
    bar:          int
    grid_step:    int    # 0..grid_steps_per_bar-1
    velocity_bin: int
    offset_ticks: int    # signed micro-timing deviation from grid
    raw_velocity: int
    raw_tick:     int
    raw_pitch:    int    # DESIGN: exact original GM note -> faithful reconstruction
    ts_id:        int = 0    # time-signature id (see TIMESIG_TO_ID), 0 = unknown
    beat_slot:    int = 0    # metric position WITHIN this note's real measure, 0..beat_slots-1
    is_flam:      int = 0    # 1 if this is the grace/second hit of a flam/drag
    flam_gap_ms:  float = 0.0  # ms after its flam partner (only meaningful if is_flam)
    tempo_bpm:    float = 120.0  # tempo at this note (for the model's tempo feature)


class Tokenizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def velocity_to_bin(self, v: int) -> int:
        # DESIGN: at 128 bins this is the identity (bin == velocity) -> lossless.
        # At coarser bin counts it groups proportionally.
        b = (int(v) * self.cfg.velocity_bins) // 128
        return min(b, self.cfg.velocity_bins - 1)

    def bin_to_velocity(self, b: int) -> int:
        # Inverse of velocity_to_bin. At 128 bins returns the velocity unchanged;
        # at coarser counts returns the bin's centre value.
        if self.cfg.velocity_bins >= 128:
            return int(np.clip(b, 0, 127))
        return int(np.clip(round((b + 0.5) * 128 / self.cfg.velocity_bins), 0, 127))

    def offset_to_bin(self, o: int) -> int:
        return int(np.clip(o + self.cfg.max_offset_ticks, 0, self.cfg.offset_bins - 1))

    def bin_to_offset(self, b: int) -> int:
        return int(b) - self.cfg.max_offset_ticks

    def _ioi(self, events: List[DrumEvent]) -> np.ndarray:
        """
        Inter-onset interval in grid steps for each hit (idea borrowed from the
        encoder-only solution). IOI[i] = grid steps since previous hit's onset.
        Tempo/phrase-agnostic; complements absolute position nicely.
        """
        n = len(events)
        ioi = np.zeros(n, dtype=np.int32)
        prev = None
        for i, e in enumerate(events):
            gpos = e.bar * self.cfg.grid_steps_per_bar + e.grid_step
            ioi[i] = 0 if prev is None else int(np.clip(gpos - prev, 0, self.cfg.max_ioi_steps))
            prev = gpos
        return ioi

    def _ioi_same(self, events: List[DrumEvent]) -> np.ndarray:
        """
        Per-INSTRUMENT inter-onset interval: grid steps since the SAME instrument
        last played. DESIGN: this is what makes "consecutive same notes" legible -
        a run of steady 16th hi-hats shows a constant same-instrument IOI even when
        kicks/snares are interleaved, so the model can cleanly recognise repetition
        (repetitive crashes, hat rhythms, tom/snare runs) and shape it. 0 = first
        hit of that instrument in the window.
        """
        n = len(events)
        ioi = np.zeros(n, dtype=np.int32)
        last = {}  # instrument -> last grid pos
        for i, e in enumerate(events):
            gpos = e.bar * self.cfg.grid_steps_per_bar + e.grid_step
            prev = last.get(e.instrument)
            ioi[i] = 0 if prev is None else int(np.clip(gpos - prev, 0, self.cfg.max_ioi_steps))
            last[e.instrument] = gpos
        return ioi

    def _metric_strength(self, events: List[DrumEvent]) -> np.ndarray:
        """
        Explicit metric-strength category per note, from its position in the measure.
        DESIGN: drummers accent strong subdivisions and ease weak ones - downbeat >
        beat > 8th > 16th > finer. This is a modular pattern that's hard for the model
        to discover from raw metric position, so we hand it over directly as a small
        ordinal feature (0 = strongest ... higher = weaker). We classify each position
        by the COARSEST musical grid it lands on: a note on the downbeat is also on the
        beat/8th/16th grids, but downbeat is the strongest label that applies.
        """
        n = len(events)
        strength = np.zeros(n, dtype=np.int32)
        S = self.cfg.beat_slots
        # grid fractions from coarsest->finest, paired with their strength label
        grids = [(1.0, 0), (0.5, 1), (0.25, 2), (0.125, 3), (0.0625, 4)]
        tol = 0.5 / S  # half a slot, in fraction units
        for i, e in enumerate(events):
            f = e.beat_slot / max(1, S)     # 0..1 through the measure
            lab = 5                          # default: finer than 16th / off-grid
            for frac, level in grids:
                # does f land on this grid? (nearest multiple of `frac` within tol)
                if abs(f - round(f / frac) * frac) < tol:
                    lab = level
                    break                    # coarsest match wins -> strongest label
            strength[i] = lab
        return strength

    def events_to_arrays(self, events: List[DrumEvent]) -> Dict[str, np.ndarray]:
        n = len(events)
        out = {
            'instruments': np.zeros(n, dtype=np.int32),
            'positions':   np.zeros(n, dtype=np.int32),
            'iois':        self._ioi(events),
            'iois_same':   self._ioi_same(events),         # per-instrument IOI
            'metric_str':  self._metric_strength(events),  # subdivision strength category
            'velocities':  np.zeros(n, dtype=np.int32),     # bin index (classification)
            'offsets':     np.zeros(n, dtype=np.int32),     # bin index (classification)
            # continuous targets (for regression mode): velocity in [0,1], offset in [-1,1]
            'vel_cont':    np.zeros(n, dtype=np.float32),
            'off_cont':    np.zeros(n, dtype=np.float32),
            # time-signature awareness
            'ts_ids':      np.zeros(n, dtype=np.int32),
            'beat_slots':  np.zeros(n, dtype=np.int32),
            # tempo (normalized) and flam features
            'tempo_norm':  np.zeros(n, dtype=np.float32),
            'is_flam':     np.zeros(n, dtype=np.int32),
            'flam_gap':    np.zeros(n, dtype=np.int32),   # quantized gap bin
        }
        mot = self.cfg.max_offset_ticks
        tmin, tmax = self.cfg.tempo_min_bpm, self.cfg.tempo_max_bpm
        for i, e in enumerate(events):
            out['instruments'][i] = e.instrument
            out['positions'][i]   = e.bar * self.cfg.grid_steps_per_bar + e.grid_step
            out['velocities'][i]  = e.velocity_bin
            out['offsets'][i]     = self.offset_to_bin(e.offset_ticks)
            out['vel_cont'][i]    = e.raw_velocity / 127.0
            out['off_cont'][i]    = float(np.clip(e.offset_ticks / mot, -1.0, 1.0))
            out['ts_ids'][i]      = e.ts_id
            out['beat_slots'][i]  = e.beat_slot
            out['tempo_norm'][i]  = float(np.clip((e.tempo_bpm - tmin) / max(1e-6, tmax - tmin), 0.0, 1.0))
            out['is_flam'][i]     = e.is_flam
            # quantize flam gap (0..flam_window_ms) into flam_gap_bins
            g = int(np.clip(round(e.flam_gap_ms / max(1e-6, self.cfg.flam_window_ms) * (self.cfg.flam_gap_bins - 1)),
                            0, self.cfg.flam_gap_bins - 1)) if e.is_flam else 0
            out['flam_gap'][i]    = g
        return out

    def bin_to_velocity_f(self, b: float) -> float:
        """Float version of bin->velocity, preserving sub-bin precision from
        expected-value decoding. Rounded to an int only at the very end."""
        if self.cfg.velocity_bins >= 128:
            return float(np.clip(b, 0, 127))
        return float(np.clip((b + 0.5) * 128 / self.cfg.velocity_bins, 0, 127))

    def arrays_to_events(self, arrays, source_events: List[DrumEvent]) -> List[DrumEvent]:
        result = []
        for i, src in enumerate(source_events):
            # velocities/offsets may be FLOAT (expected-value decode) - keep the
            # precision through the conversion, round to int only for the MIDI note.
            vb = float(np.clip(arrays['velocities'][i], 0, self.cfg.velocity_bins - 1))
            ob = float(arrays['offsets'][i])
            vel = int(round(self.bin_to_velocity_f(vb)))
            off = int(round(ob)) - self.cfg.max_offset_ticks
            result.append(DrumEvent(
                instrument=src.instrument, bar=src.bar, grid_step=src.grid_step,
                velocity_bin=int(round(vb)), offset_ticks=off,
                raw_velocity=vel, raw_tick=src.raw_tick, raw_pitch=src.raw_pitch,
                ts_id=src.ts_id, beat_slot=src.beat_slot,
                is_flam=src.is_flam, flam_gap_ms=src.flam_gap_ms, tempo_bpm=src.tempo_bpm,
            ))
        return result

    def quantize_events(self, events: List[DrumEvent]) -> List[DrumEvent]:
        """
        Robotic version: zero offset + coarsened velocity. This is the INPUT.
        DESIGN: coarsen in MIDI-velocity units (~16), NOT in bins, so the amount of
        de-humanization is the same whether we run 32 or 128 bins. Programmed drums
        typically sit on a few coarse velocity levels; this simulates that.

        AVERAGE-PRESERVING: a real dry/programmed groove keeps the OVERALL loudness
        of its genre - a dry metal part is still loud, a dry jazz part still soft.
        Naive independent rounding can drift the mean (rounding bias, clipping at the
        rails). So after coarsening we correct the whole groove by a single offset so
        the de-humanized MEAN matches the original mean (to the nearest velocity),
        then re-clip. This keeps intensity/genre intact as an input signal while still
        flattening the fine per-note dynamics the model is meant to re-learn.
        """
        coarse_step_vel = 16
        if not events:
            return []
        orig_vels = np.array([self.bin_to_velocity(e.velocity_bin) for e in events], dtype=float)
        coarse = np.clip(np.round(orig_vels / coarse_step_vel) * coarse_step_vel, 1, 127)
        # single global correction so the mean direction (intensity/genre) is retained
        mean_shift = orig_vels.mean() - coarse.mean()
        coarse = np.clip(coarse + mean_shift, 1, 127)

        for_out = []
        for e, vc in zip(events, coarse):
            v_coarse = int(round(vc))
            cb = self.velocity_to_bin(v_coarse)                      # -> bins
            for_out.append(DrumEvent(
                instrument=e.instrument, bar=e.bar, grid_step=e.grid_step,
                velocity_bin=cb, offset_ticks=0,
                raw_velocity=v_coarse, raw_tick=e.raw_tick, raw_pitch=e.raw_pitch,
                ts_id=e.ts_id, beat_slot=e.beat_slot,
                is_flam=e.is_flam, flam_gap_ms=e.flam_gap_ms, tempo_bpm=e.tempo_bpm,
            ))
        return for_out


# =============================================================================
# MIDI FILE I/O
# =============================================================================

def _build_measure_map(midi, cfg: Config):
    """
    Build a list of measures from the MIDI's time-signature changes:
      [(start_beat, end_beat, num, den, ts_id), ...] in quarter-note-beat units.
    Handles files with no time-sig (defaults to cfg.beats_per_bar/4) and files
    with multiple changes (e.g. a 4/4 song with a 6/8 bridge).
    """
    # total length in quarter-note beats
    end_time = midi.get_end_time()
    # tempo for sec->beat conversion
    tempo = 120.0
    try:
        tc = midi.get_tempo_changes()
        if len(tc[1]) > 0:
            tempo = float(tc[1][0])
    except Exception:
        pass
    total_beats = end_time * (tempo / 60.0) + 8  # pad a little

    changes = sorted(getattr(midi, 'time_signature_changes', []), key=lambda t: t.time)
    # convert change times (seconds) -> beats
    segs = []
    for ts in changes:
        segs.append((ts.time * (tempo / 60.0), ts.numerator, ts.denominator))
    if not segs or segs[0][0] > 0.01:
        segs.insert(0, (0.0, cfg.beats_per_bar, 4))  # default meter from start

    measures = []
    for i, (start_beat, num, den) in enumerate(segs):
        seg_end = segs[i + 1][0] if i + 1 < len(segs) else total_beats
        bpb = beats_per_bar_of(num, den)
        b = start_beat
        # guard against pathological zero-length meters
        if bpb <= 0:
            continue
        while b < seg_end - 1e-6:
            measures.append((b, b + bpb, num, den, timesig_id(num, den)))
            b += bpb
    return measures, tempo


def _locate_measure(measures, beat_pos):
    """Binary-ish search: return (index, (start,end,num,den,ts_id)) containing beat_pos."""
    # measures are contiguous & ordered; linear scan is fine for typical counts,
    # but do a quick bisect on starts for long files.
    lo, hi = 0, len(measures) - 1
    if hi < 0:
        return -1, None
    while lo < hi:
        mid = (lo + hi) // 2
        if measures[mid][1] <= beat_pos:
            lo = mid + 1
        else:
            hi = mid
    return lo, measures[lo]


def load_midi_events(path: str, cfg: Config):
    """Returns (events_or_None, reason). reason is None on success, otherwise a short
    code explaining why the file yielded no usable events - see build_cache's summary,
    which tallies these across the whole run so mass skips are diagnosable instead of
    collapsing into one opaque 'failed' counter."""
    if not HAS_PRETTY_MIDI:
        raise ImportError("pretty_midi required for MIDI I/O")
    try:
        return _load_midi_events_impl(path, cfg)
    except Exception as exc:
        # One malformed file must never abort a whole cache build. Report which file
        # and why (with the exact line), then signal "skip" by returning None. Full
        # per-file printing (not just tallying) is kept here because parse exceptions
        # are rare/unexpected - unlike the routine 'no drum track' skip below.
        _report_error(f"parsing MIDI file '{os.path.basename(path)}'", exc)
        return None, f"exception:{type(exc).__name__}"


_DRUM_NAME_RE = re.compile(r'drum|kit|perc|groove|beat', re.IGNORECASE)
_NON_DRUM_NAME_RE = re.compile(
    r'bass|guitar|piano|keys?|synth|vocal|lead|pad|string|brass|horn|organ|choir', re.IGNORECASE)


def _get_drum_notes(midi):
    """
    Find the notes to treat as drums, with a fallback for files that don't follow
    the MIDI-channel-10 convention. DESIGN: this cache builder is meant to run on
    purchased/downloaded drum-GROOVE MIDI packs (Toontrack/EZdrummer/BFD/Groove
    Monkee etc, see README) - single-purpose loop files, not full-band songs. Those
    packs routinely park their one and only track on whatever channel the exporting
    DAW defaulted to (often 1, not 10), so pretty_midi's is_drum flag misses them
    entirely even though the file is unambiguously "all drums". We exploit that
    single-purpose-file structure as the safety net: only fall back to treating
    non-flagged notes as drums when there's no real ambiguity about what else they
    could be (exactly one track has notes, or the track is explicitly named as
    drums) - a multi-track file with an unnamed/ambiguous second part is left alone
    rather than risk folding a bass or melodic line into the "drum" pitches.
    Returns (notes, source) where source explains which path was used.
    """
    drum_tracks = [t for t in midi.instruments if t.is_drum]
    if drum_tracks:
        return [n for t in drum_tracks for n in t.notes], 'channel10'

    non_empty = [t for t in midi.instruments if t.notes]
    if not non_empty:
        return [], 'no_notes'
    if len(non_empty) == 1:
        # only one part in the whole file - in a drum-loop-pack library this can
        # only be the drum part, regardless of what channel it was exported on.
        return list(non_empty[0].notes), 'single_track_fallback'

    named_drum = [t for t in non_empty
                  if _DRUM_NAME_RE.search(t.name or '') and not _NON_DRUM_NAME_RE.search(t.name or '')]
    if named_drum:
        return [n for t in named_drum for n in t.notes], 'name_match_fallback'

    # multiple tracks, none flagged or named as drums - too ambiguous to guess
    # (could be a full-band "_songs/" export); leave it for the caller to skip.
    return [], 'ambiguous_multi_track'


def _load_midi_events_impl(path: str, cfg: Config):
    midi = pretty_midi.PrettyMIDI(path)
    notes, note_source = _get_drum_notes(midi)
    if not notes:
        return None, ("no_notes_at_all" if note_source == 'no_notes' else note_source)
    notes.sort(key=lambda n: n.start)

    measures, tempo = _build_measure_map(midi, cfg)
    if not measures:
        return None, "no_measure_map"

    tok = Tokenizer(cfg)
    events: List[DrumEvent] = []
    unmapped_pitches = 0   # notes whose pitch isn't in GM_DRUM_MAP (e.g. extended VST articulation maps)
    last_time_by_inst = {}   # instrument -> last note start time (seconds), for flam detection
    flam_window_s = cfg.flam_window_ms / 1000.0
    for note in notes:
        inst = GM_DRUM_MAP.get(note.pitch, -1)
        if inst < 0:
            unmapped_pitches += 1
            continue
        # absolute tick on the internal fixed grid (for offset/IOI math, unchanged)
        tick = int(note.start * (tempo / 60.0) * cfg.ticks_per_beat)
        grid_tick = round(tick / cfg.ticks_per_grid) * cfg.ticks_per_grid
        offset = int(np.clip(tick - grid_tick, -cfg.max_offset_ticks, cfg.max_offset_ticks))

        # flam / grace-note detection: same instrument within the flam window
        is_flam, flam_gap_ms = 0, 0.0
        prev_t = last_time_by_inst.get(inst)
        if prev_t is not None:
            gap_s = note.start - prev_t
            if 0.0 < gap_s <= flam_window_s:
                is_flam = 1
                flam_gap_ms = gap_s * 1000.0
        last_time_by_inst[inst] = note.start

        # metric position from the REAL measure map (time-signature aware)
        beat_pos = grid_tick / cfg.ticks_per_beat          # quarter-note beats
        m_idx, m = _locate_measure(measures, beat_pos)
        if m is None:
            continue
        m_start, m_end, num, den, ts = m
        frac = (beat_pos - m_start) / max(1e-6, (m_end - m_start))   # 0..1 within bar
        beat_slot = int(np.clip(round(frac * cfg.beat_slots), 0, cfg.beat_slots - 1))
        bar = m_idx
        if bar >= cfg.max_bars:
            # keep the note but cap bar index for the (still fixed) bar embedding
            bar = cfg.max_bars - 1

        # keep grid_step for reconstruction on the internal fixed grid
        step = int((grid_tick % (cfg.ticks_per_beat * cfg.beats_per_bar)) // cfg.ticks_per_grid)

        events.append(DrumEvent(
            instrument=inst, bar=int(bar), grid_step=step,
            velocity_bin=tok.velocity_to_bin(note.velocity),
            offset_ticks=offset, raw_velocity=int(note.velocity),
            raw_tick=tick, raw_pitch=int(note.pitch),
            ts_id=ts, beat_slot=beat_slot,
            is_flam=is_flam, flam_gap_ms=flam_gap_ms, tempo_bpm=tempo,
        ))
    if events:
        return events, note_source   # note_source is 'channel10' on the normal path,
                                      # or the fallback used to recover a non-channel-10 file
    # every drum-channel note existed but none of them mapped to a known GM pitch -
    # distinct from "no drum channel at all" so an incomplete GM_DRUM_MAP is diagnosable.
    if unmapped_pitches > 0:
        return None, "all_pitches_unmapped"
    return None, "no_notes_after_measure_mapping"


def events_to_midi(events: List[DrumEvent], out_path: str, cfg: Config, tempo: float = 120.0):
    if not HAS_PRETTY_MIDI:
        raise ImportError("pretty_midi required for MIDI I/O")
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    track = pretty_midi.Instrument(program=0, is_drum=True, name="Humanized Drums")
    sec_per_tick = 60.0 / (tempo * cfg.ticks_per_beat)
    for e in events:
        # DESIGN: use the EXACT original pitch, not a class->pitch guess. This keeps
        # the specific cymbal / articulation the user programmed.
        pitch = e.raw_pitch if e.raw_pitch else INSTRUMENT_TO_GM.get(e.instrument, 38)
        grid_tick = (e.bar * cfg.grid_steps_per_bar + e.grid_step) * cfg.ticks_per_grid
        start = max(0.0, (grid_tick + e.offset_ticks) * sec_per_tick)
        # FLAM PRESERVATION: a grace note quantizes onto its partner's grid step, which
        # would collapse the flam. Restore the micro-gap (in seconds) so the flam
        # survives reconstruction. The gap itself can be humanized (see inference).
        if getattr(e, 'is_flam', 0) and getattr(e, 'flam_gap_ms', 0.0) > 0.0:
            start = max(0.0, start + e.flam_gap_ms / 1000.0)
        track.notes.append(pretty_midi.Note(
            velocity=max(1, min(127, e.raw_velocity)),
            pitch=pitch, start=start, end=start + (60.0 / tempo) * 0.1,
        ))
    track.notes.sort(key=lambda n: n.start)
    midi.instruments.append(track)
    midi.write(out_path)
    print(f"  Wrote {len(events)} notes -> {out_path}")


# =============================================================================
# SYNTHETIC DATA  (lets the pipeline run with zero real MIDI files)
# =============================================================================

def generate_synthetic_events(cfg: Config, num_samples: int = 2000, seed: int = GLOBAL_SEED):
    rng = random.Random(seed)
    tok = Tokenizer(cfg)
    spb = cfg.grid_steps_per_bar
    styles = ['rock', 'funk', 'metal', 'jazz']
    samples = []

    def mk(inst, bar, step, vel, off):
        step = max(0, min(spb - 1, step))
        off = int(np.clip(off, -cfg.max_offset_ticks, cfg.max_offset_ticks))
        # synthetic data is 4/4; beat_slot = metric position within the bar
        bslot = int(round((step / spb) * cfg.beat_slots)) % cfg.beat_slots
        return DrumEvent(inst, bar, step, tok.velocity_to_bin(vel), off,
                         vel, 0, INSTRUMENT_TO_GM.get(inst, 38),
                         ts_id=timesig_id(4, 4), beat_slot=bslot)

    for _ in range(num_samples):
        style = rng.choice(styles)
        evs: List[DrumEvent] = []
        for bar in range(2):
            for step in range(spb):
                q  = (step % spb) == 0 or (step % spb) == spb // 2
                bt = (step % (spb // 4)) == 0
                e8 = (step % (spb // 8)) == 0
                e16 = (step % (spb // 16)) == 0 if spb >= 16 else False
                backbeat = (step % spb) in (spb // 4, 3 * spb // 4)

                if style == 'rock':
                    if q and rng.random() > 0.1:
                        evs.append(mk(0, bar, step, rng.randint(100, 127), rng.randint(-3, 3)))
                    if backbeat:
                        evs.append(mk(1, bar, step, rng.randint(90, 120), rng.randint(-4, 2)))
                    elif e16 and rng.random() > 0.85:
                        evs.append(mk(12, bar, step, rng.randint(30, 60), rng.randint(-2, 2)))
                    if e8 and rng.random() > 0.05:
                        evs.append(mk(2, bar, step, rng.randint(60, 100), rng.randint(-2, 2)))
                    if step == 0 and bar == 0 and rng.random() > 0.6:
                        evs.append(mk(7, bar, step, rng.randint(80, 110), 0))
                elif style == 'funk':
                    if bt and rng.random() > 0.35:
                        evs.append(mk(0, bar, step, rng.randint(100, 127), rng.randint(-4, 4)))
                    if backbeat:
                        evs.append(mk(1, bar, step, rng.randint(100, 127), rng.randint(-3, 3)))
                    elif rng.random() > 0.6:
                        evs.append(mk(12, bar, step, rng.randint(25, 55), rng.randint(-3, 3)))
                    if e8:
                        evs.append(mk(2, bar, step, rng.randint(50, 90), rng.randint(-2, 2)))
                    if e8 and rng.random() > 0.85:
                        evs.append(mk(3, bar, step, rng.randint(70, 100), 0))
                elif style == 'metal':
                    if e8:
                        evs.append(mk(0, bar, step, rng.randint(110, 127), rng.randint(-2, 2)))
                    if bt:
                        evs.append(mk(1, bar, step, rng.randint(90, 120), rng.randint(-2, 2)))
                    if (step % 2) == 0:
                        evs.append(mk(2, bar, step, rng.randint(80, 110), rng.randint(-1, 1)))
                    if step == 0 and bar == 0 and rng.random() > 0.7:
                        evs.append(mk(7, bar, step, rng.randint(100, 127), 0))
                elif style == 'jazz':
                    if q:
                        evs.append(mk(0, bar, step, rng.randint(70, 100), rng.randint(-3, 3)))
                    if backbeat:
                        evs.append(mk(1, bar, step, rng.randint(60, 90), rng.randint(-4, 2)))
                    if bt:
                        evs.append(mk(8, bar, step, rng.randint(40, 70), rng.randint(-2, 2)))
                    if e8 and rng.random() > 0.3:
                        evs.append(mk(2, bar, step, rng.randint(30, 60), rng.randint(-2, 2)))
        evs.sort(key=lambda e: (e.bar, e.grid_step, e.instrument))
        if len(evs) >= 8:
            samples.append(evs)
    return samples


# =============================================================================
# CACHE
# =============================================================================

def _split_events_into_sections(events, cfg, section_bars=16, hop_bars=8):
    """
    Split a long groove/song into overlapping section-sized windows aligned to BAR
    boundaries. DESIGN: whole songs in the training data (verse/chorus/bridge) are
    better learned as coherent sections than as one giant sample that gets randomly
    cropped mid-phrase. Overlap (hop < section) means section transitions are still
    seen. Each window keeps ~1 section of both-side context around its notes.
    Splitting on bar lines (not note index) keeps musical alignment intact.
    """
    if not events:
        return []
    max_bar = max(e.bar for e in events)
    # short files -> single sample, unchanged
    if max_bar < section_bars * 1.5:
        return [events]
    sections = []
    b = 0
    while b <= max_bar:
        lo, hi = b, b + section_bars
        seg = [e for e in events if lo <= e.bar < hi]
        if len(seg) >= 8:
            # re-base bar indices to start at 0 so positions fit the embeddings
            seg = [DrumEvent(e.instrument, e.bar - lo, e.grid_step, e.velocity_bin,
                             e.offset_ticks, e.raw_velocity, e.raw_tick, e.raw_pitch,
                             e.ts_id, e.beat_slot, e.is_flam, e.flam_gap_ms, e.tempo_bpm)
                   for e in seg]
            sections.append(seg)
        if hi > max_bar:
            break
        b += hop_bars
    return sections or [events]


def measure_expressiveness(events, cfg: Config):
    """
    Measure how HUMANIZED a groove already is, from its target (human) performance.
    Returns a dict of stats + a boolean 'keep'. DESIGN: a sample is only rejected
    when it is flat on BOTH velocity and timing - a genuinely robotic/quantized
    pattern with nothing to teach. Rich dynamics OR rich micro-timing -> keep it,
    because it's still a useful target for that dimension.
    """
    if not events:
        return {'keep': False, 'reason': 'empty',
                'vel_std': 0.0, 'vel_range': 0.0, 'off_std': 0.0, 'off_range': 0.0}
    vels = np.array([e.raw_velocity for e in events], dtype=float)
    offs = np.array([e.offset_ticks for e in events], dtype=float)
    vel_std   = float(vels.std())
    vel_range = float(vels.max() - vels.min())
    off_std   = float(offs.std())
    off_range = float(offs.max() - offs.min())

    dynamics_flat = (vel_std < cfg.min_velocity_std) and (vel_range < cfg.min_velocity_range)
    timing_flat   = (off_std < cfg.min_offset_std) and (off_range < cfg.min_offset_range)
    keep = not (dynamics_flat and timing_flat)
    reason = 'ok'
    if not keep:
        reason = 'flat-both (robotic: no dynamics AND on-grid)'
    return {'keep': keep, 'reason': reason, 'vel_std': vel_std, 'vel_range': vel_range,
            'off_std': off_std, 'off_range': off_range,
            'dynamics_flat': dynamics_flat, 'timing_flat': timing_flat}


def _process_one_file(args) -> Optional[Dict]:
    # args may be (path, cfg) or (path, cfg, split_opts)
    path, cfg = args[0], args[1]
    split = args[2] if len(args) > 2 else {}
    try:
        events, note_source = load_midi_events(path, cfg)
        if events is None:
            return {'_failed_reason': note_source}
        if len(events) < 10:
            return {'_failed_reason': 'too_few_events'}
        events.sort(key=lambda e: (e.bar, e.grid_step, e.instrument))
        tok = Tokenizer(cfg)

        if split.get('enabled', True):
            section_bars = split.get('section_bars', 16)
            hop_bars     = split.get('hop_bars', 8)
            sections = _split_events_into_sections(events, cfg, section_bars, hop_bars)
        else:
            sections = [events]

        out = []
        rejected = 0
        for seg in sections:
            if cfg.quality_filter:
                q = measure_expressiveness(seg, cfg)
                if not q['keep']:
                    rejected += 1
                    continue
            # song_id: every section cut from THIS file carries the same id, so the
            # train/val split can keep a song whole (see _split_by_song). All sections
            # here share one `path` string object, so pickle memoises it - the id costs
            # one copy per file, not per section.
            out.append({'target': tok.events_to_arrays(seg),
                        'input':  tok.events_to_arrays(tok.quantize_events(seg)),
                        'length': len(seg), 'events': seg, 'song_id': path})
        # attach a small stats marker on the first sample so the builder can tally
        if out:
            out[0]['_rejected_sections'] = rejected
            out[0]['_file_had_rejection'] = rejected > 0
            out[0]['_note_source'] = note_source
        elif rejected:
            # everything got filtered out; signal it distinctly (not a parse failure)
            return {'_all_rejected': rejected}
        return out
    except Exception as exc:
        # runs in a worker process - report which file broke and skip it rather than
        # letting the whole pool crash with an opaque error.
        _report_error(f"processing MIDI file '{os.path.basename(path)}' in cache builder", exc)
        return {'_failed_reason': f'exception:{type(exc).__name__}'}


def build_cache(data_dir: str, cache_path: str, cfg: Config, num_workers: int = 8,
                split_songs: bool = True, section_bars: int = 16, hop_bars: int = 8):
    # DESIGN: glob.glob's recursive walk over a deep/large directory (a real library is
    # 190k+ files) is a single blocking call with no progress output - print before it
    # starts so a slow filesystem/network drive doesn't read as a hang before any output.
    print(f"Scanning {data_dir} for MIDI files...", flush=True)
    t_scan = time.time()
    paths = []
    for ext in ('mid', 'midi', 'MID', 'MIDI'):
        paths.extend(glob.glob(os.path.join(data_dir, '**', f'*.{ext}'), recursive=True))
    paths = sorted(set(paths))
    print(f"Found {len(paths)} MIDI files in {data_dir} ({time.time() - t_scan:.1f}s)")
    if split_songs:
        print(f"Song-splitting ON: long files -> {section_bars}-bar sections, hop {hop_bars}.")
    if cfg.quality_filter:
        print(f"Quality filter ON: rejecting flat/robotic sections "
              f"(vel_std<{cfg.min_velocity_std} & range<{cfg.min_velocity_range} "
              f"AND off_std<{cfg.min_offset_std} & range<{cfg.min_offset_range}).")
    split_opts = {'enabled': split_songs, 'section_bars': section_bars, 'hop_bars': hop_bars}
    samples, failed, rejected = [], 0, 0
    files_all_rejected = 0
    files_partially_rejected = 0
    failure_reasons = Counter()
    note_sources = Counter()
    work = [(p, cfg, split_opts) for p in paths]
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_process_one_file, w) for w in work]
        it = as_completed(futures)
        if HAS_TQDM:
            it = tqdm.tqdm(it, total=len(futures), desc="Parsing MIDI")
        for fut in it:
            r = fut.result()
            if isinstance(r, dict) and '_all_rejected' in r:
                rejected += r['_all_rejected']
                files_all_rejected += 1
            elif isinstance(r, dict) and '_failed_reason' in r:
                failed += 1
                failure_reasons[r['_failed_reason']] += 1
            elif r:                       # LIST of section-samples
                rejected += r[0].pop('_rejected_sections', 0)
                if r[0].pop('_file_had_rejection', False):
                    files_partially_rejected += 1
                note_sources[r[0].pop('_note_source', 'channel10')] += 1
                samples.extend(r)
            else:
                failed += 1
                failure_reasons['empty_after_splitting'] += 1
    files_affected = files_all_rejected + files_partially_rejected
    kept_files = len(paths) - failed - files_all_rejected
    print(f"  ✓ {len(samples)} section-samples kept from ~{kept_files} files")
    print(f"  ✗ {failed} failed/empty  |  {rejected} sections rejected as flat/robotic"
          f"  ({files_all_rejected} files fully rejected)")
    fallback_recovered = note_sources.get('single_track_fallback', 0) + note_sources.get('name_match_fallback', 0)
    if fallback_recovered:
        print(f"  Recovered via non-channel-10 fallback: {fallback_recovered} files "
              f"({note_sources.get('single_track_fallback', 0)} single-track, "
              f"{note_sources.get('name_match_fallback', 0)} name-matched) "
              f"- see build_cache/_get_drum_notes for the heuristic; spot-check a few "
              f"if this number looks off.")
    if failed:
        print(f"  Why files failed (top reasons):")
        for reason, count in failure_reasons.most_common(8):
            pct = 100.0 * count / failed
            print(f"    {count:>8}  ({pct:5.1f}%)  {reason}")
        if failure_reasons.get('ambiguous_multi_track', 0) > failed * 0.1:
            print(f"    -> Most failures are files with multiple tracks where none is on the "
                  f"GM drum channel (10) or clearly named as drums - too ambiguous to guess "
                  f"which track is percussion vs. e.g. bass/melodic, so they're skipped rather "
                  f"than risk corrupting training data. Likely candidates: full-band '_songs/' "
                  f"exports mixed into the library, or drum tracks with unhelpful names.")
        if failure_reasons.get('all_pitches_unmapped', 0) > failed * 0.1:
            print(f"    -> Most failures have drum-channel notes whose pitches aren't in "
                  f"GM_DRUM_MAP (only covers standard GM pitches 35-62). If this library "
                  f"uses an extended/custom articulation map, GM_DRUM_MAP needs more entries.")
    if cfg.quality_filter:
        print(f"  Quality filter: {files_affected} of {len(paths)} files had at least one "
              f"section rejected ({files_all_rejected} entirely, "
              f"{files_partially_rejected} partially).")
    # provenance metadata so train() can report original-vs-split-vs-augmented counts
    meta = {
        'original_files_found':   len(paths),
        'original_files_kept':    kept_files,
        'files_failed':           failed,
        'files_fully_rejected':   files_all_rejected,
        'files_partially_rejected': files_partially_rejected,
        'sections_rejected':      rejected,
        'split_samples':          len(samples),   # section-samples after splitting+filtering
        'split_songs':            split_songs,
        'section_bars':           section_bars,
        'hop_bars':               hop_bars,
    }
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'cfg': asdict(cfg), 'samples': samples, 'meta': meta}, f, protocol=4)
    print(f"Cache saved -> {cache_path}")
    return samples


# =============================================================================
# DATASET
# =============================================================================

# Fixed seed for the validation set's de-humanization draw. Deliberately independent
# of GLOBAL_SEED: the val inputs should stay byte-identical across runs even when the
# training seed is changed to re-roll an experiment, or val loss stops being comparable
# between those runs. Changing this value re-rolls the whole val set - don't.
_VAL_DEHUM_SEED = 20240917


class DrumDataset(Dataset):
    def __init__(self, samples: List[Dict], cfg: Config, augment: bool = True):
        self.samples = samples
        self.cfg = cfg
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _window(self, arr, start, length):
        return {k: v[start:start + length] for k, v in arr.items()}

    def _rotate_bars(self, inp, tgt):
        """
        Drop the first K bars so the phrase starts later, then re-base positions to
        start at bar 0. K is drawn uniformly from 1..cap, where the cap scales with
        groove length so longer phrases can rotate further:
            ≥16 bars -> cap 8   |   ≥8 bars -> cap 4   |   ≥4 bars -> cap 2   |  else none
        DESIGN: K used to be FIXED at the cap, which gave every groove exactly two
        possible phrase starts for the whole run (rotated or not) - thin variety for
        something applied to 60% of samples. Randomizing K costs nothing and is just
        as safe: the invariance below holds for ANY whole-bar K, not just the cap.
        Applied identically to input and target. SAFE: the result is a real performance
        starting later; only 'positions' needs shifting - metric/IOI/velocity/offset are
        invariant to whole-bar rotation. Returns (inp, tgt) unchanged if too short.
        """
        spb = self.cfg.grid_steps_per_bar
        n = len(inp['positions'])
        if n == 0:
            return inp, tgt
        max_bar = int(inp['positions'].max()) // spb      # zero-based -> total bars = max_bar+1
        total_bars = max_bar + 1
        # cap the rotation by length, then pick anywhere up to it
        if total_bars >= 16:
            k_cap = 8
        elif total_bars >= 8:
            k_cap = 4
        elif total_bars >= 4:
            k_cap = 2
        else:
            return inp, tgt                   # too short to rotate
        k = random.randint(1, k_cap)
        cut = k * spb                         # grid positions before this are dropped
        keep = inp['positions'] >= cut        # boolean mask of notes to keep
        if keep.sum() < 8:                    # don't rotate into an almost-empty tail
            return inp, tgt
        new_inp = {key: val[keep] for key, val in inp.items()}
        new_tgt = {key: val[keep] for key, val in tgt.items()}
        # re-base positions so the kept region starts at bar 0
        new_inp['positions'] = new_inp['positions'] - cut
        new_tgt['positions'] = new_tgt['positions'] - cut
        return new_inp, new_tgt

    def _pad(self, arr, target_len):
        n = len(arr['instruments'])
        pad = target_len - n
        if pad <= 0:
            return arr
        out = {}
        for k, v in arr.items():
            # instruments pad with -1 (padding sentinel); everything else with 0.
            fill = -1 if k == 'instruments' else 0
            out[k] = np.concatenate([v, np.full(pad, fill, dtype=v.dtype)])
        return out

    # Lattice sizes (in MIDI velocity units) the 'lattice' de-humanization draws from.
    # 16 is what Tokenizer.quantize_events hardcodes; the rest coarsen from there.
    # DESIGN: deliberately no step below 16. A finer lattice is finer than the cached
    # de-humanization it replaces, so it hands the velocity head its own label back
    # (measured contour corr(input, target) = 0.97 at step 8 vs 0.89 at 16) while
    # representing no input a human would ever program - real programming repeats a
    # few chosen levels per kit piece, which is what the 'palette' style below does.
    _COARSE_STEPS = (16, 24, 32, 48)

    def _dehumanize_velocities(self, human_vel, instruments, rng=random):
        """
        Build the INPUT's velocity contour from the human take - the same job
        Tokenizer.quantize_events does at cache-build time, but with the strength AND
        style drawn fresh for every sample.

        WHY: the cached input is ONE fixed function of the target (round to a 16-MIDI
        lattice, then mean-correct), so |input_vel - target_vel| <= 8 on essentially
        every note. The velocity head can ride that: at training time the answer is
        most of the way written on its own input. Real programmed MIDI - a handful of
        flat levels per kit piece, chosen from intent rather than from a performance -
        carries no such hint, so whatever the model learned from it evaporates at
        inference. Redrawing the de-humanization each epoch removes the fixed lattice
        to key on. Nothing here touches the target, so no label is ever changed.

        Two styles, 50/50:
          'lattice' - the original round-to-N-MIDI, N random over _COARSE_STEPS.
          'palette' - how drums are actually programmed: each kit piece gets 1-3
                      velocity levels for the whole groove and every hit on that piece
                      snaps to the nearest one.
        Both are AVERAGE-PRESERVING, for the same reason quantize_events is: intensity
        is read off the input's mean further down, so letting it drift would corrupt
        the conditioning signal the user steers with at inference.

        rng: any random.Random-compatible source. Defaults to the global `random`
        (training); the validation path passes a per-sample seeded one so its draw is
        fixed forever - see __getitem__.

        human_vel: the take's velocities in MIDI units (not bins). Returns the same.
        """
        v = human_vel.astype(np.float64)
        if len(v) == 0:
            return v
        if rng.random() < 0.5:
            step = rng.choice(self._COARSE_STEPS)
            out = np.round(v / step) * step
        else:
            out = v.copy()
            for inst in np.unique(instruments):
                m = instruments == inst
                vi = v[m]
                k = 1 if len(vi) < 3 else rng.randint(1, 3)
                if k == 1:
                    out[m] = vi.mean()          # one flat level for this kit piece
                else:
                    # k levels spaced through THIS piece's own dynamic range (quantile
                    # centres), each hit snapped to the nearest - a programmer picks a
                    # couple of levels per piece and still puts accents where accents
                    # belong, so some ordering survives, just far coarser than +-8.
                    levels = np.quantile(vi, (np.arange(k) + 0.5) / k)
                    out[m] = levels[np.abs(vi[:, None] - levels[None, :]).argmin(1)]
        out = np.clip(out, 1, 127)
        return np.clip(out + (v.mean() - out.mean()), 1, 127)

    def _input_velocities(self, inp, tgt, rng):
        """Rebuild the input's velocity BINS from the human take in tgt, replacing the
        one frozen into the cache. inp and tgt are rotated and windowed together above,
        so they line up note-for-note here. Shared by the train and validation paths -
        they differ only in which rng they hand in."""
        vscale = 128.0 / self.cfg.velocity_bins
        deh = self._dehumanize_velocities(tgt['velocities'].astype(float) * vscale,
                                          inp['instruments'], rng)
        return np.clip((deh / vscale).round(), 0,
                       self.cfg.velocity_bins - 1).astype(inp['velocities'].dtype)

    def _augment_input(self, inp, tgt):
        inp = {k: v.copy() for k, v in inp.items()}
        n = len(inp['instruments'])
        inp['velocities'] = self._input_velocities(inp, tgt, random)
        jitter = np.random.randint(-2, 3, size=n).astype(np.int32)
        inp['velocities'] = np.clip(inp['velocities'] + jitter, 0, self.cfg.velocity_bins - 1)
        if random.random() < 0.5:
            mask = np.random.rand(n) < 0.03
            inp['velocities'][mask] = 0
        return inp

    def __getitem__(self, idx):
        s = self.samples[idx]
        inp, tgt, length = s['input'], s['target'], s['length']

        # DESIGN: validation must re-derive its input the same way training does - a val
        # set still fed the cached de-humanization would be scoring a task the model is
        # no longer being trained for (and an easier one, see _dehumanize_velocities).
        # But it must ALSO be identical every epoch and across runs: early stopping and
        # best.pt compare val numbers BETWEEN epochs, so a val set that redraws itself
        # each pass is comparing against a moving target and the winner is partly luck.
        # A per-sample RNG seeded from the index gives both - a fixed draw that still
        # differs from sample to sample. Training keeps the global (freshly reseeded
        # every epoch) `random`.
        rng = random if self.augment else random.Random(_VAL_DEHUM_SEED + idx)

        # Bar-rotation augmentation (before windowing so phrases start at varied bars)
        if (self.augment and self.cfg.bar_rotation
                and random.random() < self.cfg.bar_rotation_prob):
            inp, tgt = self._rotate_bars(inp, tgt)
            length = len(inp['positions'])

        L = self.cfg.max_seq_len
        if length > L:
            # NOTE: drawn from `rng`, so a long val groove windows to the SAME slice
            # every epoch. It used to use the global RNG in both modes, which quietly
            # made val loss jump around on long grooves for reasons unrelated to the
            # model - noise that early stopping and best.pt were reading as signal.
            start = rng.randint(0, length - L)
            inp = self._window(inp, start, L)
            tgt = self._window(tgt, start, L)
        if self.augment:
            inp = self._augment_input(inp, tgt)
        else:
            # Validation: the realistic de-humanization, fixed per sample. None of the
            # training-only noise (velocity jitter, dropout, intensity shift) applies.
            inp = {**inp, 'velocities': self._input_velocities(inp, tgt, rng)}

        vscale = 128.0 / self.cfg.velocity_bins

        # ── Intensity-conditioning augmentation (training only) ────────────────────
        # DESIGN: the model also sees per-note velocities, so it could ignore the
        # global intensity scalar and read loudness locally. To force intensity to be
        # a REAL lever, we sometimes shift the WHOLE groove's loudness by a random
        # amount - applying the SAME shift to input AND target - and let intensity be
        # computed from the shifted input. Now intensity genuinely predicts the output
        # level, so the model must learn to obey it. Preserves per-note dynamics
        # (it's a global shift, not a squash).
        if self.augment and random.random() < 0.5:
            iv = inp['velocities'].astype(float) * vscale
            tv = tgt['velocities'].astype(float) * vscale
            cur_mean = iv.mean() if len(iv) else 64.0
            # pick a new target mean anywhere in a wide musical range
            new_mean = np.random.uniform(30, 118)
            shift = new_mean - cur_mean
            iv = np.clip(iv + shift, 1, 127)
            tv = np.clip(tv + shift, 1, 127)
            inp = {**inp}
            inp['velocities'] = np.clip((iv / vscale).round(), 0, self.cfg.velocity_bins - 1).astype(np.int32)
            tgt = {**tgt}
            tgt['velocities'] = np.clip((tv / vscale).round(), 0, self.cfg.velocity_bins - 1).astype(np.int32)
            tgt['vel_cont']   = (tv / 127.0).astype(np.float32)

        # Intensity = mean INPUT velocity over the real (unpadded) notes, in [0,1].
        # DESIGN: computed on the de-humanized input (whose mean is preserved), so it
        # matches the loudness the humanized target should keep. This is the global
        # "how hard is this being played / what genre" signal - and the exact knob the
        # user can set at inference to request an intensity.
        real_vel = inp['velocities'].astype(float) * vscale        # bins -> MIDI vel
        intensity = float(np.clip(real_vel.mean() / 127.0, 0.0, 1.0)) if len(real_vel) else 0.5

        inp = self._pad(inp, L)
        tgt = self._pad(tgt, L)

        pad_mask = torch.tensor(inp['instruments'] < 0, dtype=torch.bool)
        ni = self.cfg.num_instruments
        spb = self.cfg.grid_steps_per_bar
        clip = lambda a: np.clip(a, 0, ni - 1)
        return {
            # inputs (quantized / robotic)
            'instruments': torch.tensor(clip(inp['instruments']), dtype=torch.long),
            'positions':   torch.tensor(inp['positions'], dtype=torch.long),
            'bars':        torch.tensor(inp['positions'] // spb, dtype=torch.long),
            'iois':        torch.tensor(np.clip(inp['iois'], 0, self.cfg.max_ioi_steps), dtype=torch.long),
            'iois_same':   torch.tensor(np.clip(inp['iois_same'], 0, self.cfg.max_ioi_steps), dtype=torch.long),
            'metric_str':  torch.tensor(np.clip(inp['metric_str'], 0, self.cfg.num_metric_levels - 1), dtype=torch.long),
            'in_velocities': torch.tensor(inp['velocities'], dtype=torch.long),
            'intensity':   torch.tensor(intensity, dtype=torch.float32),
            'ts_ids':      torch.tensor(np.clip(inp['ts_ids'], 0, NUM_TIMESIGS - 1), dtype=torch.long),
            'beat_slots':  torch.tensor(np.clip(inp['beat_slots'], 0, self.cfg.beat_slots - 1), dtype=torch.long),
            'tempo_norm':  torch.tensor(inp['tempo_norm'], dtype=torch.float32),
            'is_flam':     torch.tensor(np.clip(inp['is_flam'], 0, 1), dtype=torch.long),
            'flam_gap':    torch.tensor(np.clip(inp['flam_gap'], 0, self.cfg.flam_gap_bins - 1), dtype=torch.long),
            # targets - classification (bins)
            'tgt_velocities': torch.tensor(tgt['velocities'], dtype=torch.long),
            'tgt_offsets':    torch.tensor(tgt['offsets'], dtype=torch.long),
            # targets - regression (continuous)
            'tgt_vel_cont':   torch.tensor(tgt['vel_cont'], dtype=torch.float32),
            'tgt_off_cont':   torch.tensor(tgt['off_cont'], dtype=torch.float32),
            'pad_mask':       pad_mask,
        }


# DESIGN: how many bytes of dataset we're willing to copy into EACH DataLoader worker
# on a 'spawn' platform before deciding workers aren't worth it. Copying more than this
# per process is wasteful even when it happens to fit.
_SPAWN_WORKER_BUDGET = 1024 ** 3


def _estimate_pickle_bytes(samples, probe: int = 256) -> int:
    """
    Size of the pickle buffer this sample list would produce, extrapolated from a prefix.

    DESIGN: measures pickle.dumps() rather than summing ndarray.nbytes, because the raw
    payload is NOT the dominant cost. Each sample holds ~20 small arrays, so a large
    cache carries millions of tiny objects whose pickle framing roughly DOUBLES the array
    bytes - measured on a real cache: 1.26GB of ndarray payload serialized to a 2.5GB
    pickle. Summing nbytes therefore under-reports by ~2x and would let the guard below
    sail past exactly the caches that blow up. The pickle buffer is also the thing the
    parent actually has to materialize in RAM per worker, so it's the honest number.
    """
    if not samples:
        return 0
    head = samples[:probe]
    return int(len(pickle.dumps(head, protocol=4)) / len(head) * len(samples))


def _resolve_num_workers(samples, cfg: Config) -> int:
    """
    DESIGN: on 'spawn' platforms (Windows, macOS) every DataLoader worker receives a
    full PICKLED COPY of the Dataset, and the parent must materialize that entire
    pickle buffer in RAM to hand it over. train and val each spawn cfg.num_workers, so
    the real cost is roughly (2*num_workers + 1)x the sample list. On a large cache
    this MemoryErrors inside w.start() before the first batch is ever produced - a
    2.5GB / 166k-sample cache with num_workers=4 did exactly that on a 32GB machine,
    after a multi-minute load, with a traceback pointing at multiprocessing internals
    rather than at the actual cause.

    Linux 'fork' shares those pages copy-on-write and is unaffected, so this only
    downgrades where it actually matters.

    num_workers=0 loads in the main process: no pickling, no IPC, no per-worker copy.
    __getitem__ here is numpy slicing, so workers buy little for this model anyway.
    Set DBH_FORCE_WORKERS=1 to keep the configured count regardless.
    """
    workers = max(0, cfg.num_workers)
    if workers == 0 or not samples:
        return workers
    # get_start_method(allow_none=True) returns None until a context is actually fixed,
    # which is the normal state here - so infer the platform default rather than treating
    # "not yet decided" as spawn (that would wrongly downgrade fork platforms). Not using
    # allow_none=False because that FIXES the start method as a side effect.
    method = multiprocessing.get_start_method(allow_none=True)
    if method is None:
        method = 'fork' if sys.platform.startswith(('linux', 'freebsd')) else 'spawn'
    if method == 'fork':
        return workers
    if os.environ.get('DBH_FORCE_WORKERS'):
        return workers
    est = _estimate_pickle_bytes(samples)
    if est <= _SPAWN_WORKER_BUDGET:
        return workers
    projected = est * (2 * workers + 1)
    print(f"  NOTE: dataset is ~{est / 1024**3:.1f}GB and this platform starts DataLoader "
          f"workers by 'spawn', which copies it into every worker "
          f"(~{projected / 1024**3:.1f}GB for {workers} workers x train+val).")
    print(f"        Using num_workers=0 (main-process loading) instead - otherwise this "
          f"MemoryErrors before the first batch. Set DBH_FORCE_WORKERS=1 to override.")
    return 0


def _split_by_song(samples, val_split: float):
    """
    Train/val split that keeps every section of a song on ONE side.

    DESIGN: _split_events_into_sections cuts each song into section_bars windows every
    hop_bars, so at the default 16/8 adjacent sections SHARE HALF THEIR BARS. Splitting
    the flat sample list (what this used to do) therefore put a section in val while its
    overlapping neighbour - same song, same bars, same human performance - sat in train.
    At val_split=0.05 that happens to essentially every val section, so val loss was
    partly measuring memorisation. Grouping by song_id removes it at the source.

    Samples with no song_id (an older cache, or --synthetic) each become their own
    group, which reproduces the old per-sample behaviour exactly rather than failing.
    Returns (train, val, n_songs, song_aware).
    """
    groups, loose = {}, []
    for smp in samples:
        sid = smp.get('song_id')
        if sid is None:
            loose.append([smp])          # ungrouped: its own single-sample group
        else:
            groups.setdefault(sid, []).append(smp)
    song_aware = bool(groups)
    all_groups = list(groups.values()) + loose
    random.shuffle(all_groups)           # global RNG, seeded via seed_everything()
    n_val_target = max(1, int(len(samples) * val_split))
    val, train = [], []
    for g in all_groups:
        (val if len(val) < n_val_target else train).extend(g)
    # A corpus that is one long song (or one group holding nearly everything) would put
    # every sample on one side. Fall back rather than train on an empty split.
    if not train or not val:
        print(f"[warning] song-aware split left one side empty ({len(train)} train / "
              f"{len(val)} val) - the corpus is probably a single song. Falling back to "
              f"a per-sample split; val will overlap train if sections overlap.")
        shuffled = list(samples)
        random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_split))
        return shuffled[n_val:], shuffled[:n_val], len(all_groups), False
    return train, val, len(all_groups), song_aware


def make_loaders(samples, cfg: Config):
    train, val, n_songs, song_aware = _split_by_song(samples, cfg.val_split)
    if song_aware:
        print(f"Train: {len(train)}  Val: {len(val)}  "
              f"(split by SONG across {n_songs} songs - no song appears on both sides)")
    else:
        print(f"Train: {len(train)}  Val: {len(val)}  "
              f"(per-sample split: this cache has no song_id - rebuild it with "
              f"--mode cache to get a song-aware split)")
    workers = _resolve_num_workers(samples, cfg)
    # a dedicated, seeded torch.Generator for the shuffling DataLoader does, so the
    # train-batch ORDER is reproducible too, not just the RNGs used inside a sample.
    g = torch.Generator()
    g.manual_seed(GLOBAL_SEED)
    tl = DataLoader(DrumDataset(train, cfg, augment=True),
                    batch_size=cfg.batch_size, shuffle=True,
                    num_workers=workers, pin_memory=True, drop_last=True,
                    persistent_workers=(workers > 0),
                    worker_init_fn=_worker_init_fn if workers > 0 else None,
                    generator=g)
    vl = DataLoader(DrumDataset(val, cfg, augment=False),
                    batch_size=cfg.batch_size, shuffle=False,
                    num_workers=workers, pin_memory=True,
                    persistent_workers=(workers > 0),
                    worker_init_fn=_worker_init_fn if workers > 0 else None)
    return tl, vl


# =============================================================================
# MODEL  (encoder-only, parallel per-hit prediction)
# =============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, positions):
        return self.pe[positions.clamp(max=self.pe.size(0) - 1)]


class DrumEventEmbedding(nn.Module):
    """
    Embed one hit from (instrument, musical position, bar, IOI, input-velocity)
    plus a GLOBAL intensity signal (the groove's overall average velocity).
    DESIGN: musical position (where in the bar) AND IOI (steps since last hit) are
    BOTH fed in - the first grounds phrase/beat awareness, the second gives local
    rhythmic density, which drives feel (e.g. tight vs laid-back sixteenths).
    The intensity scalar tells the model the OVERALL loudness/genre context (e.g.
    ~0.9 = loud metal, ~0.4 = soft jazz), so it doesn't have to guess the level
    from local notes alone - and so it can be *set* explicitly at inference.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.instrument_emb = nn.Embedding(cfg.num_instruments + 1, d, padding_idx=0)
        self.velocity_emb   = nn.Embedding(cfg.velocity_bins + 1,   d, padding_idx=0)
        self.ioi_emb        = nn.Embedding(cfg.max_ioi_steps + 1,   d)
        self.ioi_same_emb   = nn.Embedding(cfg.max_ioi_steps + 1,   d)   # per-instrument IOI
        self.metric_emb     = nn.Embedding(cfg.num_metric_levels,   d)   # subdivision strength
        self.pos_emb        = SinusoidalPositionEmbedding(d, cfg.max_position + 1)
        self.bar_emb        = nn.Embedding(cfg.max_bars + 1, d)
        # time-signature awareness: which meter, and metric position within the measure
        self.ts_emb         = nn.Embedding(NUM_TIMESIGS, d)
        self.beat_slot_emb  = nn.Embedding(cfg.beat_slots, d)
        # flam / grace-note: is this a flam grace hit, and its (quantized) micro-gap
        self.flam_emb       = nn.Embedding(2, d)
        self.flam_gap_emb   = nn.Embedding(cfg.flam_gap_bins, d)
        self.proj           = nn.Linear(d * 11, d)
        # intensity: scalar in [0,1] -> d-dim conditioning vector, added to every token
        self.intensity_proj = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        # tempo: per-note scalar in [0,1] -> d-dim, added as a residual (feel scales w/ tempo)
        self.tempo_proj     = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.norm           = nn.LayerNorm(d)

    def forward(self, instruments, positions, bars, iois, velocities, intensity=None,
                ts_ids=None, beat_slots=None, iois_same=None, metric_str=None,
                tempo_norm=None, is_flam=None, flam_gap=None):
        ie = self.instrument_emb(instruments + 1)
        ve = self.velocity_emb(velocities + 1)
        oe = self.ioi_emb(iois.clamp(max=self.ioi_emb.num_embeddings - 1))
        pe = self.pos_emb(positions)
        be = self.bar_emb(bars.clamp(max=self.bar_emb.num_embeddings - 1))
        # time-signature features (default to id 0 / slot 0 if not supplied)
        if ts_ids is None:
            ts_ids = torch.zeros_like(instruments)
        if beat_slots is None:
            beat_slots = torch.zeros_like(instruments)
        if iois_same is None:
            iois_same = torch.zeros_like(instruments)
        if metric_str is None:
            metric_str = torch.zeros_like(instruments)
        if is_flam is None:
            is_flam = torch.zeros_like(instruments)
        if flam_gap is None:
            flam_gap = torch.zeros_like(instruments)
        tse = self.ts_emb(ts_ids.clamp(max=self.ts_emb.num_embeddings - 1))
        bse = self.beat_slot_emb(beat_slots.clamp(max=self.beat_slot_emb.num_embeddings - 1))
        ose = self.ioi_same_emb(iois_same.clamp(max=self.ioi_same_emb.num_embeddings - 1))
        mse = self.metric_emb(metric_str.clamp(max=self.metric_emb.num_embeddings - 1))
        fle = self.flam_emb(is_flam.clamp(0, 1))
        fge = self.flam_gap_emb(flam_gap.clamp(max=self.flam_gap_emb.num_embeddings - 1))
        x = self.proj(torch.cat([ie, ve, oe, pe, be, tse, bse, ose, mse, fle, fge], dim=-1))
        if intensity is not None:
            # intensity: (B,) or (B,1) -> (B,1,d) broadcast across time
            inten = intensity.view(-1, 1).float()
            cond = self.intensity_proj(inten).unsqueeze(1)      # (B,1,d)
            x = x + cond
        if tempo_norm is not None:
            # tempo is per-NOTE (B,T): project each and add as a residual
            tcond = self.tempo_proj(tempo_norm.unsqueeze(-1).float())   # (B,T,d)
            x = x + tcond
        return self.norm(x)


class RelativePositionBias(nn.Module):
    """
    #4 T5-style relative-position attention bias. Instead of adding absolute position
    to the input, we add a learned scalar bias to each attention score based on the
    (bucketed) signed distance between query and key positions. The model keys off
    HOW FAR APART two hits are, not their absolute index - which generalises to phrase
    lengths and start-positions it never saw. Buckets grow logarithmically so nearby
    distances get fine resolution and far ones share buckets.
    """
    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 256):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.rel_bias = nn.Embedding(2 * num_buckets + 1, num_heads)

    def _bucket(self, rel_pos):
        # map signed relative position -> bucket id in [0, 2*num_buckets]
        nb = self.num_buckets
        ret = torch.zeros_like(rel_pos)
        n = -rel_pos                                   # so sign is handled symmetrically
        # half the buckets for each sign
        max_exact = nb // 2
        is_small = n.abs() < max_exact
        # logarithmic for larger distances
        val_large = (max_exact + (torch.log(n.abs().float().clamp(min=1) / max_exact) /
                     math.log(self.max_distance / max_exact) * (nb - max_exact)).long())
        val_large = torch.clamp(val_large, max=nb - 1)
        bucket = torch.where(is_small, n.abs(), val_large)
        bucket = bucket * torch.sign(n).long()         # restore sign
        return bucket + nb                             # shift to [0, 2*nb]

    def forward(self, T, device):
        pos = torch.arange(T, device=device)
        rel = pos[None, :] - pos[:, None]              # (T,T) signed distances
        buckets = self._bucket(rel).clamp(0, 2 * self.num_buckets)
        bias = self.rel_bias(buckets)                  # (T,T,H)
        return bias.permute(2, 0, 1)                   # (H,T,T)


class RelPosEncoderLayer(nn.Module):
    """
    A pre-norm transformer encoder layer with multi-head self-attention that accepts
    an additive (H,T,T) relative-position bias on the attention logits. Mirrors
    nn.TransformerEncoderLayer(norm_first=True) but exposes the attn-bias hook.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.nhead = nhead
        self.dh = d_model // nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, rel_bias=None, key_padding_mask=None):
        B, T, D = x.shape
        h = self.n1(x)
        q = self.q(h).view(B, T, self.nhead, self.dh).transpose(1, 2)   # (B,H,T,dh)
        k = self.k(h).view(B, T, self.nhead, self.dh).transpose(1, 2)
        v = self.v(h).view(B, T, self.nhead, self.dh).transpose(1, 2)
        # DESIGN: fused scaled_dot_product_attention instead of a hand-rolled
        # QK^T/softmax/@V - measured 1.74x faster and ~45% less peak memory at this
        # model's actual shapes (B=32,H=8,T=256,dh=32, base preset) on a GTX 1650,
        # numerically verified equivalent to the old path (max output diff 1.8e-6).
        # attn_mask, as a FLOAT tensor, gets ADDED to the raw scores before softmax -
        # exactly the additive rel_bias + masked-fill(-inf) padding behavior this
        # replaces, just fused into one kernel instead of four separate ops.
        # dropout_p is passed explicitly because SDPA has no nn.Module state, so
        # (unlike nn.Dropout) it won't auto-disable itself in eval() on its own.
        attn_mask = None
        if rel_bias is not None:
            attn_mask = rel_bias.unsqueeze(0).expand(B, self.nhead, T, T)
            if key_padding_mask is not None:
                attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
        elif key_padding_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, T, device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
        ctx = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.drop.p if self.training else 0.0)
        ctx = ctx.transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.o(ctx))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class HumanizationTransformer(nn.Module):
    """
    Encoder-only. Bidirectional attention over the WHOLE pattern (wide context),
    then per-position heads predict every hit's humanization IN PARALLEL - one
    forward pass, no autoregression.

    Heads depend on cfg.target_mode:
      "classification" (default): velocity_bin + offset_bin logits (ordinal loss,
          expected-value / sampling decode). Keeps a full output distribution.
      "regression": continuous velocity in [0,1] (raw linear, clamped at decode)
          and offset in [-1,1] (tanh). Infinite resolution, single-valued.

    The model predicts only velocity and timing. It never predicts or changes
    pitch - every hit keeps the exact note you played.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = DrumEventEmbedding(cfg)
        self.seq_pos = nn.Embedding(cfg.max_seq_len + 1, d)

        # #4 relative-position encoding: use custom rel-pos layers, else standard ones
        if cfg.rel_pos_encoding:
            self.rel_bias = RelativePositionBias(cfg.nhead, cfg.rel_pos_buckets)
            self.layers = nn.ModuleList([
                RelPosEncoderLayer(d, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
                for _ in range(cfg.num_layers)])
            self.encoder = None
        else:
            self.rel_bias = None
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg.nhead, dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, cfg.num_layers)
            self.layers = None

        if cfg.target_mode == "regression":
            # DESIGN: raw linear velocity head (NOT sigmoid+MSE - that combo has
            # vanishing gradients near the rails); we clamp to [0,1] at decode.
            self.velocity_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
            self.offset_head   = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1), nn.Tanh())
            # #6 distribution-matching: predict a log-variance (spread) per head so the
            #    model outputs a DISTRIBUTION, not just the mean (regression only).
            if cfg.distribution_match:
                self.vel_logvar = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
                self.off_logvar = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        else:
            self.velocity_head = nn.Linear(d, cfg.velocity_bins)
            self.offset_head   = nn.Linear(d, cfg.offset_bins)

        # #12 per-instrument feel profiles: a learned scalar bias per drum voice added
        #     to each head's pre-activation, so voices can hold distinct feel.
        if cfg.per_instrument_feel:
            self.feel_vel = nn.Embedding(cfg.num_instruments + 1, 1)
            self.feel_off = nn.Embedding(cfg.num_instruments + 1, 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # DESIGN: these two MUST be zeroed AFTER the blanket xavier_uniform_ above,
        # not before. Their weight tensors are 2-D ((num_instruments+1, 1)), so the
        # `p.dim() > 1` sweep matches them and silently overwrites a zero-init done
        # earlier - which is exactly what used to happen here. With fan_in=1 xavier
        # gives U(-0.63, 0.63), i.e. every voice started with a large RANDOM feel
        # bias added straight onto its velocity/offset output, contradicting the
        # "start neutral, learn the offset from data" intent of the feature.
        if cfg.per_instrument_feel:
            nn.init.zeros_(self.feel_vel.weight)   # start neutral (see note above)
            nn.init.zeros_(self.feel_off.weight)

        # DESIGN: depth-scaled residual init. In a pre-norm stack every layer ADDS its
        # branch output to the residual stream, so the stream's variance grows ~linearly
        # with depth; with plain Xavier on every projection a 20-layer model starts with
        # a much hotter residual signal than a 6-layer one, which shows up as loss
        # spikes / divergence in the first few hundred steps precisely at the depths
        # `very_deep` introduces. Scaling each residual branch's OUTPUT projection by
        # 1/sqrt(2*num_layers) (the standard GPT-2 style fix) keeps the post-stack
        # variance roughly depth-independent, so one LR range stays usable across the
        # whole preset ladder instead of needing a per-depth tune. Shallow presets are
        # affected too, but harmlessly - it is a mild, uniform down-scaling there.
        if self.layers is not None:
            resid_scale = 1.0 / math.sqrt(2 * max(1, cfg.num_layers))
            with torch.no_grad():
                for layer in self.layers:
                    layer.o.weight.mul_(resid_scale)          # attention out-projection
                    layer.ff[-1].weight.mul_(resid_scale)     # FFN out-projection

    def _encode(self, batch):
        B, T = batch['instruments'].shape
        idx = torch.arange(T, device=batch['instruments'].device).unsqueeze(0).expand(B, -1)
        x = self.embed(batch['instruments'], batch['positions'], batch['bars'],
                       batch['iois'], batch['in_velocities'],
                       intensity=batch.get('intensity'),
                       ts_ids=batch.get('ts_ids'), beat_slots=batch.get('beat_slots'),
                       iois_same=batch.get('iois_same'), metric_str=batch.get('metric_str'),
                       tempo_norm=batch.get('tempo_norm'), is_flam=batch.get('is_flam'),
                       flam_gap=batch.get('flam_gap'))
        if self.cfg.rel_pos_encoding:
            # relative encoding carries position via attention bias - no absolute add
            rb = self.rel_bias(T, x.device)
            for layer in self.layers:
                x = layer(x, rel_bias=rb, key_padding_mask=batch['pad_mask'])
            return x
        else:
            x = x + self.seq_pos(idx)
            return self.encoder(x, src_key_padding_mask=batch['pad_mask'])

    def forward(self, batch):
        x = self._encode(batch)
        vel = self.velocity_head(x)
        off = self.offset_head(x)
        # #12 add per-instrument feel bias (broadcast over the head's output dim)
        if self.cfg.per_instrument_feel:
            inst = (batch['instruments'] + 1).clamp(0, self.cfg.num_instruments)
            vel = vel + self.feel_vel(inst)     # (B,T,1) broadcasts across bins/scalar
            off = off + self.feel_off(inst)
        out = {'vel': vel, 'off': off}
        # #6 spread heads (regression + distribution_match)
        if self.cfg.target_mode == "regression" and self.cfg.distribution_match:
            out['vel_logvar'] = self.vel_logvar(x)
            out['off_logvar'] = self.off_logvar(x)
        return out

    @torch.no_grad()
    def infer(self, batch, temperature_vel=0.8, temperature_off=0.6, top_k=0,
              vel_decode="expected", off_decode="sample",
              top_k_velocity=None, top_k_timing=None):
        """
        Single forward pass, then decode each head. Returns dict of FLOAT arrays:
          'vel' : velocity in BINS (classification) or [0,1] (regression)
          'off' : offset  in BINS (classification) or [-1,1] (regression)
        top_k_velocity / top_k_timing override the shared top_k per head if given.
        """
        self.eval()
        k_vel = top_k if top_k_velocity is None else top_k_velocity
        k_off = top_k if top_k_timing   is None else top_k_timing
        out = self.forward(batch)
        res = {}
        if self.cfg.target_mode == "regression":
            res['vel'] = out['vel'].squeeze(-1).clamp(0.0, 1.0)     # [0,1]
            res['off'] = out['off'].squeeze(-1).clamp(-1.0, 1.0)    # [-1,1]
        else:
            res['vel'] = self._decode_head(out['vel'], vel_decode, temperature_vel, k_vel)
            res['off'] = self._decode_head(out['off'], off_decode, temperature_off, k_off)
        return res

    @staticmethod
    def _decode_head(logits, mode, temperature, top_k):
        B, T, C = logits.shape
        if mode == "expected":
            bins = torch.arange(C, device=logits.device).float()
            return (F.softmax(logits / max(1e-6, temperature), dim=-1) * bins).sum(-1)
        if mode == "argmax":
            return logits.argmax(-1).float()
        lg = logits / max(1e-6, temperature)
        if top_k > 0:
            lg = _top_k_filter(lg, top_k)
        idx = torch.multinomial(F.softmax(lg, dim=-1).view(B * T, -1), 1).view(B, T)
        return idx.float()


def _top_k_filter(logits, k):
    k = min(k, logits.size(-1))
    vals, _ = torch.topk(logits, k, dim=-1)
    minv = vals[..., -1].unsqueeze(-1)
    return torch.where(logits < minv, torch.full_like(logits, float('-inf')), logits)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def fmt_params(n: int) -> str:
    """Compact parameter count for one-line summaries: 37,741,793 -> '37M'.
    Truncates rather than rounds, so the number never reads higher than the model
    actually is. Falls back to K/exact below a million (custom --d_model can land
    there even though every MODEL_PRESETS entry is well above it)."""
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


# =============================================================================
# TRAINING
# =============================================================================

def _soft_ordinal_targets(tgt: torch.Tensor, num_classes: int, sigma: float) -> torch.Tensor:
    """
    Build soft target distributions: a Gaussian centred on the true class over the
    ordinal axis. DESIGN: with fine velocity bins, predicting bin 101 when the truth
    is 100 should be *almost right*, not "as wrong as 20". A Gaussian soft target
    encodes that neighbourhood structure. Returns (N, num_classes), rows sum to 1.
    """
    device = tgt.device
    idx = torch.arange(num_classes, device=device).float().unsqueeze(0)   # (1, C)
    centre = tgt.float().unsqueeze(1)                                     # (N, 1)
    d2 = (idx - centre) ** 2
    logits = -d2 / (2.0 * sigma * sigma)
    return F.softmax(logits, dim=-1)


def _ordinal_ce(logits: torch.Tensor, tgt: torch.Tensor, sigma: float,
                label_smoothing: float) -> torch.Tensor:
    """Cross-entropy against a soft ordinal target (or plain CE if sigma<=0)."""
    if sigma <= 0:
        return F.cross_entropy(logits, tgt, label_smoothing=label_smoothing)
    soft = _soft_ordinal_targets(tgt, logits.size(-1), sigma)             # (N, C)
    logp = F.log_softmax(logits, dim=-1)
    return -(soft * logp).sum(dim=-1).mean()


def compute_loss(out: dict, batch: dict, cfg: Config):
    """
    Unified loss over the model output dict. Handles:
      • classification mode: ordinal soft-target CE on velocity & offset bins
      • regression mode:      SmoothL1 on continuous velocity [0,1] & offset [-1,1],
        OR Gaussian NLL when distribution_match is on (#6, predicts a spread).
      • #7 optional correlation-aware term: rewards matching the metric-strength↔
        velocity relationship (on-beat stronger) the human performance has.
    """
    valid = ~batch['pad_mask']            # (B,T) bool
    parts = {}

    if cfg.target_mode == "regression":
        pv = out['vel'].squeeze(-1)[valid]
        po = out['off'].squeeze(-1)[valid]
        tv = batch['tgt_vel_cont'][valid]
        to = batch['tgt_off_cont'][valid]
        if cfg.distribution_match and 'vel_logvar' in out:
            # #6 Gaussian negative-log-likelihood: model predicts mean AND spread, so
            #    it's rewarded for honest uncertainty instead of collapsing to the mean.
            vlv = out['vel_logvar'].squeeze(-1)[valid].clamp(-6, 4)
            olv = out['off_logvar'].squeeze(-1)[valid].clamp(-6, 4)
            vel_l = 0.5 * (vlv + (pv - tv) ** 2 / vlv.exp()).mean()
            off_l = 0.5 * (olv + (po - to) ** 2 / olv.exp()).mean()
        else:
            vel_l = F.smooth_l1_loss(pv, tv)
            off_l = F.smooth_l1_loss(po, to)
    else:
        vel_l = _ordinal_ce(out['vel'][valid], batch['tgt_velocities'][valid],
                            cfg.vel_soft_sigma, cfg.label_smoothing)
        off_l = _ordinal_ce(out['off'][valid], batch['tgt_offsets'][valid],
                            cfg.off_soft_sigma, cfg.label_smoothing)
    parts['vel_loss'] = vel_l.item()
    parts['off_loss'] = off_l.item()
    total = cfg.vel_loss_weight * vel_l + cfg.off_loss_weight * off_l

    # #7 correlation-aware loss: match the (metric_strength ↔ velocity) relationship.
    # Human drummers make on-beat (low metric_str) hits stronger; we penalise the
    # model's predicted correlation for deviating from the target's. Uses predicted
    # velocity (expected value in classification, raw in regression).
    if cfg.correlation_loss and 'metric_str' in batch:
        ms = batch['metric_str'].float()
        if cfg.target_mode == "regression":
            vpred = out['vel'].squeeze(-1)
        else:
            vbins = torch.arange(cfg.velocity_bins, device=ms.device).float()
            vpred = (F.softmax(out['vel'], -1) * vbins).sum(-1)
        vtrue = (batch['tgt_velocities'].float() if cfg.target_mode != "regression"
                 else batch['tgt_vel_cont'])

        def _corr(a, b, m):
            a = a[m]; b = b[m]
            if a.numel() < 4:
                return torch.tensor(0.0, device=a.device)
            a = a - a.mean(); b = b - b.mean()
            denom = (a.std() * b.std()).clamp(min=1e-6)
            return (a * b).mean() / denom

        corr_pred = _corr(-ms, vpred, valid)     # -ms so "stronger on-beat" is +corr
        corr_true = _corr(-ms, vtrue, valid)
        corr_l = (corr_pred - corr_true) ** 2
        total = total + cfg.correlation_weight * corr_l
        parts['corr_loss'] = corr_l.item()

    return total, parts


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    """
    Reports mean-absolute-error in real units (velocity in MIDI-velocity units,
    timing in ticks) for both classification and regression modes.
    """
    model.eval()
    vel_scale = 128.0 / cfg.velocity_bins
    mot = cfg.max_offset_ticks
    tot = {'loss': 0.0, 'vel_mae': 0.0, 'off_mae': 0.0}
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        loss, _ = compute_loss(out, batch, cfg)
        valid = ~batch['pad_mask']

        if cfg.target_mode == "regression":
            vhat_v = out['vel'].squeeze(-1).clamp(0, 1) * 127.0            # MIDI units
            vtru_v = batch['tgt_vel_cont'] * 127.0
            ohat_t = out['off'].squeeze(-1).clamp(-1, 1) * mot            # ticks
            otru_t = batch['tgt_off_cont'] * mot
        else:
            vbins = torch.arange(cfg.velocity_bins, device=device).float()
            vhat_v = (F.softmax(out['vel'], -1) * vbins).sum(-1) * vel_scale
            vtru_v = batch['tgt_velocities'].float() * vel_scale
            ohat_t = out['off'].argmax(-1).float() - mot                 # ticks (bin->signed)
            otru_t = batch['tgt_offsets'].float() - mot

        tot['vel_mae'] += (vhat_v[valid] - vtru_v[valid]).abs().mean().item()
        tot['off_mae'] += (ohat_t[valid] - otru_t[valid]).abs().mean().item()

        tot['loss'] += loss.item(); n += 1
    res = {k: v / max(n, 1) for k, v in tot.items()}
    return res


# =============================================================================
# CROSS-RUN LEADERBOARD
# =============================================================================
# A val_loss is only a number ABOUT something. Two runs' val_losses can be ranked
# against each other only if they were measured on the same validation data with the
# same loss - otherwise the "winner" is just whichever run had the easier task. That
# is not hypothetical here: the input de-humanization is re-derived per sample now,
# which made val strictly harder than it was for any run trained before it, and
# data_fraction subsamples the sample list BEFORE make_loaders splits it, so two runs
# at different fractions validate on different sets entirely. So every run stamps what
# it was measured against, and only matching stamps get ranked together.

# Bump this whenever a change alters what val_loss MEANS - the input recipe, the val
# protocol, the loss definition. Runs carrying a different version, or none at all
# (trained before stamping existed), are shown but never ranked.
RECIPE_VERSION = 3

# Config fields that set the SCALE of val_loss, or decide which samples land in val.
_COMPARABILITY_FIELDS = ('target_mode', 'velocity_bins', 'max_offset_ticks',
                         'grid_resolution', 'max_seq_len', 'val_split',
                         'label_smoothing', 'vel_loss_weight', 'off_loss_weight')


def run_fingerprint(cfg: Config, data_fraction: float, n_samples_full: int,
                    song_aware: bool = False) -> Dict:
    """Everything that must match for two runs' val_loss to mean the same thing.
    n_samples_full and the seed are in here because the val split is the first
    val_split share of the seed-shuffled sample list - change the corpus size or the
    seed and the val set is different data, however similar the config looks."""
    fp = {k: getattr(cfg, k, None) for k in _COMPARABILITY_FIELDS}
    fp['recipe_version'] = RECIPE_VERSION
    fp['data_fraction'] = round(float(data_fraction), 4)
    fp['seed'] = GLOBAL_SEED
    fp['n_samples_full'] = int(n_samples_full)
    # Which samples land in val depends on whether the split could group by song, so a
    # song-aware run and a per-sample-split run are not measuring the same val set even
    # with everything else identical.
    fp['song_aware_split'] = bool(song_aware)
    return fp


def samples_are_song_tagged(samples) -> bool:
    """True if the cache carries song_id, i.e. make_loaders can split by song."""
    return any('song_id' in smp for smp in samples)


def _fingerprint_mismatch(saved: Optional[Dict], current: Dict) -> Optional[str]:
    """None if the two are comparable, else a short human reason why they aren't."""
    if not saved:
        return "no run_meta.json (trained before runs were stamped)"
    if saved.get('recipe_version') != current['recipe_version']:
        return (f"recipe v{saved.get('recipe_version', '?')} "
                f"!= v{current['recipe_version']} (different val task)")
    diffs = [k for k, v in current.items() if saved.get(k) != v]
    if diffs:
        k = diffs[0]
        extra = f" (+{len(diffs)-1} more)" if len(diffs) > 1 else ""
        return f"{k}: {saved.get(k)} != {current[k]}{extra}"
    return None


def scan_previous_runs(current_fp: Dict, exclude=(), ckpt_root: str = 'checkpoints'):
    """
    Every checkpoints/<run>/ that already holds a finished epoch, split into the ones
    rankable against current_fp and the ones that aren't (with the reason).

    best_val is read from log.jsonl, not best.pt: recovering one float by torch.load-ing
    a 60M-parameter checkpoint would cost seconds and gigabytes per run, and log.jsonl
    already carries every epoch's val loss. Resumed runs append to it, so the minimum
    over the file is still that run's best.
    """
    ranked, rejected = [], []
    if not os.path.isdir(ckpt_root):
        return ranked, rejected
    for run_dir in sorted(glob.glob(os.path.join(ckpt_root, '*', ''))):
        run_name = os.path.basename(os.path.normpath(run_dir))
        if run_name in exclude:
            continue          # trained in THIS sweep - the fresh result supersedes disk
        log_path = os.path.join(run_dir, 'log.jsonl')
        if not os.path.exists(log_path):
            continue          # never finished an epoch; nothing to rank
        try:
            with open(log_path) as f:
                vals = [json.loads(ln)['loss'] for ln in f if ln.strip()]
            if not vals:
                continue
            meta_path = os.path.join(run_dir, 'run_meta.json')
            saved = None
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    saved = json.load(f)
            reason = _fingerprint_mismatch((saved or {}).get('fingerprint'), current_fp)
            if reason:
                rejected.append((run_name, reason))
                continue
            cfg_path = os.path.join(run_dir, 'config.json')
            with open(cfg_path) as f:
                rc = json.load(f)
            ranked.append({
                'run_name': run_name,
                'model_size': saved.get('model_size', rc.get('model_size', '?')),
                'batch_size': saved.get('batch_size', rc.get('batch_size', 0)),
                'lr': saved.get('lr', rc.get('lr', 0.0)),
                'best_val': min(vals),
                'best_ckpt': os.path.join(run_dir, 'best.pt'),
                'status': 'ok', 'source': 'prior',
            })
        except Exception as exc:
            # A half-written log or a hand-edited config must not take the sweep down.
            rejected.append((run_name, f"unreadable ({type(exc).__name__})"))
    return ranked, rejected


GRID_RESULTS_FILE = os.path.join('checkpoints', 'grid_results.json')

# DESIGN: resolved against THIS FILE's directory, not the current working
# directory - --mode infer and drum_bass_studio.py can both be launched from
# anywhere, and the bundled pretrained model should be found either way.
PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained')
DEFAULT_PRETRAINED_CHECKPOINT = os.path.join(PRETRAINED_DIR, 'humanizer_best.pt')
DEFAULT_PRETRAINED_METADATA = os.path.join(PRETRAINED_DIR, 'humanizer_metadata.json')


def export_pretrained(ckpt_path: str, run_name: str, val_loss: float,
                      fingerprint: Dict, pretrained_dir: str = PRETRAINED_DIR) -> None:
    """
    Copy the winning checkpoint from checkpoints/<run>/best.pt into pretrained/, in a
    form meant to be committed to the repo and used as the default at inference time.

    DESIGN: NOT a raw copy of best.pt. That file also carries optimizer + scheduler
    state (needed to resume training, useless for inference) which for this model is
    ~2x the size of the weights themselves (measured: 250.8MB model vs 483.7MB
    optimizer on a 60.6M-param 'huge' model - stripping it cuts the file to ~1/3).
    load_model() only ever reads ck['model'] and ck['config'], so that's all this
    keeps. metadata.json duplicates the human-relevant facts in a form readable
    without loading the (still large) .pt file - what run this came from, how good it
    is, and what data/recipe it's comparable against.

    Best-effort: this runs after a long sweep/final-train has already produced the
    real result, so a failure here (disk full, bad permissions) must not make the
    run as a whole look like it failed - report and move on, same as write_grid_results.
    """
    try:
        os.makedirs(pretrained_dir, exist_ok=True)
        ck = torch.load(ckpt_path, map_location='cpu')
        lean = {'model': ck['model'], 'config': ck['config']}
        ckpt_out = os.path.join(pretrained_dir, 'humanizer_best.pt')
        tmp = ckpt_out + '.tmp'
        torch.save(lean, tmp)
        os.replace(tmp, ckpt_out)   # atomic - a reader never sees a half-written model

        meta = {
            'saved': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_run': run_name,
            'source_checkpoint': ckpt_path,
            'val_loss': float(val_loss),
            'model_size': ck['config'].get('model_size'),
            'batch_size': ck['config'].get('batch_size'),
            'lr': ck['config'].get('lr'),
            'params': (lambda n: {'count': n, 'fmt': fmt_params(n)})(
                sum(t.numel() for t in ck['model'].values())),
            'fingerprint': fingerprint,
            'note': ("Inference-only checkpoint (model weights + architecture config "
                     "only) - no optimizer/scheduler state, so it cannot be used to "
                     "--resume training. Produced by --mode grid_search; overwritten "
                     "each time that sweep's winner changes."),
        }
        meta_out = os.path.join(pretrained_dir, 'humanizer_metadata.json')
        tmp_meta = meta_out + '.tmp'
        with open(tmp_meta, 'w') as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_meta, meta_out)

        size_mb = os.path.getsize(ckpt_out) / 1e6
        print(f"Pretrained model updated -> {ckpt_out}  ({size_mb:.0f}MB, "
              f"val_loss={val_loss:.4f}, from '{run_name}')")
    except Exception as exc:
        _report_error(f"could not export the winning checkpoint to '{pretrained_dir}' "
                      f"(the run itself is unaffected - '{ckpt_path}' still has the "
                      f"full result)", exc)


def write_grid_results(results, rejected, fingerprint: Dict, axes: Dict,
                       final_info: Optional[Dict] = None,
                       path: str = GRID_RESULTS_FILE) -> None:
    """
    Persist the leaderboard grid_search() just printed, so it survives the terminal.

    DESIGN: the sweep's whole output used to be stdout only - lose the scrollback and
    the ranking was gone, even though every run's log.jsonl was still on disk. The
    numbers were recoverable but only by hand-rebuilding what scan_previous_runs()
    already computes, so this writes the assembled table once instead.

    Called TWICE per sweep - right after the leaderboard prints, and again once the
    auto final train has finished - because the final pass is the long, OOM-prone one:
    if it dies, the ranking that cost the whole sweep must already be on disk. The
    second call simply overwrites with `final` filled in.

    Best-effort: a sweep that trained for hours must never die on a failed write, so
    every error here is reported and swallowed.
    """
    try:
        payload = {
            'saved': time.strftime('%Y-%m-%d %H:%M:%S'),
            'fingerprint': fingerprint,   # what makes these val_losses comparable
            'axes': axes,                 # what was swept, and at what data_fraction
            'leaderboard': [
                # null, not inf, for a failed run: json.dump would emit bare `Infinity`,
                # which is not valid JSON and chokes strict parsers (jq included).
                {'rank': i,
                 'val_loss': (float(r['best_val']) if r['status'] == 'ok' else None),
                 'model_size': r['model_size'], 'batch_size': int(r['batch_size']),
                 'lr': float(r['lr']), 'run_name': r['run_name'],
                 'source': r.get('source', 'sweep'), 'status': r['status'],
                 'best_ckpt': r.get('best_ckpt')}
                for i, r in enumerate(results, 1)
            ],
            # kept, not dropped: "why isn't my earlier run in the table" is exactly the
            # question someone reads this file to answer.
            'not_ranked': [{'run_name': n, 'reason': why} for n, why in rejected],
            'final': final_info,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)   # atomic: a reader never sees a half-written table
        print(f"Leaderboard saved -> {path}")
    except Exception as exc:
        _report_error(f"could not save the grid-search leaderboard to '{path}' "
                      f"(the sweep itself is unaffected - the table is printed above "
                      f"and every run's log.jsonl is intact)", exc)


# The epoch's closing batches are the ones worth seeing (final loss, the numbers the
# epoch summary is about to be built from), so the live display always refreshes on
# each of the last few batches regardless of where the print_every cadence landed.
_LIVE_TAIL_BATCHES = 3


def train(cfg: Config, samples: List[Dict], run_name: str, resume: Optional[str] = None,
          meta: Optional[Dict] = None, data_fraction: float = 1.0, use_tensorboard: bool = True):
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    ckpt_dir = os.path.join('checkpoints', run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, 'config.json'), 'w') as f:
        json.dump(asdict(cfg), f, indent=2)

    # AUTO-RESUME: if the caller didn't explicitly pass --resume, but this run_name
    # already has a checkpoint (e.g. this exact command is being run again), continue
    # from it automatically rather than silently starting over and overwriting the
    # previous run's progress under the same name. An explicit --resume always wins -
    # this only fills in the case where the user just re-ran the same training command.
    auto_resumed = False
    if resume is None:
        auto_path = os.path.join(ckpt_dir, 'last.pt')
        if os.path.exists(auto_path):
            resume = auto_path
            auto_resumed = True

    # DESIGN: writes to checkpoints/<run_name>/tb, i.e. the SAME per-run directory
    # grid_search() already gives each (batch_size, lr, model_size) combo. That means
    # `tensorboard --logdir checkpoints` picks up every run - a standalone train() run
    # AND every combo of a grid_search sweep - as its own line in one dashboard, with
    # no extra plumbing needed in grid_search() beyond passing this flag through.
    writer = None
    if use_tensorboard:
        if HAS_TENSORBOARD:
            tb_dir = os.path.join(ckpt_dir, 'tb')
            writer = SummaryWriter(log_dir=tb_dir)
            print(f"TensorBoard: logging to {tb_dir}  "
                  f"(view with: tensorboard --logdir checkpoints)")
        else:
            print("[note] TensorBoard requested but not installed - "
                  "pip install tensorboard to enable. Continuing without it.")

    # ── Optional cap on training-file usage ─────────────────────────────────────
    # DESIGN: data_fraction subsamples the (already split+filtered) sample list
    # BEFORE augmentation. Augmentation (e.g. bar-rotation) is applied on-the-fly
    # per epoch and has no fixed count, so "fraction of the whole (+augmented) set"
    # means: keep this fraction of the underlying samples, and augmentation still
    # varies THOSE normally. Random subset (not a prefix) so it isn't biased toward
    # whatever order the cache happened to store files in. Useful for a cheap,
    # fast SCREENING pass - e.g. during grid search - before a full-data confirm run.
    n_full = len(samples)
    data_fraction = float(np.clip(data_fraction, 0.0, 1.0))
    if data_fraction < 1.0:
        n_keep = max(cfg.batch_size * 2, int(round(n_full * data_fraction)))
        n_keep = min(n_keep, n_full)
        rng = random.Random(GLOBAL_SEED)   # fixed seed -> reproducible subset across runs
        samples = rng.sample(samples, n_keep)
        print(f"[data_fraction={data_fraction:.2f}] Using {n_keep}/{n_full} training "
              f"samples (random subset) - FASTER but less data than a full run.")

    # Stamp what this run's val_loss will be measured against, so a later sweep can
    # tell whether it is allowed to rank this run against its own (see run_fingerprint).
    # Written here rather than beside config.json above because data_fraction and
    # n_full - both part of what defines the val set - are only settled by this point.
    try:
        with open(os.path.join(ckpt_dir, 'run_meta.json'), 'w') as f:
            json.dump({'fingerprint': run_fingerprint(cfg, data_fraction, n_full,
                                                     samples_are_song_tagged(samples)),
                       'model_size': cfg.model_size, 'batch_size': cfg.batch_size,
                       'lr': cfg.lr, 'run_name': run_name,
                       'saved': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
    except Exception as exc:
        # Losing the stamp costs this run its place in future leaderboards; it must
        # not cost the training run itself.
        print(f"[warning] couldn't write run_meta.json for '{run_name}' "
              f"({type(exc).__name__}: {exc}) - this run won't be rankable later.")

    # ── Dataset provenance summary ────────────────────────────────────────────────
    # original MIDI files -> section-samples after song-splitting/quality-filter ->
    # effective items per epoch including on-the-fly augmentation.
    n_split = len(samples)
    print("\n── Dataset ──────────────────────────────────────────")
    if meta:
        print(f"  Original MIDI files:      {meta.get('original_files_kept', '?')} "
              f"(of {meta.get('original_files_found', '?')} found; "
              f"{meta.get('files_failed', 0)} failed, "
              f"{meta.get('files_fully_rejected', 0)} fully filtered out)")
        if meta.get('split_songs', False):
            print(f"  After song-splitting:     {n_full} section-samples "
                  f"({meta.get('section_bars', '?')}-bar sections, hop {meta.get('hop_bars', '?')})")
        else:
            print(f"  Section-samples:          {n_full} (song-splitting off)")
        if meta.get('sections_rejected', 0):
            files_affected = meta.get('files_fully_rejected', 0) + meta.get('files_partially_rejected', 0)
            print(f"  Quality-filtered out:     {meta['sections_rejected']} flat/robotic sections "
                  f"across {files_affected} files")
    else:
        print(f"  Section-samples:          {n_full} (no cache metadata - "
              f"synthetic or pre-metadata cache)")
    if data_fraction < 1.0:
        print(f"  Capped by --data_fraction: {n_split}/{n_full} samples used for this run")
    # augmentation is applied on-the-fly per sample, so it multiplies the EFFECTIVE
    # variety seen across epochs rather than the item count. Report the expected
    # number of augmented views per epoch.
    if cfg.bar_rotation:
        aug_per_epoch = int(round(n_split * cfg.bar_rotation_prob))
        print(f"  + augmentation (per epoch): ~{aug_per_epoch} of {n_split} samples "
              f"bar-rotated (p={cfg.bar_rotation_prob:.2f}); different picks each epoch")
        print(f"  Effective items/epoch:    {n_split}")
    else:
        print(f"  Total training items:     {n_split} (augmentation off)")
    print("─────────────────────────────────────────────────────\n")

    train_loader, val_loader = make_loaders(samples, cfg)
    model = HumanizationTransformer(cfg).to(device)
    print(f"Model size: '{cfg.model_size}'  "
          f"(d_model={cfg.d_model}, layers={cfg.num_layers}, heads={cfg.nhead}, "
          f"ff={cfg.dim_feedforward}, dropout={cfg.dropout}, "
          f"params={fmt_params(count_params(model))})")
    if cfg.bar_rotation:
        print(f"Bar-rotation augmentation ON, p={cfg.bar_rotation_prob:.2f}.")

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.max_epochs
    sched = OneCycleLR(opt, max_lr=cfg.lr, total_steps=total_steps, pct_start=cfg.warmup_pct)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda')) if HAS_AMP else None

    start_epoch, best_val = 0, float('inf')
    if resume:
        try:
            ck = torch.load(resume, map_location=device)
            model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
            start_epoch = ck['epoch'] + 1; best_val = ck.get('best_val', float('inf'))

            saved_total_steps = ck['scheduler'].get('total_steps')
            if saved_total_steps == total_steps:
                sched.load_state_dict(ck['scheduler'])
            else:
                # DESIGN: OneCycleLR bakes total_steps into its ENTIRE warmup/anneal
                # shape at construction. Loading a state_dict built for a different
                # total_steps (--epochs, --batch_size, or --data_fraction changed since
                # this checkpoint was written - all of which change steps_per_epoch or
                # cfg.max_epochs) doesn't fail here at load time - it silently corrupts
                # internal bookkeeping and crashes much later on a real .step() call
                # ("Tried to step N times, the specified number of total steps is M").
                # `sched` was already constructed above using the CURRENT total_steps,
                # so instead of loading incompatible old state, fast-forward this fresh
                # one to the equivalent point (same completed-epoch count) in the NEW
                # schedule - optimizer/model state restore normally; only the LR curve
                # itself restarts shaped for the new run length.
                steps_done = min(start_epoch * steps_per_epoch, total_steps)
                # Fast-forwarding necessarily steps the schedule with no optimizer
                # step behind it, which trips the same "lr_scheduler.step() before
                # optimizer.step()" warning. It's correct here (the optimizer state
                # for those steps was restored from the checkpoint), so silence it
                # rather than printing a misleading warning on every such resume.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore', message=r'.*lr_scheduler\.step\(\).*before.*optimizer\.step\(\).*')
                    for _ in range(steps_done):
                        sched.step()
                print(f"[note] LR schedule length changed since this checkpoint was "
                      f"written (total_steps {saved_total_steps} -> {total_steps} - "
                      f"likely --epochs/--batch_size/--data_fraction differs). "
                      f"Continuing the LR schedule at the equivalent point in the NEW "
                      f"schedule rather than loading incompatible old state.")

            tag = "Auto-resumed" if auto_resumed else "Resumed"
            print(f"{tag} from {resume} at epoch {start_epoch} (best val {best_val:.4f})")
        except Exception as exc:
            if not auto_resumed:
                raise   # an explicit --resume that fails to load is a real error
            # AUTO-resume found a checkpoint for this run_name but it wouldn't load -
            # almost always an architecture/config mismatch (different d_model,
            # target_mode, velocity_bins, max_seq_len, etc. than whatever produced it).
            # The user only asked to auto-resume when it's actually compatible, so fall
            # back to a fresh model rather than crashing on an incompatibility they
            # likely introduced on purpose (e.g. changed --model_size for this run).
            print(f"[warning] found an existing checkpoint for run_name '{run_name}' "
                  f"at '{resume}' but couldn't load it ({type(exc).__name__}: {exc}).")
            print(f"          Likely a different model config than produced it "
                  f"(architecture/target_mode/velocity_bins/max_seq_len/etc). Starting "
                  f"fresh instead - this WILL overwrite that checkpoint as training "
                  f"proceeds. Use a different --run_name to keep both.")
            start_epoch, best_val = 0, float('inf')

    bad = 0
    log_path = os.path.join(ckpt_dir, 'log.jsonl')
    num_batches = steps_per_epoch
    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        run_loss = run_vel = run_off = 0.0
        t0 = time.time()
        # DESIGN: tqdm gives a live countdown/ETA (elapsed<remaining) - the same style
        # drum_theme_segmentation.py's dataset-build step already uses. Falls back to
        # the original in-place text print (loss/lr every --print_every batches, no
        # ETA) when tqdm isn't installed, so it never becomes a hard dep.
        # bar_format drops tqdm's default rate ({rate_fmt}) - per-batch throughput
        # swings with sequence length and tells you nothing --epochs/ETA doesn't.
        # mininterval=inf disables tqdm's own timed refresh so the bar redraws ONLY
        # on the cadence below, keeping the displayed numbers and the redraw the same
        # event instead of two competing clocks. That also means the bar is driven
        # manually (bar.update(1) per batch) rather than wrapped around the loader:
        # tqdm's iterator only advances its counter inside a refresh, so wrapping it
        # with refreshes suppressed would freeze n/ETA at 0 for the whole epoch.
        step = -1
        bar = (tqdm.tqdm(total=num_batches,
                         desc=f"Epoch {epoch+1}/{cfg.max_epochs}", unit="batch",
                         mininterval=float('inf'),
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                    "[{elapsed}<{remaining}{postfix}]")
               if HAS_TQDM else None)
        try:
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                opt.zero_grad()
                if scaler is not None:
                    with autocast('cuda'):
                        out = model(batch)
                        loss, parts = compute_loss(out, batch, cfg)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    # DESIGN: GradScaler SKIPS optimizer.step() entirely on any batch
                    # whose unscaled grads came out inf/nan - which is guaranteed for
                    # the first batch or two of every run, since the scale starts at
                    # 65536 and has to overflow to find a workable value. Stepping the
                    # LR schedule on a skipped batch advances OneCycleLR past LRs the
                    # weights never actually trained at, and on batch 0 it's also what
                    # raises PyTorch's "lr_scheduler.step() before optimizer.step()"
                    # warning. update() halves the scale exactly when the step was
                    # skipped and never lowers it otherwise, so comparing the scale
                    # across it is the documented way to detect that.
                    scale_before = scaler.get_scale()
                    scaler.step(opt); scaler.update()
                    stepped = scaler.get_scale() >= scale_before
                else:
                    out = model(batch)
                    loss, parts = compute_loss(out, batch, cfg)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
                    stepped = True
                if stepped:
                    sched.step()
                run_loss += loss.item(); run_vel += parts['vel_loss']; run_off += parts['off_loss']
                cur_loss = loss.item()
                # One cadence for both display paths: every cfg.print_every batches,
                # plus each of the epoch's last _LIVE_TAIL_BATCHES.
                show_live = ((step + 1) % cfg.print_every == 0
                             or step >= num_batches - _LIVE_TAIL_BATCHES)
                if bar is not None:
                    bar.update(1)      # counter/ETA advance every batch; drawing does not
                    if show_live:
                        # set_postfix_str(refresh=True) is the redraw - it bypasses
                        # mininterval, so this is the only thing that paints the bar.
                        bar.set_postfix_str(
                            f"loss={cur_loss:.4f} vel={parts['vel_loss']:.4f} "
                            f"off={parts['off_loss']:.4f} lr={sched.get_last_lr()[0]:.2e}")
                # in-place text print: overwrite the SAME line on that same cadence -
                # only when tqdm isn't doing the display, so the two update mechanisms
                # never fight.
                elif show_live:
                    print(f"\r  Epoch {epoch+1}/{cfg.max_epochs}  batch {step+1}/{num_batches}  "
                          f"loss={cur_loss:.4f} vel={parts['vel_loss']:.4f} "
                          f"off={parts['off_loss']:.4f} lr={sched.get_last_lr()[0]:.2e}",
                          end='', flush=True)
                # logged on the print_every cadence regardless of display mode - a
                # scalar every single batch (often thousands/epoch) adds real overhead
                # for no visual gain, since TensorBoard itself downsamples dense series.
                if writer is not None and ((step + 1) % cfg.print_every == 0 or step == num_batches - 1):
                    gstep = epoch * num_batches + step
                    writer.add_scalar('batch/loss', cur_loss, gstep)
                    writer.add_scalar('batch/vel_loss', parts['vel_loss'], gstep)
                    writer.add_scalar('batch/off_loss', parts['off_loss'], gstep)
                    writer.add_scalar('batch/lr', sched.get_last_lr()[0], gstep)
            if bar is not None:
                bar.close()
        except RuntimeError as exc:
            # Most training-step RuntimeErrors are CUDA OOM or a shape mismatch.
            # Report epoch/batch and the exact line, with actionable guidance for OOM.
            if bar is not None:
                bar.close()
            if 'out of memory' in str(exc).lower():
                _report_error(f"training ran out of GPU memory at epoch {epoch+1} "
                              f"batch {step+1}/{num_batches} - reduce --batch_size or "
                              f"use a smaller --model_size", exc, fatal=True)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                _report_error(f"training step failed at epoch {epoch+1} "
                              f"batch {step+1}/{num_batches}", exc, fatal=True)
            if writer is not None:
                writer.close()
            raise   # a broken training step is fatal - don't silently continue

        nb = num_batches
        val = evaluate(model, val_loader, device, cfg)
        # The epoch summary APPENDS to the in-place batch-counter line (no newline was
        # printed above), so each epoch occupies exactly one line: the batch counter
        # ends at the last batch, then '|' separates it from the epoch's own stats.
        is_best = val['loss'] < best_val
        best_tag = "  (✓ new best)" if is_best else ""
        print(f" | train={run_loss/nb:.4f} (vel={run_vel/nb:.4f} off={run_off/nb:.4f})  "
              f"val={val['loss']:.4f}  vel_mae={val['vel_mae']:.2f}  "
              f"off_mae={val['off_mae']:.2f}t  {time.time()-t0:.0f}s{best_tag}")
        with open(log_path, 'a') as f:
            f.write(json.dumps({'epoch': epoch, 'train_loss': run_loss/nb, **val}) + '\n')
        if writer is not None:
            writer.add_scalar('epoch/train_loss', run_loss / nb, epoch)
            writer.add_scalar('epoch/val_loss', val['loss'], epoch)
            writer.add_scalar('epoch/val_vel_mae', val['vel_mae'], epoch)
            writer.add_scalar('epoch/val_off_mae', val['off_mae'], epoch)
            writer.add_scalar('epoch/lr', sched.get_last_lr()[0], epoch)
            writer.flush()

        # DESIGN: best_val is updated BEFORE building `ck`, so last.pt (saved every
        # epoch) always carries the TRUE current best - not the value from before this
        # epoch's own improvement. Previously best_val only updated inside the `if
        # is_best` branch after last.pt had already been written, so resuming from
        # last.pt (including auto-resume) silently restored a stale, one-epoch-old
        # best_val - which also corrupts early-stopping's baseline after a resume.
        if is_best:
            best_val = val['loss']
        ck = {'epoch': epoch, 'model': model.state_dict(),
              'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
              'best_val': best_val, 'config': asdict(cfg)}
        try:
            torch.save(ck, os.path.join(ckpt_dir, 'last.pt'))
            if is_best:
                torch.save(ck, os.path.join(ckpt_dir, 'best.pt'))
                bad = 0
            else:
                bad += 1
                if bad >= cfg.early_stop_patience:
                    print(f"Early stopping at epoch {epoch}."); break
        except Exception as exc:
            # Checkpoint I/O failure (disk full, permissions). Report clearly - the
            # epoch's training is done, so warn and keep going rather than lose the run.
            _report_error(f"saving checkpoint for epoch {epoch+1} to '{ckpt_dir}' "
                          f"(disk full or permissions?)", exc, fatal=True)
    print(f"\nDone. Best val loss: {best_val:.4f}  ->  {ckpt_dir}/best.pt")
    if writer is not None:
        writer.close()
    return {'best_val': best_val, 'ckpt_dir': ckpt_dir,
            'best_ckpt': os.path.join(ckpt_dir, 'best.pt')}


def grid_search(base_cfg: Config, samples: List[Dict], meta: Optional[Dict] = None,
                batch_sizes: Optional[List[int]] = None,
                lrs: Optional[List[float]] = None,
                model_sizes: Optional[List[str]] = None,
                run_prefix: str = "grid", data_fraction: Optional[float] = None,
                use_tensorboard: bool = True,
                auto_final_train: bool = True, final_epoch_multiplier: int = 10,
                final_run_name: Optional[str] = None):
    """
    Full-factorial sweep over batch_size × lr × model_size. Trains one model per
    combination (starting from base_cfg for everything else), evaluates each on its
    own held-out validation split via the SAME train()/evaluate() path used normally,
    and prints every result sorted BEST FIRST (lowest validation loss).

    DESIGN: this is intentionally the small, explicit grid you asked for, not the
    full 60-argument space. Every other setting is held fixed at base_cfg (whatever
    defaults/overrides you already applied, e.g. bar_rotation ON, rel_pos ON, etc.),
    so this sweep isolates exactly the batch_size/lr/model_size axes. Each run gets
    its own checkpoint dir so nothing overwrites.

    data_fraction: SCREENING speed-up. A full grid trains N models on the FULL
    dataset, which is wasteful - you mostly need each combo's RELATIVE ranking, not
    its final quality. Default (None) auto-picks a reasonable fraction based on how
    many combos there are (more combos -> smaller fraction per run), floored at 15%
    and capped at 100%. Pass 1.0 to disable and use the full dataset for every run.

    auto_final_train: after ranking, automatically retrain the winning combo alone
    at data_fraction=1.0 for final_epoch_multiplier x base_cfg.max_epochs epochs -
    the sweep is for RANKING (cheap, on a data subset), this second pass is for the
    actual DELIVERABLE (full data, more epochs since it's the one model you keep).
    Only fires when the sweep itself used a subset (data_fraction < 1.0) AND at
    least one combo succeeded - a sweep already run at data_fraction=1.0 has nothing
    to "upgrade" to, and there's no winner to retrain if every combo failed.
    Skippable (auto_final_train=False) if you'd rather review the leaderboard first
    and retrain by hand.
    """
    # DESIGN: narrowed from the original 3 batch sizes x 2 lrs x {huge, very_deep} grid
    # (Aug 2026, this RTX 4090, recipe v3) to just the winning corner, on the evidence
    # of that actual run (checkpoints/grid_results.json):
    #  - batch_size: 128 beat 64 in all 4 head-to-head pairings (every model x lr combo)
    #    by 0.02-0.03 val_loss. 192 doesn't fit `huge` on a 24GB card (measured: 128
    #    peaks at 18.7GB, 160 at 23.1GB with ~450MB headroom, 192 OOMs outright) - 128
    #    is both the best-scoring and the largest that fits with real margin.
    #  - model_size: huge beat very_deep at every batch_size x lr point tested (e.g.
    #    7.8583 vs 7.8783 at bs128/lr8e-4) despite very_deep being deeper (20 vs 18
    #    layers) - more depth stopped helping once huge's wider d_model (512 vs 384)
    #    was in play, so very_deep is dropped rather than kept as a hedge.
    #  - lr: at bs128 higher lr won BOTH times tested (4e-4 -> 8e-4 improved val_loss
    #    ~0.02-0.03), and 8e-4 was the top of the tested range - the trend was still
    #    climbing, not yet peaked. The default axis below extends upward from there
    #    (toward the linear-scaling-rule prediction of ~1.2e-3 for bs128 off
    #    Config.lr=3e-4 @ bs=32) rather than re-testing 4e-4, which lost clearly. The
    #    old 8e-4 checkpoint still surfaces in the leaderboard automatically via
    #    scan_previous_runs (same fingerprint), so it anchors the comparison for free.
    batch_sizes = batch_sizes or [128]
    lrs = lrs or [1e-3, 1.2e-3, 1.6e-3]
    model_sizes = model_sizes or ["huge"]

    combos = [(bs, lr, ms) for bs in batch_sizes for lr in lrs for ms in model_sizes]
    total = len(combos)

    if data_fraction is None:
        # more combos -> cheaper each run needs to be to keep total sweep time sane.
        # Heuristic: 100% for ≤4 combos, tapering down to a 15% floor by ~24 combos.
        data_fraction = float(np.clip(1.0 - 0.85 * (total - 4) / 20.0, 0.15, 1.0))

    print(f"\n=================== GRID SEARCH ===================")
    print(f"  batch_sizes = {batch_sizes}")
    print(f"  lrs         = {lrs}")
    print(f"  model_sizes = {model_sizes}")
    print(f"  -> {len(batch_sizes)} × {len(lrs)} × {len(model_sizes)} = {total} runs")
    print(f"  data_fraction per run = {data_fraction:.2f}  "
          f"({'full dataset' if data_fraction >= 1.0 else 'SCREENING subset - retrain the winner at 1.0 for the final model'})")
    if use_tensorboard and HAS_TENSORBOARD:
        print(f"  TensorBoard: tensorboard --logdir checkpoints   "
              f"(every combo's run_name appears as its own line)")

    # ── PREFLIGHT: disk ──────────────────────────────────────────────────────────
    # DESIGN: every run writes best.pt AND last.pt, and each holds model params +
    # AdamW's two momentum buffers (~12 bytes/param total, x2 files = ~24). At the
    # big end of the preset ladder that is ~0.9GB PER COMBO, so a wide sweep can
    # quietly need tens of GB. A disk-full failure surfaces as a checkpoint-save
    # error mid-sweep (train() warns and continues), meaning an unattended overnight
    # run could finish having silently saved nothing - worth 2 lines to predict.
    est_ckpt_bytes = 0
    for ms in model_sizes:
        p = MODEL_PRESETS[ms]
        approx_params = (p['num_layers'] * (4 * p['d_model'] ** 2
                                            + 2 * p['d_model'] * p['dim_feedforward'])
                         + 4_000_000)          # rough embeddings/heads allowance
        est_ckpt_bytes += approx_params * 24 * len(batch_sizes) * len(lrs)
    est_gb = est_ckpt_bytes / 1024 ** 3
    print(f"  Estimated checkpoint disk usage: ~{est_gb:.1f} GB "
          f"({total} runs x best.pt+last.pt)")
    try:
        free_gb = shutil.disk_usage(os.path.abspath('checkpoints') if os.path.isdir('checkpoints')
                                    else '.').free / 1024 ** 3
        print(f"  Free disk on that volume:        ~{free_gb:.1f} GB")
        if free_gb < est_gb * 1.15:
            print(f"  *** WARNING: that is not a comfortable margin. Checkpoint saves "
                  f"fail LATE (mid-sweep) and train() only warns, so the sweep would "
                  f"keep running while saving nothing. Free space, or narrow the grid.")
    except Exception:
        pass   # disk_usage is best-effort - never block a sweep over a stat() failure

    print(f"====================================================\n")

    results = []
    for i, (bs, lr, ms) in enumerate(combos):
        # DESIGN: .2e, not .0e - zero decimal digits collapses close lrs onto the same
        # name (1e-3 and 1.2e-3 both round to "1e-03"), which made a real sweep here
        # silently auto-resume combo 2 from combo 1's already-finished checkpoint and
        # report combo 1's val_loss as combo 2's, without training combo 2 at all.
        run_name = f"{run_prefix}_{ms}_bs{bs}_lr{lr:.2e}"
        print(f"\n##### RUN {i+1}/{total}: model_size={ms}  bs={bs}  lr={lr:.2e} "
              f"(run_name='{run_name}') #####")
        # Re-seed before EVERY combo: without this, combo #5's model init/dropout/
        # augmentation stream depends on how much RNG state combos #1-4 consumed,
        # so results wouldn't be reproducible independent of sweep order/composition.
        # Re-seeding here means each combo trains identically whether run inside this
        # sweep or standalone via --mode train with the same settings.
        seed_everything(GLOBAL_SEED, verbose=False)
        cfg = Config(**{k: v for k, v in asdict(base_cfg).items()
                       if k in Config.__dataclass_fields__ and Config.__dataclass_fields__[k].init})
        apply_model_preset(cfg, ms)          # sets d_model/nhead/num_layers/dim_ff/dropout
        cfg.batch_size = bs
        cfg.lr = lr
        try:
            info = train(cfg, samples, run_name, meta=meta, data_fraction=data_fraction,
                        use_tensorboard=use_tensorboard)
            results.append({
                'run_name': run_name, 'model_size': ms, 'batch_size': bs, 'lr': lr,
                'best_val': info['best_val'], 'best_ckpt': info['best_ckpt'],
                'status': 'ok', 'source': 'sweep',
            })
        except Exception as exc:
            # One failing config (e.g. OOM on a large model_size/batch_size combo)
            # must not abort the rest of the sweep - report it and continue.
            _report_error(f"grid-search run {i+1}/{total} ('{run_name}') failed", exc)
            results.append({
                'run_name': run_name, 'model_size': ms, 'batch_size': bs, 'lr': lr,
                'best_val': float('inf'), 'best_ckpt': None,
                'status': f'FAILED: {type(exc).__name__}: {exc}', 'source': 'sweep',
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── merge in earlier runs, sort BEST FIRST, print the leaderboard ─────────────
    # Combos from this sweep plus every run already in checkpoints/ that was measured
    # against the same val set with the same loss, so the ranking reflects everything
    # tried so far and not just this invocation. Runs that AREN'T comparable are listed
    # separately with the reason rather than silently dropped or - far worse - ranked
    # anyway, where an easier val task would look like a better model.
    fingerprint = run_fingerprint(base_cfg, data_fraction, len(samples),
                                  samples_are_song_tagged(samples))
    prior, rejected = scan_previous_runs(
        fingerprint, exclude={r['run_name'] for r in results})
    results.extend(prior)
    results.sort(key=lambda r: r['best_val'])
    print(f"\n\n=================== GRID SEARCH RESULTS (best first) ===================")
    if data_fraction < 1.0:
        print(f"  (each run used a {data_fraction:.0%} random subset of the data - "
              f"for RANKING/screening, not final quality)")
    if prior:
        print(f"  ({len(prior)} earlier run(s) from checkpoints/ included - same val "
              f"set, same loss, so the numbers are directly comparable)")
    print(f"{'rank':>4}  {'val_loss':>10}  {'model_size':>10}  {'batch':>6}  {'lr':>9}  "
          f"{'run_name':<28}  {'from':>6}  status")
    print("-" * 110)
    for rank, r in enumerate(results, 1):
        vloss = f"{r['best_val']:.4f}" if r['status'] == 'ok' else "   -"
        print(f"{rank:>4}  {vloss:>10}  {r['model_size']:>10}  {r['batch_size']:>6}  "
              f"{r['lr']:>9.2e}  {r['run_name']:<28}  {r.get('source', 'sweep'):>6}  "
              f"{r['status']}")
    if rejected:
        print(f"\n  Not ranked - these runs exist in checkpoints/ but their val_loss "
              f"was not measured on this val set / loss, so it is not a like-for-like "
              f"number:")
        for run_name, reason in rejected[:12]:
            print(f"    {run_name:<34}  {reason}")
        if len(rejected) > 12:
            print(f"    ... and {len(rejected) - 12} more")
    # Save NOW, before the auto final train: that pass is the long, OOM-prone one, and
    # the ranking above cost the entire sweep. Rewritten with `final` once it finishes.
    axes = {'batch_sizes': batch_sizes, 'lrs': lrs, 'model_sizes': model_sizes,
            'data_fraction': data_fraction, 'sweep_epochs': base_cfg.max_epochs,
            'run_prefix': run_prefix}
    write_grid_results(results, rejected, fingerprint, axes)
    ok = [r for r in results if r['status'] == 'ok']
    final_info = None
    if ok:
        best = ok[0]
        origin = ("this sweep" if best.get('source', 'sweep') == 'sweep'
                  else "an EARLIER run in checkpoints/")
        print(f"\nBest: {best['run_name']}  (val_loss={best['best_val']:.4f})  "
              f"from {origin}  -> {best['best_ckpt']}")
        if data_fraction < 1.0:
            final_epochs = base_cfg.max_epochs * final_epoch_multiplier
            fr_name = final_run_name or f"{run_prefix}_final"
            if not auto_final_train:
                print(f"  -> This ranking used only {data_fraction:.0%} of the data. Retrain "
                      f"model_size={best['model_size']} batch_size={best['batch_size']} "
                      f"lr={best['lr']:.2e} with --data_fraction 1.0 for the real final model.")
            else:
                print(f"\n=================== AUTO FINAL TRAIN ===================")
                print(f"  The sweep above only ranked combos on a {data_fraction:.0%} data "
                      f"subset - now training the winner on the FULL dataset for the "
                      f"actual deliverable.")
                print(f"  model_size={best['model_size']}  batch_size={best['batch_size']}  "
                      f"lr={best['lr']:.2e}")
                print(f"  run_name={fr_name}  epochs={final_epochs} "
                      f"({final_epoch_multiplier}x the sweep's {base_cfg.max_epochs})  "
                      f"data_fraction=1.0")
                print(f"==========================================================\n")
                seed_everything(GLOBAL_SEED, verbose=False)
                final_cfg = Config(**{k: v for k, v in asdict(base_cfg).items()
                                      if k in Config.__dataclass_fields__
                                      and Config.__dataclass_fields__[k].init})
                apply_model_preset(final_cfg, best['model_size'])
                final_cfg.batch_size = best['batch_size']
                final_cfg.lr = best['lr']
                final_cfg.max_epochs = final_epochs
                # DESIGN: the final train is the ONLY deliverable of the whole run, so
                # unlike a sweep combo (where a failure just drops one leaderboard row)
                # an OOM here must not end the session empty-handed. The sweep ranked
                # this combo at data_fraction<1.0; the final pass uses the FULL dataset,
                # which does not change per-batch VRAM but does change how long memory
                # stays fragmented - so a batch size that squeaked through screening can
                # still OOM here. Retry once at half batch (LR scaled with it, per the
                # linear-scaling rule) rather than losing the run.
                for attempt in (1, 2):
                    try:
                        final_info = train(final_cfg, samples, fr_name, meta=meta,
                                           data_fraction=1.0, use_tensorboard=use_tensorboard)
                        print(f"\nFinal model -> {final_info['best_ckpt']}  "
                              f"(best_val={final_info['best_val']:.4f})")
                        break
                    except Exception as exc:
                        is_oom = 'out of memory' in str(exc).lower()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if attempt == 1 and is_oom and final_cfg.batch_size > 1:
                            new_bs = max(1, final_cfg.batch_size // 2)
                            new_lr = final_cfg.lr * (new_bs / final_cfg.batch_size)
                            print(f"\n[final train] OOM at batch_size="
                                  f"{final_cfg.batch_size}. Retrying once at "
                                  f"batch_size={new_bs}, lr={new_lr:.2e} (LR halved with "
                                  f"the batch, per the linear-scaling rule).")
                            final_cfg.batch_size = new_bs
                            final_cfg.lr = new_lr
                            # a different run_name so the retry cannot auto-resume from
                            # the OOMed attempt's incompatible half-written checkpoint
                            fr_name = f"{fr_name}_bs{new_bs}"
                            seed_everything(GLOBAL_SEED, verbose=False)
                            continue
                        _report_error(f"auto final-train on the winning combo "
                                      f"('{fr_name}') failed", exc, fatal=True)
                        break
    else:
        print("\nAll runs failed - see [ERROR] messages above.")
    if final_info is not None:
        write_grid_results(results, rejected, fingerprint, axes, final_info=final_info)
    # DESIGN: the pretrained/ export always reflects the actual DELIVERABLE. If the
    # auto final train ran, that's the full-data retrain (final_info); if it didn't
    # (data_fraction=1.0, so there was nothing to "upgrade" to, or --no_grid_final_train),
    # it's the sweep's own best row instead. Either way this is "the best model of the
    # last run," never a run from a PRIOR invocation's leaderboard - overwritten every
    # time grid_search finishes, unconditionally, so pretrained/ always tracks the most
    # recent run rather than silently keeping an older-but-still-technically-better one.
    if ok:
        if final_info is not None:
            export_pretrained(final_info['best_ckpt'],
                              final_run_name or f"{run_prefix}_final",
                              final_info['best_val'], fingerprint)
        else:
            export_pretrained(best['best_ckpt'], best['run_name'], best['best_val'],
                              fingerprint)
    print("=" * 100)
    return {'sweep_results': results, 'final': final_info}


# =============================================================================
# INFERENCE
# =============================================================================

def load_model(checkpoint: str, device):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: '{checkpoint}'. Check the path "
                                f"and that training produced a best.pt.")
    try:
        ck = torch.load(checkpoint, map_location=device)
        cfg = Config(**{k: v for k, v in ck['config'].items()
                        if k in Config.__dataclass_fields__ and Config.__dataclass_fields__[k].init})
        model = HumanizationTransformer(cfg).to(device)
        model.load_state_dict(ck['model']); model.eval()
    except Exception as exc:
        _report_error(f"loading checkpoint '{checkpoint}' (corrupt file, or trained with "
                      f"an incompatible model version?)", exc, fatal=True)
        raise
    size = getattr(cfg, 'model_size', '?')
    print(f"Loaded {checkpoint}  (model size '{size}': "
          f"d_model={cfg.d_model}, layers={cfg.num_layers}, "
          f"params={fmt_params(count_params(model))})")
    return model, cfg


def _chunk_batch(arr, cfg, device, intensity=None):
    spb = cfg.grid_steps_per_bar
    t = lambda a, dt=torch.long: torch.tensor(a, dtype=dt).unsqueeze(0).to(device)
    # intensity: if not overridden, derive from this chunk's input velocity mean
    if intensity is None:
        vscale = 128.0 / cfg.velocity_bins
        rv = arr['velocities'].astype(float) * vscale
        intensity = float(np.clip(rv.mean() / 127.0, 0.0, 1.0)) if len(rv) else 0.5
    return {
        'instruments':   t(np.clip(arr['instruments'], 0, cfg.num_instruments - 1)),
        'positions':     t(arr['positions']),
        'bars':          t(arr['positions'] // spb),
        'iois':          t(np.clip(arr['iois'], 0, cfg.max_ioi_steps)),
        'iois_same':     t(np.clip(arr.get('iois_same', np.zeros_like(arr['instruments'])), 0, cfg.max_ioi_steps)),
        'metric_str':    t(np.clip(arr.get('metric_str', np.zeros_like(arr['instruments'])), 0, cfg.num_metric_levels - 1)),
        'in_velocities': t(arr['velocities']),
        'intensity':     torch.tensor([intensity], dtype=torch.float32, device=device),
        'ts_ids':        t(np.clip(arr.get('ts_ids', np.zeros_like(arr['instruments'])), 0, NUM_TIMESIGS - 1)),
        'beat_slots':    t(np.clip(arr.get('beat_slots', np.zeros_like(arr['instruments'])), 0, cfg.beat_slots - 1)),
        'tempo_norm':    torch.tensor(arr.get('tempo_norm', np.zeros(len(arr['instruments']), dtype=np.float32)), dtype=torch.float32, device=device).unsqueeze(0),
        'is_flam':       t(np.clip(arr.get('is_flam', np.zeros_like(arr['instruments'])), 0, 1)),
        'flam_gap':      t(np.clip(arr.get('flam_gap', np.zeros_like(arr['instruments'])), 0, cfg.flam_gap_bins - 1)),
        'pad_mask':      torch.zeros(1, len(arr['instruments']), dtype=torch.bool, device=device),
    }


def apply_fast_hit_velocity_cap(events, velocities, cfg,
                                ceiling=85.0, full_ceiling_hz=18.0, no_cap_hz=6.0,
                                compress=True):
    """
    OPTION B - physical-plausibility filter (a RULE, not learned behaviour).

    Real drummers can't hit hard when hitting fast: a blast beat (32nds on snare)
    tops out around ~85 velocity because the physical stroke can't be both fast and
    powerful. This applies a SOFT, speed-scaled velocity ceiling to runs of fast
    SAME-INSTRUMENT hits, using per-instrument timing (tempo-aware).

    How the ceiling scales with speed (hits/second for that instrument):
      • ≤ no_cap_hz         -> no ceiling (slow enough to hit full power)
      • ≥ full_ceiling_hz   -> full ceiling applied (blast-beat fast)
      • between             -> ceiling ramps in linearly
    So 8ths are untouched, 16ths lightly capped, 32nds fully capped.

    compress=True softly compresses velocities toward the ceiling (keeps some
    dynamic variation below the wall) instead of hard-clipping everyone to it,
    which would sound mechanical.

    Returns (new_velocities, n_affected). Purely a plausibility guard - off by
    default; the model may already have learned this from data (try without first).
    """
    v = velocities.astype(float).copy()
    n = len(events)
    if n == 0:
        return v, 0
    # per-instrument hits/second: use same-instrument IOI (grid steps) + tempo
    spb = cfg.grid_steps_per_bar
    affected = 0
    for i, e in enumerate(events):
        # steps since same instrument last hit (recompute locally, tempo-aware)
        # find previous same-instrument event
        prev_gpos = None
        gpos_i = e.bar * spb + e.grid_step
        for j in range(i - 1, -1, -1):
            if events[j].instrument == e.instrument:
                prev_gpos = events[j].bar * spb + events[j].grid_step
                break
        if prev_gpos is None:
            continue
        steps = gpos_i - prev_gpos
        if steps <= 0:
            continue
        # convert grid steps -> seconds using tempo. one beat = grid_resolution steps.
        tempo = getattr(e, 'tempo_bpm', 120.0) or 120.0
        sec_per_step = (60.0 / tempo) / cfg.grid_resolution
        interval_s = steps * sec_per_step
        hz = 1.0 / max(1e-6, interval_s)
        if hz <= no_cap_hz:
            continue
        # ramp fraction 0..1 between no_cap_hz and full_ceiling_hz
        frac = np.clip((hz - no_cap_hz) / max(1e-6, full_ceiling_hz - no_cap_hz), 0.0, 1.0)
        # effective ceiling for this note: interpolate from 127 (no cap) down to `ceiling`
        eff_ceiling = 127.0 - frac * (127.0 - ceiling)
        if v[i] > eff_ceiling:
            if compress:
                # soft-compress the excess (2:1) so some variation survives above line
                excess = v[i] - eff_ceiling
                v[i] = eff_ceiling + excess * 0.5
            else:
                v[i] = eff_ceiling
            affected += 1
    return v, affected


def humanize_file(checkpoint, input_path, output_path,
                  temperature_vel=0.8, temperature_off=0.6, strength=1.0, top_k=0,
                  vel_decode=None, off_decode=None,
                  strength_velocity=None, strength_timing=None,
                  preserve_grid_distance=0.0, preserve_floor=0.2,
                  top_k_velocity=None, top_k_timing=None,
                  preserve_velocity_dynamics=0.0, preserve_velocity_floor=0.2,
                  intensity=None, context_overlap=0.33,
                  fast_hit_cap=False, fast_hit_ceiling=85.0, refine_passes=1,
                  ar_timing=None, ar_timing_weight=None):
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model, cfg = load_model(checkpoint, device)
    tok = Tokenizer(cfg)
    # inference-time overrides of AR-timing (a post-process, so it doesn't need to
    # match training - the user chooses it at humanize time)
    if ar_timing is not None:
        cfg.ar_timing = bool(ar_timing)
    if ar_timing_weight is not None:
        cfg.ar_timing_weight = float(ar_timing_weight)
    vel_decode = vel_decode or getattr(cfg, 'vel_decode', 'expected')
    off_decode = off_decode or getattr(cfg, 'off_decode', 'sample')
    mode = cfg.target_mode
    mot = cfg.max_offset_ticks
    vscale = 128.0 / cfg.velocity_bins
    print(f"Mode: {mode}  |  velocity_bins={cfg.velocity_bins} "
          f"({vscale:.0f} MIDI-vel/bin)  vel_decode={vel_decode}  off_decode={off_decode}")

    events, load_reason = load_midi_events(input_path, cfg)
    if not events:
        print(f"ERROR: no drum events in {input_path} (reason: {load_reason})"); sys.exit(1)
    events.sort(key=lambda e: (e.bar, e.grid_step, e.instrument))
    print(f"Loaded {len(events)} events from {input_path}")

    original = tok.events_to_arrays(events)
    N = len(events)
    L = cfg.max_seq_len
    # DESIGN: large overlap so every note is decided with plenty of BOTH-side context
    # (well over a measure on each side). A note near a chunk edge would otherwise
    # miss the "1 measure before/after" info the offset depends on; the overlap-blend
    # then cross-fades the duplicated region. Default 1/3 of the window.
    overlap = int(L * context_overlap)
    overlap = max(1, min(overlap, L - 1))

    # ── Intensity conditioning ────────────────────────────────────────────────────
    # The whole-file average input velocity (in [0,1]) is the intensity/genre signal
    # the model was trained on. By default we derive it from your input file, so a
    # loud input -> loud humanization, a soft input -> soft. You can OVERRIDE it to
    # request a specific intensity: e.g. feed a soft groove but ask for metal energy.
    file_intensity = float(np.clip((original['velocities'].astype(float) * vscale).mean() / 127.0, 0.0, 1.0))
    if intensity is None:
        use_intensity = file_intensity
        print(f"Intensity (from input): {use_intensity:.2f}  "
              f"(~{use_intensity*127:.0f} avg velocity - the playing intensity/genre cue)")
    else:
        use_intensity = float(np.clip(intensity, 0.0, 1.0))
        print(f"Intensity (OVERRIDE): {use_intensity:.2f} (~{use_intensity*127:.0f} avg vel); "
              f"input file was {file_intensity:.2f} (~{file_intensity*127:.0f}).")

    # DESIGN: accumulate everything in REAL UNITS (velocity 0..127, offset in ticks)
    # rather than bins. This (a) makes the overlap-blend and --strength blend correct
    # regardless of classification vs regression, and (b) keeps full full precision.

    def _one_pass(source_arrays):
        """Run the chunked model over the given input arrays, return (vel_real, off_real)."""
        hum_ = {'vel': np.zeros(N), 'off': np.zeros(N)}
        counts_ = np.zeros(N)
        start = ci = 0
        while start < N:
            end = min(start + L, N)
            chunk = {k: v[start:end] for k, v in source_arrays.items()}
            b = _chunk_batch(chunk, cfg, device, intensity=use_intensity)
            res = model.infer(b, temperature_vel, temperature_off, top_k,
                              vel_decode=vel_decode, off_decode=off_decode,
                              top_k_velocity=top_k_velocity, top_k_timing=top_k_timing)
            pv = res['vel'].squeeze(0).cpu().numpy().astype(np.float64)
            po = res['off'].squeeze(0).cpu().numpy().astype(np.float64)
            if mode == "regression":
                vel_real = np.clip(pv, 0, 1) * 127.0
                off_real = np.clip(po, -1, 1) * mot
            else:
                vel_real = np.array([tok.bin_to_velocity_f(x) for x in pv])
                off_real = po - mot
            w = np.ones(end - start)
            ramp = min(overlap, (end - start) // 4)
            if ramp > 0 and start > 0:
                w[:ramp] = np.linspace(0, 1, ramp)
            if ramp > 0 and end < N:
                w[-ramp:] = np.linspace(1, 0, ramp)
            hum_['vel'][start:end] += vel_real * w
            hum_['off'][start:end] += off_real * w
            counts_[start:end] += w
            if end >= N:
                break
            start += L - overlap; ci += 1
        counts_[:] = np.maximum(counts_, 1)
        hum_['vel'] /= counts_
        hum_['off'] /= counts_
        return hum_

    # #11 Iterative refinement: run the model, then feed its humanized output back in
    # as the input for another pass. Once neighbours are humanized, a second pass can
    # see context the first (all-quantized) pass could not. 1 = single pass (default).
    passes = max(1, int(refine_passes))
    work = {k: v.copy() for k, v in original.items()}
    for p in range(passes):
        print(f"  {'refine ' if p else ''}pass {p+1}/{passes}...")
        hum = _one_pass(work)
        if p < passes - 1:
            # write this pass's output back into the working arrays for the next pass
            work['velocities'] = np.clip(
                np.array([tok.velocity_to_bin(int(round(v))) for v in hum['vel']]),
                0, cfg.velocity_bins - 1).astype(np.int32)
            work['offsets'] = np.clip(
                np.round(hum['off']).astype(np.int32) + mot, 0, cfg.offset_bins - 1).astype(np.int32)

    # original in the same real units, as the anchor for the strength blend/extrapolation
    orig_vel_real = original['velocities'].astype(float) * vscale   # bins->vel (or vel at 128)
    orig_off_real = original['offsets'].astype(float) - mot         # bins->ticks (YOUR baked-in offsets)

    # ── Strength = how far to move FROM the dry input TOWARD (and past) the model ──
    # result = original + strength * (model - original)
    #   0.0 -> dry input untouched
    #   1.0 -> exactly what the model inferred
    #   >1.0 -> EXTRAPOLATE: push further in the same direction the model moved each
    #          hit ("model made it late -> make it later; louder -> louder").
    # We keep separate velocity and timing strengths because timing saturates at the
    # ±offset ceiling much faster than velocity does. If strength_timing is None it
    # follows strength.
    s_vel = strength if strength_velocity is None else strength_velocity
    s_off = strength if strength_timing   is None else strength_timing

    # ── Preserve already-humanized notes (grid-distance-modulated strength) ────────
    # DESIGN: if the input MIDI already has micro-timing baked in, a note sitting FAR
    # from the grid was probably placed there deliberately (already humanized), while
    # a note sitting exactly ON the grid is "raw" and safe to humanize fully. So we
    # scale the effective TIMING strength per note by its existing grid distance:
    #   on-grid  (|offset|≈0)      -> full strength (let the model work)
    #   far off  (|offset|≈ceiling) -> strength floored toward `preserve_floor`
    # preserve_grid_distance in [0,1] sets how strongly this applies (0 = off, the old
    # behaviour). preserve_floor is the minimum strength a maximally-off-grid note keeps.
    per_note_s_off = np.full(N, s_off, dtype=float)
    if preserve_grid_distance > 0.0:
        dist = np.abs(orig_off_real) / max(1.0, mot)          # 0 (on grid) … 1 (at ceiling)
        dist = np.clip(dist, 0.0, 1.0)
        # scale factor: 1 at grid, -> preserve_floor as dist->1, mixed by preserve_grid_distance
        scale = 1.0 - preserve_grid_distance * (1.0 - preserve_floor) * dist
        per_note_s_off = s_off * scale
        n_preserved = int(np.sum(dist > 0.15))
        print(f"Grid-distance preservation ON (amount={preserve_grid_distance:.2f}, "
              f"floor={preserve_floor:.2f}): {n_preserved}/{N} off-grid notes will keep "
              f"more of their baked-in timing.")

    # ── Preserve already-expressive DYNAMICS (velocity analog of the above) ────────
    # DESIGN: the velocity analog of "already off-grid" is "already dynamically
    # varied". If the velocities around a note already swing a lot, you've probably
    # dialed expressive dynamics there by hand - so scale the model's VELOCITY
    # influence down in those spots, and let it work fully where the input is flat
    # (a run of identical velocities that clearly wasn't humanized). We measure each
    # note's deviation from a local rolling mean of the input velocity.
    per_note_s_vel = np.full(N, s_vel, dtype=float)
    if preserve_velocity_dynamics > 0.0 and N > 1:
        v = orig_vel_real.astype(float)
        win = min(9, N if N % 2 else N - 1)          # odd window ≤ 9
        if win < 3:
            win = 3
        pad = win // 2
        vpad = np.pad(v, pad, mode='edge')
        local_mean = np.convolve(vpad, np.ones(win) / win, mode='valid')[:N]
        local_dev = np.abs(v - local_mean)
        # normalize deviation to [0,1] against a musically meaningful spread (~24 vel)
        norm_dev = np.clip(local_dev / 24.0, 0.0, 1.0)
        scale = 1.0 - preserve_velocity_dynamics * (1.0 - preserve_velocity_floor) * norm_dev
        per_note_s_vel = s_vel * scale
        n_dyn = int(np.sum(norm_dev > 0.25))
        print(f"Velocity-dynamics preservation ON (amount={preserve_velocity_dynamics:.2f}, "
              f"floor={preserve_velocity_floor:.2f}): {n_dyn}/{N} already-expressive notes "
              f"will keep more of their programmed dynamics.")

    if (not (abs(s_vel - 1.0) < 1e-9 and abs(s_off - 1.0) < 1e-9)
            or preserve_grid_distance > 0.0 or preserve_velocity_dynamics > 0.0):
        print(f"Strength blend  velocity={s_vel:.2f}  timing={s_off:.2f}"
              f"{'  (EXTRAPOLATING >100%)' if max(s_vel, s_off) > 1.0 else ''}")
    hum['vel'] = orig_vel_real + per_note_s_vel * (hum['vel'] - orig_vel_real)
    hum['off'] = orig_off_real + per_note_s_off * (hum['off'] - orig_off_real)

    # ── #3 Autoregressive-over-beats timing: give micro-timing MOMENTUM ───────────
    # Real drummers drift and recover - a push on one beat carries into the next
    # rather than resetting. We add a causal exponential carry: each note inherits a
    # fraction of the running timing "drift" from earlier notes (in time order). This
    # turns independent per-note offsets into a correlated push/pull, off by default.
    if cfg.ar_timing and len(events) > 1:
        w = float(np.clip(cfg.ar_timing_weight, 0.0, 0.9))
        order = np.argsort([e.raw_tick for e in events])   # time order
        drift = 0.0
        smoothed = hum['off'].copy()
        for oi in order:
            smoothed[oi] = hum['off'][oi] + w * drift
            # update running drift as an EMA of realized offsets
            drift = (1 - w) * drift + w * smoothed[oi]
        # keep within the ±ceiling; re-clip happens below too
        hum['off'] = np.clip(smoothed, -mot, mot)
        print(f"AR-over-beats timing ON (weight={w:.2f}): micro-timing given momentum.")

    # ── OPTION B: fast-hit velocity cap (physical-plausibility rule, opt-in) ───────
    if fast_hit_cap:
        capped, n_aff = apply_fast_hit_velocity_cap(events, hum['vel'], cfg,
                                                    ceiling=fast_hit_ceiling)
        hum['vel'] = capped
        print(f"Fast-hit velocity cap ON (ceiling ~{fast_hit_ceiling:.0f}): "
              f"{n_aff}/{len(events)} fast hits softened (e.g. blast-beat snares).")

    # Safety rails (always applied): velocity ∈ [1,127], timing ∈ [±mot]. Extrapolation
    # past these saturates rather than corrupting the MIDI. Track how much clipped so
    # the user gets feedback when they push strength high.
    vel_clipped = int(np.sum((hum['vel'] < 1) | (hum['vel'] > 127)))
    off_clipped = int(np.sum(np.abs(hum['off']) > mot))

    # write results back onto the events (real units). Pitch is ALWAYS the original
    # note the user played - the model only changes velocity and micro-timing.
    hum_events = []
    for i, src in enumerate(events):
        hum_events.append(DrumEvent(
            instrument=src.instrument, bar=src.bar, grid_step=src.grid_step,
            velocity_bin=0,  # unused downstream; raw_velocity is authoritative
            offset_ticks=int(round(np.clip(hum['off'][i], -mot, mot))),
            raw_velocity=int(round(np.clip(hum['vel'][i], 1, 127))),
            raw_tick=src.raw_tick, raw_pitch=src.raw_pitch,
            ts_id=src.ts_id, beat_slot=src.beat_slot,
            is_flam=src.is_flam, flam_gap_ms=src.flam_gap_ms, tempo_bpm=src.tempo_bpm,
        ))
    events_to_midi(hum_events, output_path, cfg)

    ov = orig_vel_real
    hv = np.array([e.raw_velocity for e in hum_events], dtype=float)
    oo = orig_off_real
    ho = np.array([e.offset_ticks for e in hum_events], dtype=float)
    print("\n── Humanization Summary ─────────────────────────────")
    print(f"  Velocity  mean {ov.mean():.1f} -> {hv.mean():.1f}   "
          f"std {ov.std():.1f} -> {hv.std():.1f}  (MIDI velocity units)")
    print(f"  Timing    mean {oo.mean():.1f} -> {ho.mean():.1f}   "
          f"std {oo.std():.1f} -> {ho.std():.1f} ticks")
    if vel_clipped or off_clipped:
        print(f"  Saturation (from high strength): "
              f"{vel_clipped} velocities hit [1,127], "
              f"{off_clipped} timings hit the ±{mot}-tick ceiling.")


# =============================================================================
# CLI
# =============================================================================

def main():
    seed_everything(GLOBAL_SEED)   # single seed (42) for every RNG in the program
    p = argparse.ArgumentParser(description="MIDI Drum Humanization Transformer (v3, encoder-only)")
    p.add_argument('--mode', required=True, choices=['cache', 'train', 'infer', 'grid_search'])
    p.add_argument('--data_dir', default='./midi_collection')
    p.add_argument('--cache', default='cache/samples.pkl')
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--synthetic_n', type=int, default=2000)
    p.add_argument('--run_name', default='run_001')
    p.add_argument('--resume', default=None)
    p.add_argument('--checkpoint', default=None,
                   help="trained model (best.pt). If omitted, falls back to the "
                        "bundled pretrained/humanizer_best.pt if present.")
    p.add_argument('--input', default=None)
    p.add_argument('--output', default='humanized.mid')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--model_size', default=None,
                   choices=list(MODEL_PRESETS.keys()),
                   help='progressively bigger presets: tiny->small->base(default)->'
                        'deep->deeper->very_deep->huge. Deeper models may capture more '
                        'musical intuition but need more data/compute and can overfit. '
                        'very_deep is the DEEPEST (20 layers) but narrower than huge, so '
                        'it has fewer params (37.7M vs 60.6M) - depth-first by design; '
                        'see the --model_size note in README.md for the measurements. '
                        'Set at TRAIN time; inference reads it from the checkpoint.')
    p.add_argument('--grid_batch_sizes', type=str, default=None,
                   help="GRID_SEARCH: comma-separated batch sizes, e.g. '64,128' "
                        "(default: 128 - the Aug 2026 sweep on this 4090 showed 128 "
                        "beating 64 in every pairing, and 128 is the largest that fits "
                        "'huge' with real headroom; 192 OOMs. Pass your own list to "
                        "explore other sizes, e.g. on a smaller GPU).")
    p.add_argument('--grid_lrs', type=str, default=None,
                   help="GRID_SEARCH: comma-separated learning rates, e.g. "
                        "'8e-4,1e-3,1.2e-3' (default: 1e-3,1.2e-3,1.6e-3 - extends "
                        "upward from the prior sweep's 8e-4, which beat 4e-4 at bs128 "
                        "and was still the top of the tested range, toward the "
                        "linear-scaling-rule prediction of ~1.2e-3 for bs128 off "
                        "Config.lr=3e-4 @ bs=32).")
    p.add_argument('--grid_model_sizes', type=str, default=None,
                   help="GRID_SEARCH: comma-separated model sizes, e.g. 'huge,very_deep' "
                        "(default: huge - it beat very_deep at every batch_size x lr "
                        "point in the Aug 2026 sweep despite very_deep's extra layers).")
    p.add_argument('--no_grid_final_train', action='store_true',
                   help="GRID_SEARCH: after ranking combos on the (usually subsampled) "
                        "sweep, the winner is automatically retrained on the FULL "
                        "dataset for --grid_final_epoch_multiplier x --epochs epochs - "
                        "one command does screening AND produces the real deliverable. "
                        "Pass this to skip that and just get the leaderboard.")
    p.add_argument('--grid_final_epoch_multiplier', type=int, default=10,
                   help="GRID_SEARCH: epoch multiplier for the automatic final full-data "
                        "training pass (default 10x --epochs). E.g. --epochs 5 sweeps at "
                        "5 epochs/combo, then trains the winner for 50 epochs on all the "
                        "data. NOTE this multiplies against the FULL dataset while the "
                        "sweep ran on --data_fraction of it, so at --epochs 20 the final "
                        "pass alone costs ~5x the entire sweep - lower it if that is more "
                        "than you meant to spend.")
    p.add_argument('--grid_final_run_name', default=None,
                   help="GRID_SEARCH: run_name for the automatic final training pass "
                        "(default: '<run_name>_final').")
    p.add_argument('--data_fraction', type=float, default=None,
                   help='TRAIN / GRID_SEARCH: cap training-data usage to this fraction '
                        '(0.0–1.0) of the whole (+augmented) training set - a random '
                        'subset, taken before augmentation (augmentation still varies '
                        'the kept subset normally). Speeds up experiments/screening. '
                        'TRAIN mode: default 1.0 (use everything). GRID_SEARCH: default '
                        'auto-scales down with the number of combos (floored at 0.15) '
                        'unless you set this explicitly; pass 1.0 to use full data '
                        'for every combo.')
    p.add_argument('--d_model', type=int, default=None,
                   help='override d_model from the chosen preset (advanced)')
    p.add_argument('--num_layers', type=int, default=None,
                   help='override num_layers from the chosen preset (advanced)')
    p.add_argument('--dropout', type=float, default=None,
                   help='override dropout from the chosen preset (advanced)')
    p.add_argument('--max_seq_len', type=int, default=None,
                   help='TRAIN: notes per training window; EVERY sample is padded to this '
                        'length, so an oversized value burns compute on padding - attention '
                        'is O(n^2), so 4x too long is ~16x the attention work. Match it to '
                        'your actual note counts (the cache build prints them). Samples '
                        'longer than this are randomly cropped during training and chunked '
                        'with overlap at inference, so a smaller value is safe, not lossy. '
                        'Also sets the positional-embedding size, so it is baked into the '
                        'checkpoint and re-read automatically at infer time.')
    p.add_argument('--num_workers', type=int, default=None)
    p.add_argument('--print_every', type=int, default=None,
                   help='TRAIN: refresh the live batch-progress display every N '
                        'batches, in-place (default 15; the last 3 batches of each '
                        'epoch always refresh). 1 = every batch.')
    p.add_argument('--no_tensorboard', action='store_true',
                   help='TRAIN / GRID_SEARCH: disable TensorBoard logging (on by default '
                        'when the tensorboard package is installed). Logs go to '
                        "checkpoints/<run_name>/tb - view with: "
                        "tensorboard --logdir checkpoints")
    p.add_argument('--velocity_bins', type=int, default=None,
                   help='128=lossless (1 MIDI-vel/bin), 64=Δ2, 32=Δ4. '
                        'Must match between cache-build and train.')
    p.add_argument('--target_mode', default=None, choices=['classification', 'regression'],
                   help='classification (binned+ordinal, default) or regression '
                        '(continuous scalar heads). Set at train time.')
    p.add_argument('--temperature_vel', type=float, default=0.8)
    p.add_argument('--temperature_off', type=float, default=0.6)
    p.add_argument('--intensity', type=float, default=None,
                   help='INFERENCE: overall playing intensity / genre cue, either 0..1 '
                        '(normalized) or a raw MIDI velocity 1..127. Default: derived '
                        'from the input file average (loud in->loud out, soft in->soft). '
                        'Override to request a target energy: e.g. 0.9 (~114) for metal '
                        'intensity, 0.4 (~50) for soft jazz, regardless of input level.')
    p.add_argument('--context_overlap', type=float, default=0.33,
                   help='INFERENCE: fraction of the window (0..0.9) that consecutive '
                        'chunks overlap, so each note is decided with more surrounding '
                        'context (the measures before/after). Higher = more context but '
                        'slower. Default 0.33.')
    p.add_argument('--fast_hit_cap', action='store_true',
                   help='INFERENCE: apply a physical-plausibility velocity ceiling to '
                        'fast same-instrument runs (e.g. blast-beat 32nd snares cap ~85). '
                        'A RULE, not learned - off by default; try WITHOUT first to see '
                        'if the model already learned the ceiling from data, then A/B.')
    p.add_argument('--fast_hit_ceiling', type=float, default=85.0,
                   help='INFERENCE: the velocity ceiling for maximally-fast hits when '
                        '--fast_hit_cap is on (default 85). Slower hits get a higher '
                        'ceiling automatically; 8ths and below are untouched.')
    # ── "Smarter network" toggles ────────────────────────────────────────────────
    p.add_argument('--no_rel_pos', action='store_true',
                   help='TRAIN: disable relative-position encoding (#4), fall back to '
                        'absolute learned positions. Rel-pos is ON by default.')
    p.add_argument('--per_instrument_feel', dest='per_instrument_feel',
                   action='store_true', default=None,
                   help='TRAIN: learn per-voice feel biases (#12). ON by default.')
    p.add_argument('--no_per_instrument_feel', dest='per_instrument_feel',
                   action='store_false',
                   help='TRAIN: disable per-instrument feel profiles (#12).')
    p.add_argument('--distribution_match', action='store_true',
                   help='TRAIN (regression only): predict a spread via Gaussian NLL (#6) '
                        'so output is a distribution, not just the timid mean. Off by default.')
    p.add_argument('--correlation_loss', action='store_true',
                   help='TRAIN: add a loss rewarding the metric-strength↔velocity '
                        'relationship - on-beat stronger, off-beat mellower (#7). Off by default.')
    p.add_argument('--ar_timing', action='store_true',
                   help='INFERENCE: give micro-timing momentum across beats (#3) so '
                        'pushes drift and recover like a real player. Off by default.')
    p.add_argument('--ar_timing_weight', type=float, default=None,
                   help='INFERENCE: strength of the AR-timing carry (default 0.3).')
    p.add_argument('--refine_passes', type=int, default=1,
                   help='INFERENCE: iterative refinement (#11) - re-run the model on its '
                        'own output N times so later passes see humanized context. '
                        'Default 1 (single pass); try 2–3.')
    p.add_argument('--no_split_songs', action='store_true',
                   help='CACHE: disable splitting long files into sections (default: '
                        'split ON, so whole songs become verse/chorus-sized samples).')
    p.add_argument('--section_bars', type=int, default=16,
                   help='CACHE: section length in bars when splitting songs (default 16).')
    p.add_argument('--hop_bars', type=int, default=8,
                   help='CACHE: hop between sections in bars; <section_bars gives '
                        'overlapping sections so transitions are still seen (default 8).')
    p.add_argument('--no_quality_filter', action='store_true',
                   help='CACHE: keep flat/robotic sections (default: filter them out, '
                        'since a mechanical MIDI teaches the model nothing about feel).')
    p.add_argument('--min_velocity_std', type=float, default=None,
                   help='CACHE: reject-if-flat velocity std threshold (MIDI vel units, default 8).')
    p.add_argument('--min_velocity_range', type=float, default=None,
                   help='CACHE: reject-if-flat velocity range threshold (default 24).')
    p.add_argument('--min_offset_std', type=float, default=None,
                   help='CACHE: reject-if-flat timing std threshold (ticks, default 4).')
    p.add_argument('--min_offset_range', type=float, default=None,
                   help='CACHE: reject-if-flat timing range threshold (ticks, default 12).')
    p.add_argument('--bar_rotation', dest='bar_rotation', action='store_true',
                   default=None,
                   help='TRAIN: enable bar-rotation augmentation - drop the first bars so '
                        'phrases start at varied points (drops 2/4/8 bars for grooves '
                        '≥4/≥8/≥16 bars), so the model does not over-index on phrase '
                        'openings. Safe (result is still a real performance). ON by default.')
    p.add_argument('--no_bar_rotation', dest='bar_rotation', action='store_false',
                   help='TRAIN: disable bar-rotation augmentation (it is ON by default).')
    p.add_argument('--bar_rotation_prob', type=float, default=None,
                   help='TRAIN: probability of applying bar rotation per eligible sample '
                        '(default 0.6).')
    p.add_argument('--bar_rotation_max', type=int, default=None,
                   help=argparse.SUPPRESS)   # deprecated: rotation now drops exactly 2 bars
    p.add_argument('--strength', type=float, default=1.0,
                   help='how far to move from the dry input toward the model. '
                        '0=no change, 1.0=exactly the model, >1.0=EXTRAPOLATE / '
                        'exaggerate past the model (e.g. 1.5 = 50%% further). '
                        'Sane range ~0.0–2.0. Applies to both velocity and timing '
                        'unless overridden below.')
    p.add_argument('--strength_velocity', type=float, default=None,
                   help='override --strength for VELOCITY only (dynamics). '
                        'Good for pushing accents harder while keeping timing tame.')
    p.add_argument('--strength_timing', type=float, default=None,
                   help='override --strength for TIMING only (push/pull). Note timing '
                        'saturates at the ±1/16-note ceiling, so very high values clip.')
    p.add_argument('--preserve_grid_distance', type=float, default=0.0,
                   help='0=off. >0 preserves timing you already baked into the input: '
                        'notes already FAR from the grid (likely already humanized) get '
                        'LESS model timing-influence, notes ON the grid get full strength. '
                        'Try 1.0 for strong preservation. Only affects timing.')
    p.add_argument('--preserve_floor', type=float, default=0.2,
                   help='with --preserve_grid_distance, the minimum timing-strength a '
                        'maximally off-grid note keeps (0=freeze it entirely at its baked-in '
                        'position, 1=no reduction). Default 0.2.')
    p.add_argument('--preserve_velocity_dynamics', type=float, default=0.0,
                   help='0=off. >0 preserves dynamics you already programmed: notes in '
                        'already-varied passages get LESS model velocity-influence, notes '
                        'in flat/uniform passages get full strength. Velocity analog of '
                        '--preserve_grid_distance. Try 1.0 for strong preservation.')
    p.add_argument('--preserve_velocity_floor', type=float, default=0.2,
                   help='with --preserve_velocity_dynamics, the minimum velocity-strength a '
                        'maximally-expressive note keeps (0=freeze its programmed velocity, '
                        '1=no reduction). Default 0.2.')
    p.add_argument('--top_k', type=int, default=0,
                   help='shared sampling top-k (0=off). Applies to both heads unless '
                        'overridden per-head below.')
    p.add_argument('--top_k_velocity', type=int, default=None,
                   help='override --top_k for VELOCITY sampling only (128-bin vocab; '
                        'a looser k allows more dynamic variety).')
    p.add_argument('--top_k_timing', type=int, default=None,
                   help='override --top_k for TIMING sampling only (61-bin vocab; '
                        'a tighter k keeps the groove disciplined).')
    p.add_argument('--vel_decode', default=None, choices=['expected', 'sample', 'argmax'],
                   help='velocity decode at inference (default: model config / expected)')
    p.add_argument('--off_decode', default=None, choices=['expected', 'sample', 'argmax'],
                   help='timing-offset decode at inference (default: model config / sample)')
    p.add_argument('--vel_soft_sigma', type=float, default=None,
                   help='TRAIN-time ordinal loss width for VELOCITY (bins). Larger = '
                        'smoother/softer velocity learning. Default 1.0; 0 = hard one-hot.')
    p.add_argument('--off_soft_sigma', type=float, default=None,
                   help='TRAIN-time ordinal loss width for TIMING (bins). Default 1.0; '
                        '0 = hard one-hot.')
    args = p.parse_args()

    # DESIGN: printed immediately, before any of the slow/silent steps below (cache
    # pickle.load of a multi-GB file, MIDI scanning, model build), so a GPU-vs-CPU
    # question never depends on watching a long training preamble first, and so the
    # terminal shows SOMETHING right after argparse instead of going quiet.
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  (CUDA available - will train on GPU)")
    else:
        print("GPU: none detected - training will run on CPU (slow). "
              "If you have an NVIDIA GPU, check that torch was installed with CUDA "
              "support: python -c \"import torch; print(torch.__version__)\" should "
              "show a '+cuXXX' suffix, not '+cpu'.")

    cfg = Config()
    def apply_overrides(c):
        # 1) whole-model preset first (sets d_model/nhead/num_layers/dim_ff/dropout)
        if args.model_size is not None:
            apply_model_preset(c, args.model_size)
        # 2) then any fine-grained overrides win on top of the preset
        if args.epochs is not None:        c.max_epochs = args.epochs
        if args.batch_size is not None:    c.batch_size = args.batch_size
        if args.lr is not None:            c.lr = args.lr
        if args.d_model is not None:       c.d_model = args.d_model
        if args.num_layers is not None:    c.num_layers = args.num_layers
        if args.dropout is not None:       c.dropout = args.dropout
        if args.max_seq_len is not None:   c.max_seq_len = args.max_seq_len
        if args.num_workers is not None:   c.num_workers = args.num_workers
        if args.print_every is not None:   c.print_every = args.print_every
        if args.target_mode is not None:   c.target_mode = args.target_mode
        if args.vel_soft_sigma is not None: c.vel_soft_sigma = args.vel_soft_sigma
        if args.off_soft_sigma is not None: c.off_soft_sigma = args.off_soft_sigma
        if getattr(args, 'bar_rotation', None) is not None:
            c.bar_rotation = bool(args.bar_rotation)
        if getattr(args, 'no_rel_pos', False): c.rel_pos_encoding = False
        if getattr(args, 'per_instrument_feel', None) is not None:
            c.per_instrument_feel = args.per_instrument_feel
        if getattr(args, 'distribution_match', False): c.distribution_match = True
        if getattr(args, 'correlation_loss', False): c.correlation_loss = True
        if getattr(args, 'bar_rotation_prob', None) is not None:
            c.bar_rotation_prob = float(np.clip(args.bar_rotation_prob, 0.0, 1.0))
        if getattr(args, 'bar_rotation_max', None) is not None:
            c.bar_rotation_max = max(1, int(args.bar_rotation_max))
        if getattr(args, 'no_quality_filter', False): c.quality_filter = False
        if getattr(args, 'min_velocity_std', None) is not None:   c.min_velocity_std = args.min_velocity_std
        if getattr(args, 'min_velocity_range', None) is not None: c.min_velocity_range = args.min_velocity_range
        if getattr(args, 'min_offset_std', None) is not None:     c.min_offset_std = args.min_offset_std
        if getattr(args, 'min_offset_range', None) is not None:   c.min_offset_range = args.min_offset_range
        if args.velocity_bins is not None:
            c.velocity_bins = args.velocity_bins
            c.__post_init__()   # recompute derived fields
        # safety: nhead must divide d_model. If a manual d_model override broke
        # that, fall back to the largest head count that divides it.
        if c.d_model % c.nhead != 0:
            for h in (8, 6, 4, 2, 1):
                if c.d_model % h == 0:
                    print(f"[!] d_model={c.d_model} not divisible by nhead={c.nhead}; "
                          f"using nhead={h}.")
                    c.nhead = h
                    break
        return c
    cfg = apply_overrides(cfg)

    if args.mode == 'cache':
        build_cache(args.data_dir, args.cache, cfg, num_workers=max(1, cfg.num_workers),
                    split_songs=not args.no_split_songs,
                    section_bars=args.section_bars, hop_bars=args.hop_bars)

    def _load_training_data(cfg):
        """Shared by --mode train and --mode grid_search: load synthetic or cached
        samples, apply cache-derived config, and back-fill older cache formats.
        Returns (cfg, samples, meta)."""
        if args.synthetic:
            print(f"Generating {args.synthetic_n} synthetic samples...")
            tok = Tokenizer(cfg)
            ev_lists = generate_synthetic_events(cfg, args.synthetic_n)
            samples = [{'input': tok.events_to_arrays(tok.quantize_events(ev)),
                        'target': tok.events_to_arrays(ev),
                        'length': len(ev), 'events': ev} for ev in ev_lists]
            meta = {'original_files_kept': args.synthetic_n,
                    'original_files_found': args.synthetic_n,
                    'split_samples': len(samples), 'split_songs': False}
            return cfg, samples, meta
        if not os.path.exists(args.cache):
            print(f"Cache not found: {args.cache}. Run --mode cache first, or use --synthetic.")
            sys.exit(1)
        # DESIGN: a real-library cache is a few GB, and pickle.load has no built-in way
        # to report progress - unpickling it is a single blocking call that otherwise
        # leaves the terminal silent for tens of seconds with zero feedback, which reads
        # as a hang. Print size before, elapsed after.
        cache_mb = os.path.getsize(args.cache) / (1024 ** 2)
        print(f"Loading cache: {args.cache} ({cache_mb:,.0f} MB)...", flush=True)
        t_load = time.time()
        with open(args.cache, 'rb') as f:
            blob = pickle.load(f)
        samples = blob['samples']
        print(f"Cache loaded: {len(samples):,} samples ({time.time() - t_load:.1f}s)")
        meta = blob.get('meta')
        saved = blob.get('cfg', {})
        cache_bins = saved.get('velocity_bins', 32)
        if args.velocity_bins is not None and args.velocity_bins != cache_bins:
            print(f"ERROR: --velocity_bins {args.velocity_bins} conflicts with the "
                  f"cache's {cache_bins}. Rebuild the cache at the desired resolution:\n"
                  f"  --mode cache --velocity_bins {args.velocity_bins} ...")
            sys.exit(1)
        cfg = Config(**{k: v for k, v in saved.items()
                        if k in Config.__dataclass_fields__ and Config.__dataclass_fields__[k].init})
        saved_vb = args.velocity_bins; args.velocity_bins = None
        cfg = apply_overrides(cfg)
        args.velocity_bins = saved_vb
        if samples and ('vel_cont' not in samples[0]['target']
                        or 'ts_ids' not in samples[0]['target']
                        or 'iois_same' not in samples[0]['target']
                        or 'tempo_norm' not in samples[0]['target']):
            print(f"[*] Upgrading {len(samples):,} older-cache samples with "
                  f"regression/pitch target arrays...")
            tok = Tokenizer(cfg)
            it = tqdm.tqdm(samples, desc="Upgrading cache") if HAS_TQDM else samples
            for s in it:
                ev = s.get('events')
                if ev is not None:
                    s['target'] = tok.events_to_arrays(ev)
                    s['input']  = tok.events_to_arrays(tok.quantize_events(ev))
        return cfg, samples, meta

    if args.mode == 'train':
        cfg, samples, train_meta = _load_training_data(cfg)
        train(cfg, samples, args.run_name, resume=args.resume, meta=train_meta,
             data_fraction=args.data_fraction if args.data_fraction is not None else 1.0,
             use_tensorboard=not args.no_tensorboard)

    elif args.mode == 'grid_search':
        cfg, samples, train_meta = _load_training_data(cfg)
        bs_list = [int(x) for x in args.grid_batch_sizes.split(',')] if args.grid_batch_sizes else None
        lr_list = [float(x) for x in args.grid_lrs.split(',')] if args.grid_lrs else None
        ms_list = [x.strip() for x in args.grid_model_sizes.split(',')] if args.grid_model_sizes else None
        grid_search(cfg, samples, meta=train_meta,
                   batch_sizes=bs_list, lrs=lr_list, model_sizes=ms_list,
                   run_prefix=args.run_name or "grid", data_fraction=args.data_fraction,
                   use_tensorboard=not args.no_tensorboard,
                   auto_final_train=not args.no_grid_final_train,
                   final_epoch_multiplier=args.grid_final_epoch_multiplier,
                   final_run_name=args.grid_final_run_name)

    elif args.mode == 'infer':
        if not args.checkpoint:
            if os.path.exists(DEFAULT_PRETRAINED_CHECKPOINT):
                args.checkpoint = DEFAULT_PRETRAINED_CHECKPOINT
                print(f"[note] no --checkpoint given - using the bundled pretrained "
                      f"model at '{DEFAULT_PRETRAINED_CHECKPOINT}'")
            else:
                p.error(f"--checkpoint is required for infer mode (no bundled "
                        f"pretrained model found at '{DEFAULT_PRETRAINED_CHECKPOINT}' "
                        f"either - run --mode grid_search first, or pass --checkpoint "
                        f"explicitly)")
        if not args.input:
            p.error("--input is required for infer mode")
        if args.model_size is not None:
            print(f"[note] --model_size is ignored at inference; the architecture "
                  f"is read from the checkpoint so it always matches training.")
        # gentle guard: extrapolation is supported, but flag clearly absurd values
        for label, val in (('strength', args.strength),
                           ('strength_velocity', args.strength_velocity),
                           ('strength_timing', args.strength_timing)):
            if val is not None and (val < 0 or val > 3.0):
                print(f"[warning] --{label}={val} is outside the sane 0–2 range; "
                      f"expect heavy saturation/clipping or inverted feel.")
        args.preserve_grid_distance = float(np.clip(args.preserve_grid_distance, 0.0, 1.0))
        args.preserve_floor = float(np.clip(args.preserve_floor, 0.0, 1.0))
        args.preserve_velocity_dynamics = float(np.clip(args.preserve_velocity_dynamics, 0.0, 1.0))
        args.preserve_velocity_floor = float(np.clip(args.preserve_velocity_floor, 0.0, 1.0))
        # intensity convenience: accept 0..1 OR a raw MIDI velocity 1..127
        intensity = args.intensity
        if intensity is not None and intensity > 1.0:
            intensity = intensity / 127.0
        humanize_file(args.checkpoint, args.input, args.output,
                      temperature_vel=args.temperature_vel,
                      temperature_off=args.temperature_off,
                      strength=args.strength, top_k=args.top_k,
                      vel_decode=args.vel_decode, off_decode=args.off_decode,
                      strength_velocity=args.strength_velocity,
                      strength_timing=args.strength_timing,
                      preserve_grid_distance=args.preserve_grid_distance,
                      preserve_floor=args.preserve_floor,
                      top_k_velocity=args.top_k_velocity,
                      top_k_timing=args.top_k_timing,
                      preserve_velocity_dynamics=args.preserve_velocity_dynamics,
                      preserve_velocity_floor=args.preserve_velocity_floor,
                      intensity=intensity,
                      context_overlap=float(np.clip(args.context_overlap, 0.0, 0.9)),
                      fast_hit_cap=args.fast_hit_cap,
                      fast_hit_ceiling=args.fast_hit_ceiling,
                      refine_passes=args.refine_passes,
                      ar_timing=(True if args.ar_timing else None),
                      ar_timing_weight=args.ar_timing_weight)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted] Stopped by user (Ctrl-C).")
        sys.exit(130)
    except SystemExit:
        raise   # argparse errors / explicit sys.exit - already reported
    except Exception as exc:
        # Last-resort catch-all: pinpoint the exact file:line and give the type/message,
        # plus the full traceback, so an otherwise-cryptic crash is debuggable.
        _report_error("fatal error (see traceback below for the full call chain)",
                      exc, fatal=True)
        sys.exit(1)
