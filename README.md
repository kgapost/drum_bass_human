(# drum_bass_human

Tools for humanizing drum MIDI (and syncing bass to it), finding similar
grooves in a library, and auto-detecting theme/section boundaries in a song.

## Files

- **config.py** - constants/defaults used by `drum_bass_studio.py`
- **drum_humanizer_v3.py** - Model that learns human drum feel (timing +
  velocity) from a MIDI library and applies it to stiff MIDI.
  (modes: `cache`, `train`, `infer`).
- **drum_theme_segmentation.py** - detects section boundaries are (modes `dataset`, `train`, `infer`).
- **find_similar_grooves.py** - given a query MIDI, ranks your library by how
  similar it feels (rhythm/velocity/density/tempo). (modes: `index`, `query`).
- **groove_finder_ui.py** - Tkinter desktop UI wrapper around
  `find_similar_grooves.py`.
  (Windows only - uses the built-in GS Wavetable synth for playback.)
- **drum_bass_studio.py** - main all-in-one app. Combines the humanizer +
  segmentation model + bass sync into one window
  segment, humanize each segment, tweak rush/drag, sync bass, render.
- **parse_midi_library.py** - standalone housekeeping tool for a large *external*
  MIDI sample library (not part of the humanizer pipeline itself -
  tidies up a folder of purchased/downloaded MIDI packs). Prunes unwanted
  genres, dedupes exact-duplicate files, flattens the folder structure, and
  sorts long files out by length.

## 1. Set up environment
```bash
python3 -m venv dbh
source dbh/bin/activate
pip install -r requirements.txt
```

or

```bash
python3 -m venv dbh
dbh\Scripts\activate.bat
pip install -r requirements.txt
```



Note: `tkinter` (needed by `groove_finder_ui.py` and `drum_bass_studio.py`) is
not in requirements.txt - it's not pip-installable, comes from the system.
On Linux, if `import tkinter` fails: `sudo apt install python3-tk`.
On Windows, the official python.org installer (and `winget install
Python.Python.3.12`) bundles Tcl/Tk by default, so `tkinter` just works out of
the box - no separate install step. (Only exception: Python from the
Microsoft Store excludes it: reinstall via python.org/winget with the "tcl/tk
and IDLE" optional feature checked.)

## 2. drum_theme_segmentation.py - dataset -> train -> infer
```bash
# a) build the (synthetic) training dataset from a MIDI library
python drum_theme_segmentation.py --mode dataset --data_dir ./data --cache ./cache/segments.pkl --num_samples 1000

# b) train
python drum_theme_segmentation.py --mode train --cache ./cache/segments.pkl --run_name seg_v1 --epochs 100 --windows_per_sample 8

# c) run on a real song, print predicted boundary measures
#    (use the threshold the training run's sweep recommended, not necessarily 0.5)
python drum_theme_segmentation.py --mode infer --checkpoint ./checkpoints/seg_v1/best.pt --input my_song.mid --threshold 0.5
```

### Getting a decent val_F1 out of the segmentation model

The knobs that actually moved the needle, roughly in order of payoff:

| Flag | Why it matters |
|---|---|
| `--windows_per_sample` (default 8) | A cached sample averages ~9k notes but one window only covers `--max_seq_len` of them, so it takes ~17 windows to tile one sample. At the old fixed 1-window-per-sample the model saw **~5% of the cache per epoch** - it was data-*starved*, not data-poor. This is free extra training signal: no cache rebuild, no extra disk. |
| `--epochs` / `--early_stop_patience` (default 15) | `OneCycleLR` anneals the LR across *all* `--epochs`, and most of the final gain is in that low-LR tail. Too-tight patience kills the run mid-schedule (seen: stopped at epoch ~50/100 with lr still at 2.2e-4, only 28% into the decay). Set `--epochs` to what you actually intend to run. |
| `--max_seq_len` (default 512) / `--batch_size` | To call a bar a boundary the model has to compare it against the **previous theme block**. At 512 notes a window spans only ~2x one block, so near a boundary it often sees just a fragment of what came before. 1024 gives ~4 blocks of context - but attention is O(n^2), so drop `--batch_size` to 8 alongside it on a 4GB card. |
| `--d_model` / `--num_layers` | Defaults (128 / 3) are ~500k params and epochs take seconds on a GPU. Plenty of headroom to go to 256 / 6. |
| `--num_samples` (dataset mode) | Only worth raising *after* the above - window coverage is the cheaper lever. Costs ~240MB of cache per 1000 samples. Whether more samples add genuinely new material depends on how many name-families your library has: the dataset step prints this (a large library can have tens of thousands, in which case there is a lot left to draw on). |
| `--pos_weight` (default 8.0) | Class-imbalance weight. Compare it against the real imbalance - if training shows precision **below** recall the model is over-predicting, so lower it. |

Every training run ends with a **validation threshold sweep** over the best
checkpoint. `val_F1` during training is scored at a fixed 0.5 cutoff, which is
rarely the F1-optimal operating point when positives are rare (~1 boundary per
17 measure-start notes) - the sweep reports the threshold that actually
maximizes F1, and that is the number to pass to `--threshold` at infer time.


## 3. drum_humanizer_v3.py - build cache -> train -> infer
```bash
# build a training cache from a folder of MIDI files (once)
python drum_humanizer_v3.py --mode cache --data_dir "./data" --cache ./cache/samples.pkl

# train a model on that cache (--model_size small - see the size-tradeoff note below
# for why: 2.2x faster than the 'base' default at a real capacity cost, benchmarked)
python drum_humanizer_v3.py --mode train --cache cache/samples.pkl --run_name v1 --model_size small --epochs 100

# quick smoke test with no real data:
python drum_humanizer_v3.py --mode train --synthetic --epochs 3 --run_name smoke

# humanize a loop with the trained checkpoint
python drum_humanizer_v3.py --mode infer --checkpoint checkpoints/v1/best.pt --input my_loop.mid --output my_loop_human.mid --strength 0.85
```



## 4. find_similar_grooves.py - index -> query

```bash
# a) index your MIDI library once
python find_similar_grooves.py --mode index --data_dir ./data \
       --cache ./cache/groove_index.pkl

# b) query: rank the library against one groove
python find_similar_grooves.py --mode query --cache cache/groove_index.pkl \
       --query "/path/to/some_groove.mid" --top_k 15
```

## 5. Running the UIs

```bash
# Groove Finder (needs a cache from find_similar_grooves.py --mode index first)
python groove_finder_ui.py

# Drum + Bass Humanization Studio (needs trained checkpoints from
# drum_humanizer_v3.py and drum_theme_segmentation.py)
python drum_bass_studio.py
```

Both just open a window - drag/drop or Browse for the MIDI file(s), no other
args needed.

## 6. parse_midi_library.py - external MIDI library housekeeping

A standalone script for cleaning up a large external folder of purchased/
downloaded MIDI packs (mine lives at `/media/kapost/Schemsis/data`, an
external drive - point it at wherever the equivalent folder is on this
machine). Not part of the humanizer pipeline - just keeps the raw MIDI
source library tidy before I feed any of it into `drum_humanizer_v3.py`'s
`--mode cache` step.

It has **four separate modes**, picked with a flag. Only one mode runs per
invocation. Every mode defaults to a **dry run** (prints what it would do,
changes nothing) - pass `--execute` to actually touch files. Renaming/moving
modes also support `--preview N` to sample N random results without a full
dry-run listing.

**Arguments (all modes):**

| Argument | Meaning |
|---|---|
| `base_dir` (positional, required) | Path to the library root, e.g. `"/media/kapost/Schemsis/data"` |
| `--execute` | Actually delete/rename/move files. Without it, every mode is a dry run. |
| `--preview N` | (flatten / move-by-measures / move-g24 only) Print N randomly sampled before -> after results instead of doing a full run. Implies dry run unless combined with `--execute`. |
| `--seed N` | Random seed for `--preview` sampling, so repeated previews are reproducible. |
| `--keep-format {sd3,ezd}` | (default mode only) Which plugin-format copy to keep when a groove was shipped for both Superior Drummer and EZdrummer/EZX. Default `sd3`. |
| `--flatten` | Switch to flatten mode (see below). |
| `--move-by-measures` | Switch to move-by-measures mode (see below). |
| `--move-g24` | Switch to move-g24 mode (see below). |

### Mode 1: default (no mode flag) - cleanup

```bash
python parse_midi_library.py "/media/kapost/Schemsis/data"            # dry run
python parse_midi_library.py "/media/kapost/Schemsis/data" --execute  # for real
```

Runs 7 phases in order: delete the ViR2 pack (unconfirmed real-drummer
provenance), delete specific unwanted genres (punk, jungle, rave, cha cha,
marcha/rancho, afrobeat, NWOBHM, EDM, trance, industrial), delete house-genre
folders + the Groove Monkee Electronic pack, dedupe exact-duplicate files by
content hash (keeping the `--keep-format` plugin edition, or the `@`-numbered
canonical folder for plain redundant copies), remove newly-empty folders,
delete every `header` marker file, remove newly-empty folders again.
Library-metadata marker files (`header`, `Aversion`, `kitpieces`, `midiDB`,
`.dummy`, and anything 0 bytes) are protected from the dedup step since
Toontrack/EZdrummer/BFD need their own local copy per pack folder to
recognize it as valid content.

### Mode 2: `--flatten` - rename into Company/Genre structure

```bash
python parse_midi_library.py "/media/kapost/Schemsis/data" --flatten --preview 20
python parse_midi_library.py "/media/kapost/Schemsis/data" --flatten --execute
```

Rewrites every file from its deep, numbered, "@"-riddled original path into
a flat `Company/Genre/renamed_file.ext` structure, e.g.:

```
data/210@GROOVE_MONKEE_BLUES/21@078 SLOW BLUES A/078 Slow Blues Hats (8) F1 S.mid
  -> data/GROOVE/SLOW BLUES A/groove_Slow_Blues_Hats_(8)_F1S.mid
```

Folds as much of the original path into the filename as it can without
repeating what's already implied (capped at 4 folder-lineage segments, with
cross-segment word dedup and a library of word abbreviations like
`straight`->`s`, `variation`->`v`, `fills`->`f`). Never overwrites - collisions
get an incrementing `_2`, `_3`, ... suffix. Verifies the total file count is
unchanged after `--execute`.

### Mode 3: `--move-by-measures` - sort long files into _songs/ and _g48/

```bash
python parse_midi_library.py "/media/kapost/Schemsis/data" --move-by-measures --preview 20
python parse_midi_library.py "/media/kapost/Schemsis/data" --move-by-measures --execute
```

Counts every `.mid`/`.midi` file's length in bars (via `pretty_midi`'s
downbeat detection) and moves it into one of two new top-level folders,
first match wins:
1. `_songs/` - "song" or "songs" appears anywhere in the file's old path
   (case-insensitive) AND it's longer than 64 bars.
2. `_g48/` - longer than 48 bars (checked only if #1 didn't match).

The old path is folded into the new filename so nothing about where a file
came from is lost once it's sitting in a flat folder.

### Mode 4: `--move-g24` - sort remaining 25-48 bar files into _g24/

```bash
python parse_midi_library.py "/media/kapost/Schemsis/data" --move-g24 --preview 20
python parse_midi_library.py "/media/kapost/Schemsis/data" --move-g24 --execute
```

Same idea, simpler: every `.mid`/`.midi` file **not already under `_songs/` or
`_g48/`** that's longer than 24 bars moves into a new top-level `_g24/` folder.
Since mode 3 already relocated everything over 48 bars, this only picks up
the 25-48 bar range. Run mode 3 first if starting from scratch - mode 4
explicitly excludes `_songs/` and `_g48/` from its scan either way.

**Suggested order on a fresh copy of the library:** mode 1 (cleanup) -> mode 2
(flatten) -> mode 3 (move-by-measures) -> mode 4 (move-g24). Each mode
defaults to a dry run, so it's safe to just run each one first and read the
output before adding `--execute`.

## Quick order of operations (from nothing)

1. Set up venv + install deps (step 1 above).
2. Build a groove-similarity index (`find_similar_grooves.py --mode index`) if
   I want to use Groove Finder.
3. Build the humanizer cache + train it (`drum_humanizer_v3.py` cache -> train)
   if I want fresh/better humanization.
4. Build the segmentation dataset + train it (`drum_theme_segmentation.py`
   dataset -> train) if I want fresh/better auto-segmentation.
5. Open `drum_bass_studio.py` for the actual humanize-a-song workflow, or
   `groove_finder_ui.py` just to find similar grooves.

## Notes (things that aren't obvious from the commands alone)

**Caches vs. checkpoints - these are NOT interchangeable:**

| Producer | File it makes | Who actually reads it |
|---|---|---|
| `drum_humanizer_v3.py --mode cache` | `cache/samples.pkl` (raw training data) | only `drum_humanizer_v3.py --mode train` |
| `drum_humanizer_v3.py --mode train` | `checkpoints/<run_name>/best.pt` | `drum_humanizer_v3.py --mode infer` **and** `drum_bass_studio.py` |
| `find_similar_grooves.py --mode index` (or Groove Finder's "Build Index" button) | `cache/groove_index.pkl` | `find_similar_grooves.py --mode query` **and** `groove_finder_ui.py` |

- `drum_bass_studio.py` never builds or touches a cache. It only needs a
  trained **checkpoint** (`.pt`) from `drum_humanizer_v3.py` and one from
  `drum_theme_segmentation.py`, picked via its "browse for checkpoint" buttons.
  If I haven't trained yet, Studio has nothing to load.
- `groove_finder_ui.py`'s "Build Index" button calls the exact same
  `build_index()` function as `find_similar_grooves.py --mode index` - it's
  literally the same `.pkl` format, just built through the GUI instead of the
  CLI. Either one can build it, either one can load it.
- At `infer` time, `drum_humanizer_v3.py` reads the model architecture
  straight out of the checkpoint file - no need to pass `--model_size` etc.
  again when humanizing.

**Rebuild triggers - some CLI flags are baked into the cache/index at build time,**
**not applied later at query/train time:**
- `drum_humanizer_v3.py --mode cache`: `--no_quality_filter`,
  `--min_velocity_std/range`, `--min_offset_std/range` only take effect when
  building the cache. Changing them later means rebuilding `cache/samples.pkl`.
- `find_similar_grooves.py --mode index`: `--min_notes` and the per-instrument
  velocity floors are baked into the index. Changing them means rebuilding
  with `--mode index` again - a `--mode query` re-run won't pick up the change.

**Other things worth remembering:**
- `--synthetic` on `drum_humanizer_v3.py` lets me smoke-test training
  end-to-end with fake data, no MIDI library needed - useful to sanity check
  a code change before waiting on a real cache build. `drum_theme_segmentation.py`
  has no `--synthetic` equivalent (its whole dataset is already synthesized from
  real loops, so it always needs a library); the fast-iteration knob there is a
  small `--num_samples` for a quick cache plus a low `--epochs`. Don't mistake
  a small `--num_samples` for the quality knob - see the segmentation tuning
  table above for what actually moves val_F1.
- `drum_humanizer_v3.py` also has a hidden `--mode grid_search` (not shown in
  its own usage examples) for sweeping `--grid_batch_sizes` /
  `--grid_model_sizes` / `--grid_lrs` combos.
- `--resume <checkpoint>` on both trainers continues training from a saved
  checkpoint instead of starting over.
- **`--max_seq_len` is a compute knob, not just a ceiling - and an oversized one
  is catastrophic on a small card.** EVERY sample is padded to it, and attention
  is O(n^2). Measured on a 166k-sample library: median sample is **38 notes**,
  p95 is 176, and only 0.10% exceed 1024 - so the old 1024 default made **93.9%
  of every batch pure padding** (16x wasted work in the linear layers, 123x in
  attention). Worse, at `max_seq_len=1024 --batch_size 32` the activations need
  ~15.5GB; on a 4GB card Windows WDDM does not hard-OOM, it silently spills to
  system RAM over PCIe, so training still "runs" - at 17 seconds per batch, with
  `nvidia-smi` showing a misleading 100% GPU utilization (thrashing, not math).
  Benchmarked on a GTX 1650:

  | `max_seq_len` / `batch_size` | s/batch | samples/s | peak VRAM |
  |---|---|---|---|
  | 1024 / 32 (old default) | 17.11 | 2 | 15,563 MB |
  | **256 / 32 (new default)** | **1.18** | **27** | **1,557 MB** |
  | 192 / 32 | 0.88 | 36 | 1,044 MB |
  | 128 / 64 | 1.02 | 63 | 1,153 MB |

  Coverage tradeoff: 256 leaves 98.5% of samples uncropped, 192 leaves 96.2%,
  128 only 88%. Samples longer than the window are randomly cropped in training
  and chunked with overlap-blending at inference, so a smaller value is a speed
  win rather than a quality loss - until the crop rate gets high enough to start
  truncating real phrases. **The value is baked into the cache**, so an existing
  cache keeps its old one: pass `--max_seq_len 256` explicitly, or rebuild.
- **`--model_size` (`tiny`/`small`/`base`/`deep`/`deeper`/`huge`) is a real speed
  lever, not just a quality knob - and the payoff drops off fast past `small`.**
  Benchmarked on a GTX 1650 at `--batch_size 8 --max_seq_len 256` (with the SDPA
  attention fusion in place):

  | size | params | ms/batch | samples/s | vs `base` | peak VRAM |
  |---|---|---|---|---|---|
  | tiny | 0.9M | 55 | 145.2 | 4.94x faster | 98MB |
  | **small** | **2.4M** | **124** | **64.6** | **2.20x faster** | **178MB** |
  | base (default) | 5.8M | 272 | 29.4 | 1.00x | 326MB |
  | deep | 13.9M | 620 | 12.9 | 0.44x | 638MB |
  | deeper | 27.1M | 1205 | 6.6 | 0.23x | 1062MB |
  | very_deep | 37.7M | (not benchmarked on the 1650) | - | - | - |
  | huge | 60.6M | 2678 | 3.0 | 0.10x | 1871MB |

  **`very_deep` is DEPTH-first, and deliberately not just "bigger".** It is the only
  preset that goes *deeper* than `huge` (20 layers vs 18) while staying *narrower*
  (d_model 384 vs 512), so it costs ~37.7M params instead of ~60.6M. The reasoning,
  from this library's own measurements: ~166k section-samples hold only ~10.4M
  supervised note-events (median 38 notes/sample), and a real sweep put `base`
  (5.8M) at val_loss 8.2999 against `deep` (13.9M) at 8.2949 - a 0.006 gap, i.e.
  noise. Capacity was not the binding constraint, so extra *width* (params grow
  ~quadratically in d_model) mostly buys overfitting; extra *depth* buys more
  rounds of relating distant hits at only ~linear param cost, which is what
  "is this a build / is this the bar before a fill" actually needs. Dropout rises
  to 0.25 to match. **Be honest about the odds though: the same evidence that
  motivates the shape also predicts it may not beat `deep` at all** - it is worth
  one sweep slot, not a default choice, and it is the most expensive combo in any
  grid it appears in.

  `small` is the pick used above: more than 2x faster than `base` while staying
  the same order of magnitude in parameters (unlike `tiny`, which is a genuinely
  smaller model at 16% of `base`'s capacity - nearly 5x faster, but a real
  capacity cut, not just a speed one). VRAM is not the constraint at ANY of these
  sizes on a 4GB card at this batch size - if a smaller model isn't warranted,
  raising `--batch_size` is a separate, still-available lever. Whether `small`'s
  humanization actually sounds as good as `base`'s is a quality question this
  benchmark can't answer - only listening to real output can.
- **Big cache + DataLoader workers on Windows = MemoryError before the first
  batch.** Windows/macOS start workers by `spawn`, so every worker gets a full
  *pickled copy* of the dataset, and the parent has to build that whole pickle
  buffer in RAM to hand over. Train and val each spawn `--num_workers`, so the
  real cost is about `(2*num_workers + 1)` x the cache. A 2.5GB cache with
  `--num_workers 4` projects to ~21GB and dies inside `w.start()` with a
  traceback pointing at `multiprocessing/reduction.py`, not at the real cause.
  Both trainers now measure this up front and fall back to `num_workers=0`
  (main-process loading, no pickling, no copies) with a printed explanation.
  `__getitem__` is numpy slicing in both, so workers were buying little anyway.
  Set `DBH_FORCE_WORKERS=1` to keep the configured count regardless. Linux
  `fork` shares those pages copy-on-write and is never downgraded.
- `drum_theme_segmentation.py`'s **validation windows are deterministic** (fixed,
  evenly-spaced crops) while training windows are random. This is deliberate: a
  `val_F1` measured on a different random slice each epoch isn't comparable
  epoch-to-epoch, which silently corrupts both "new best" checkpoint selection
  and early stopping (they end up reacting to sampling noise instead of to the
  model). Don't "simplify" the val loader back to random crops.
- `find_similar_grooves.py --mode query --exclude_same_family` filters out
  results whose filename is just a near-duplicate/variation of the query
  (e.g. "Fill 1" vs "Fill 14") - useful when the top match is trivially the
  same take as the query.
- `groove_finder_ui.py`'s audition/playback only makes real sound on
  **Windows** (it drives the built-in Microsoft GS Wavetable Synth through
  `mido`/`python-rtmidi`). It was built/tested in a headless Linux sandbox, so
  the UI and matching logic work everywhere, but actual audio needs Windows.
- `config.py` tags each constant as a `JUDGMENT CALL` (developer intuition,
  fine to retune by feel) vs. `VERIFIED FINDING` / `HARD TECHNICAL CONSTRAINT`
  (derived from something real - don't casually change without re-checking
  why it's there).
- All training/inference entry points already auto-select the best available
  device (`cuda` -> `mps` -> `cpu`), so nothing needs to be passed to use a
  GPU - it just happens if `torch.cuda.is_available()` is `True`.
- `pip install -r requirements.txt` installs a **CPU-only** `torch` by
  default. On an NVIDIA-GPU machine, install the CUDA build instead (match
  the CUDA version to your driver - check with `nvidia-smi`, then see
  https://pytorch.org/get-started/locally/ for the right `--index-url`), e.g.:
  ```bash
  pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
  ```
  Verify it took with `python -c "import torch; print(torch.cuda.is_available())"`.
  On a 4GB-class card, drop `--batch_size` if training hits a CUDA
  out-of-memory error.
- The `dbh` venv is set as this workspace's default interpreter (see
  `drum_bass_human.code-workspace`), so a fresh VS Code terminal should
  already have it active - no need to `source dbh/bin/activate` manually
  unless running from a plain shell outside VS Code.

```bash
dbh\Scripts\activate.bat

tensorboard --logdir checkpoints
http://localhost:6006

python drum_theme_segmentation.py --mode dataset --data_dir ./data --cache ./cache/segments.pkl --num_samples 15000
python drum_theme_segmentation.py --mode train --cache ./cache/segments.pkl --run_name seg_model --epochs 100 --windows_per_sample 8

python drum_humanizer_v3.py --mode cache --data_dir "./data" --cache ./cache/samples.pkl
python drum_humanizer_v3.py --mode train --cache cache/samples.pkl --run_name humanizer_final_v1 --model_size deep --batch_size 32 --lr 1.5e-4 --data_fraction 1.0 --epochs 100

python drum_humanizer_v3.py --mode grid_search --cache cache/samples.pkl --run_name grid --grid_model_sizes base,deep --grid_batch_sizes 32,16 --grid_lrs 7.5e-5,1.5e-4 --data_fraction 0.1 --epochs 3


```

## 7. Fresh setup on a new Ubuntu machine

```bash
python3 -m venv dbh
source dbh/bin/activate
pip install -r requirements.txt

pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python drum_humanizer_v3.py --mode cache --data_dir ./data --cache cache/samples.pkl --hop_bars 16 --section_bars 16

python drum_humanizer_v3.py --mode grid_search --cache cache/samples.pkl --run_name grid --grid_model_sizes huge,very_deep --grid_batch_sizes 128,64 --grid_lrs 8e-4,4e-4 --data_fraction 0.25 --epochs 20 --grid_final_epoch_multiplier 5

```





