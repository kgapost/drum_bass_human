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

Note: `tkinter` (needed by `groove_finder_ui.py` and `drum_bass_studio.py`) is
not in requirements.txt - it's not pip-installable, comes from the system.
On Linux, if `import tkinter` fails: `sudo apt install python3-tk`.

## 2. drum_humanizer_v3.py - build cache -> train -> infer
```bash
# build a training cache from a folder of MIDI files (once)
python drum_humanizer_v3.py --mode cache --data_dir "/path/to/SD3/MIDI" --cache cache/samples.pkl

# train a model on that cache
python drum_humanizer_v3.py --mode train --cache cache/samples.pkl --run_name sd3_v1 --epochs 100

# quick smoke test with no real data:
python drum_humanizer_v3.py --mode train --synthetic --epochs 3 --run_name smoke

# humanize a loop with the trained checkpoint
python drum_humanizer_v3.py --mode infer --checkpoint checkpoints/sd3_v1/best.pt --input my_loop.mid --output my_loop_human.mid --strength 0.85
```

## 3. drum_theme_segmentation.py - dataset -> train -> infer

```bash
# a) build the (synthetic) training dataset from a MIDI library
python drum_theme_segmentation.py --mode dataset --data_dir "/path/to/MIDI" --cache cache/segments.pkl --num_samples 300

# b) train
python drum_theme_segmentation.py --mode train --cache cache/segments.pkl \
       --run_name seg_v1 --epochs 40

# c) run on a real song, print predicted boundary measures
python drum_theme_segmentation.py --mode infer --checkpoint checkpoints/seg_v1/best.pt \
       --input my_song.mid --threshold 0.5
```

## 4. find_similar_grooves.py - index -> query

```bash
# a) index your MIDI library once
python find_similar_grooves.py --mode index --data_dir "/path/to/MIDI" \
       --cache cache/groove_index.pkl

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
- `--synthetic` (on both `drum_humanizer_v3.py` and effectively via
  `--num_samples` on `drum_theme_segmentation.py`) lets me smoke-test training
  end-to-end with fake data, no MIDI library needed - useful to sanity check
  a code change before waiting on a real cache build.
- `drum_humanizer_v3.py` also has a hidden `--mode grid_search` (not shown in
  its own usage examples) for sweeping `--grid_batch_sizes` /
  `--grid_model_sizes` / `--grid_lrs` combos.
- `--resume <checkpoint>` on both trainers continues training from a saved
  checkpoint instead of starting over.
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
- This machine has no NVIDIA GPU (`torch.cuda.is_available()` is `False` -
  Intel iGPU only), so training runs on CPU. Works, just slow - budget more
  time for `--mode train` runs than a CUDA box would need.
- The `dbh` venv is set as this workspace's default interpreter (see
  `drum_bass_human.code-workspace`), so a fresh VS Code terminal should
  already have it active - no need to `source dbh/bin/activate` manually
  unless running from a plain shell outside VS Code.
