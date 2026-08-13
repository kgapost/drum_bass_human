# drum_bass_human

Tools for humanizing drum MIDI (and syncing bass to it), finding similar
grooves in a library, and auto-detecting theme/section boundaries in a song.

## Files

- **config.py** - no code, just constants/defaults used by `drum_bass_studio.py`
  (slider defaults, thresholds, etc). Nothing to run here.
- **drum_humanizer_v3.py** - the core model. Learns human drum feel (timing +
  velocity) from a MIDI library and applies it to stiff/quantized MIDI.
  Has 3 modes: `cache`, `train`, `infer`.
- **drum_theme_segmentation.py** - detects where a long song's theme/section
  boundaries are (new pattern vs. repeat). Has 3 modes: `dataset`, `train`, `infer`.
- **find_similar_grooves.py** - given a query MIDI, ranks your library by how
  similar it feels (rhythm/velocity/density/tempo). Has 2 modes: `index`, `query`.
- **groove_finder_ui.py** - Tkinter desktop UI wrapper around
  `find_similar_grooves.py`. Drag a MIDI in, see ranked matches, audition them.
  (Windows only - uses the built-in GS Wavetable synth for playback.)
- **drum_bass_studio.py** - the main all-in-one app. Combines the humanizer +
  segmentation model + bass sync into one window: drop a song's drum+bass MIDI,
  segment it, humanize each segment, tweak rush/drag, sync bass, render.

## 1. Set up the environment (once)

```bash
python3 -m venv dbh
source dbh/bin/activate
pip install -r requirements.txt
```

Reactivate later with just:

```bash
source dbh/bin/activate
```

Note: `tkinter` (needed by `groove_finder_ui.py` and `drum_bass_studio.py`) is
not in requirements.txt - it's not pip-installable, comes from the system.
On Linux, if `import tkinter` fails: `sudo apt install python3-tk`.

## 2. drum_humanizer_v3.py - build cache -> train -> infer

```bash
# a) build a training cache from a folder of MIDI files (once)
python drum_humanizer_v3.py --mode cache --data_dir "/path/to/SD3/MIDI" \
       --cache cache/samples.pkl

# b) train a model on that cache
python drum_humanizer_v3.py --mode train --cache cache/samples.pkl \
       --run_name sd3_v1 --epochs 100

# quick smoke test with no real data:
python drum_humanizer_v3.py --mode train --synthetic --epochs 3 --run_name smoke

# c) humanize a loop with the trained checkpoint
python drum_humanizer_v3.py --mode infer --checkpoint checkpoints/sd3_v1/best.pt \
       --input my_loop.mid --output my_loop_human.mid --strength 0.85
```

## 3. drum_theme_segmentation.py - dataset -> train -> infer

```bash
# a) build the (synthetic) training dataset from a MIDI library
python drum_theme_segmentation.py --mode dataset --data_dir "/path/to/MIDI" \
       --cache cache/segments.pkl --num_samples 300

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
