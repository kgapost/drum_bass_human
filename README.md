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
