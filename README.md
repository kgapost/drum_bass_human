# drum_bass_human

Tools that make robotic MIDI drums sound like a real drummer played them, sync
a bass line to match, find similar-sounding grooves in a MIDI library, and
automatically detect where a song changes section (verse/chorus/etc).

**A few words you'll see everywhere in this doc:**
- **checkpoint** - a saved, trained model (a `.pt` file). You make one by
  training; other commands then *load* it to actually do something useful.
- **cache** - a `.pkl` file holding pre-processed data, built once from your
  MIDI library so training doesn't have to re-read/re-parse thousands of MIDI
  files every time you experiment.
- **epoch** - one full pass through the training data. Training runs many
  epochs in a row.
- **mode** - most scripts here do several different jobs; you pick which one
  with a `--mode` flag (e.g. `--mode train`).

## Files

- **config.py** - constants/defaults used by `drum_bass_studio.py`.
- **drum_humanizer_v3.py** - the main model. It learns what a human drummer's
  timing and velocity ("feel") sounds like from a MIDI library, then applies
  that feel to stiff, robotic MIDI. Has three modes: `cache` (prepare data),
  `train` (learn), `infer` (use it on a real file).
- **drum_theme_segmentation.py** - a model that detects where a song's
  section boundaries are (e.g. verse -> chorus). Modes: `dataset`, `train`,
  `infer`.
- **find_similar_grooves.py** - given one MIDI file, ranks your whole library
  by how similar each file feels (rhythm/velocity/density/tempo). Modes:
  `index`, `query`.
- **groove_finder_ui.py** - a desktop window (built with Tkinter) around
  `find_similar_grooves.py`, so you don't need the command line.
  (Windows only for audio playback - it uses the built-in GS Wavetable synth.)
- **drum_bass_studio.py** - the all-in-one app. Opens a window where you can:
  detect song sections, humanize each one, nudge the timing feel, sync a bass
  part to it, and render the result - combining the humanizer model and the
  segmentation model in one place.
- **parse_midi_library.py** - a separate cleanup tool for a big *external*
  folder of purchased/downloaded MIDI packs. It's not part of the humanizer
  pipeline - it just tidies your source library (removes unwanted genres,
  deletes exact duplicates, flattens messy folder structures, sorts long
  files by length) before you feed any of it into `drum_humanizer_v3.py`.

## 1. Set up your environment

A **venv** is an isolated Python install just for this project, so its
packages don't clash with anything else on your machine.

```bash
python3 -m venv dbh
source dbh/bin/activate
pip install -r requirements.txt
```

or, on Windows:

```bash
python3 -m venv dbh
dbh\Scripts\activate.bat
pip install -r requirements.txt
```

Note: `tkinter` (needed by `groove_finder_ui.py` and `drum_bass_studio.py`
for their windows) is not in requirements.txt - it can't be installed with
pip, it comes from your system's Python install.
On Linux, if `import tkinter` fails, run: `sudo apt install python3-tk`.
On Windows, the official installer from python.org (and
`winget install Python.Python.3.12`) already includes it, so nothing extra
is needed. The one exception is Python from the Microsoft Store, which
leaves it out - reinstall from python.org/winget and make sure the "tcl/tk
and IDLE" option is checked.

## 2. drum_theme_segmentation.py - build data, then train, then use it

```bash
# a) turn a folder of MIDI files into a training cache
python drum_theme_segmentation.py --mode dataset --data_dir ./data --cache ./cache/segments.pkl --num_samples 1000

# b) train the model on that cache
python drum_theme_segmentation.py --mode train --cache ./cache/segments.pkl --run_name seg_v1 --epochs 100 --windows_per_sample 8

# c) run it on a real song - prints which measures it thinks are section boundaries
#    (use the threshold number the training run recommended, not necessarily 0.5)
python drum_theme_segmentation.py --mode infer --checkpoint ./checkpoints/seg_v1/best.pt --input my_song.mid --threshold 0.5
```

### Getting a good score (val_F1) out of the segmentation model

`val_F1` is a 0-1 score measuring how well the model finds real boundaries
without also flagging false ones - higher is better. Here are the settings
that move that score the most, roughly in order of how much they help:

| Flag | Why it matters |
|---|---|
| `--windows_per_sample` (default 8) | Each cached sample has ~9,000 notes, but the model only looks at `--max_seq_len` notes at a time - so it takes about 17 "windows" to cover one whole sample. With the old setting of 1 window per sample, the model only ever saw about 5% of the cache each epoch - it wasn't short on data, it just wasn't *looking* at most of it. Raising this is free: same cache, same disk space, just more training signal used per epoch. |
| `--epochs` / `--early_stop_patience` (default 15) | The learning-rate schedule (`OneCycleLR`) is spread across your whole `--epochs` count, and most of the final improvement happens near the end, once the learning rate has dropped low. If `--early_stop_patience` is too tight, training can stop partway through that schedule - one run stopped at epoch ~50 of 100, with the learning rate still relatively high, missing most of its planned improvement. Set `--epochs` to what you actually intend to run. |
| `--max_seq_len` (default 512) / `--batch_size` | To recognize "this measure starts a new section," the model needs to compare it against the *previous* section. At 512 notes per window, that's only about 2 sections' worth of context, so near a boundary the model often only sees a fragment of what came before. 1024 gives about 4 sections of context - but the compute cost of "attention" (how the model relates notes to each other) grows much faster than the window size (quadratically), so pair a bigger window with a smaller `--batch_size` (try 8) if you're on a 4GB graphics card. |
| `--d_model` / `--num_layers` | The defaults (128 / 3) make a small model (~500k parameters) that trains in seconds per epoch on a GPU - there's plenty of room to try 256 / 6 if you want more capacity. |
| `--num_samples` (dataset mode) | Only worth raising *after* trying the settings above - giving the model more of what it already has (via `--windows_per_sample`) is the cheaper win first. Costs about 240MB of extra cache per 1,000 samples. Whether more samples actually add new material depends on how varied your MIDI library is; the dataset-build step prints how many distinct "families" of grooves it found. |
| `--pos_weight` (default 8.0) | Boundaries are rare (about 1 in every 17 candidate spots), so this weight tells the model to pay extra attention to catching them. If training shows the model is more often wrong-when-it-guesses-boundary than missing real ones (precision below recall), lower this number. |

Every training run finishes by testing a range of decision thresholds on
the validation set. `val_F1` shown *during* training always uses a fixed
0.5 cutoff, which usually isn't the best cutoff when true boundaries are
rare - the sweep at the end reports whichever threshold actually scores
best, and that's the number to pass to `--threshold` when running `infer`.


## 3. drum_humanizer_v3.py - build a cache, train, then humanize

```bash
# build a training cache from a folder of MIDI files (do this once)
python drum_humanizer_v3.py --mode cache --data_dir "./data" --cache ./cache/samples.pkl

# train a model on that cache
# (--model_size small: measured 2.2x faster than the 'base' default, for a real
# but modest capacity cost - see the size-tradeoff table further down)
python drum_humanizer_v3.py --mode train --cache cache/samples.pkl --run_name v1 --model_size small --epochs 100

# quick sanity check with made-up data, no MIDI library needed:
python drum_humanizer_v3.py --mode train --synthetic --epochs 3 --run_name smoke

# humanize a MIDI loop using your trained checkpoint
python drum_humanizer_v3.py --mode infer --checkpoint checkpoints/v1/best.pt --input my_loop.mid --output my_loop_human.mid --strength 0.85
```

### Pretrained model (skip training and just try it out)

`--mode grid_search` and `drum_bass_studio.py` both know how to find a
`pretrained/` folder holding the current best humanizer checkpoint, so you
can try the tool without training one yourself first:

- `pretrained/humanizer_best.pt` holds the winning run's trained weights and
  its architecture settings only - no optimizer state, so it's ready for
  humanizing but can't be used to *resume* training.
  `pretrained/humanizer_metadata.json` records which training run produced
  it, its score (val_loss), and a fingerprint of the data/settings it's
  comparable against.
- Every time `--mode grid_search` finishes, it copies its winning run here
  automatically, overwriting whatever was there before.
- `--mode infer` and `drum_bass_studio.py` both automatically use
  `pretrained/humanizer_best.pt` whenever you don't explicitly pass
  `--checkpoint` / click "Load...".

**This folder is intentionally not committed to git** - a ~250MB checkpoint
doesn't belong in git history (see `knowledge.md` for why). Instead it's
shared via a Google Drive folder, and `download_pretrained.py` fetches it
for you:

```bash
python download_pretrained.py             # fetch only if it's missing locally
python download_pretrained.py --force     # re-fetch even if already present
```

- `drum_bass_studio.py` calls this automatically when it starts up: if
  `pretrained/humanizer_best.pt` isn't there yet (e.g. right after cloning
  this repo on a new machine), it downloads it before the window opens.
  Every later launch finds the file already there, so this download only
  happens once.
- This needs the `gdown` package (already in `requirements.txt`) - a plain
  web request to a Google Drive file URL gets blocked by Google's
  virus-scan warning page for files this large; `gdown` knows how to get
  past that.
- Shared folder: https://drive.google.com/drive/folders/1Q7PnRZUZ5Xm1V9DnC1PX3jJHGW2d45Ye
  - It's matched by **filename** inside that folder, not a fixed file ID -
    so replacing the file there with a better model is picked up
    automatically, with no code change needed.
  - Uploading a new model to that folder requires a real Google account
    with write access - it's a manual step, nothing in this repo automates
    it.
- If `gdown` isn't installed, Drive is unreachable, or nothing's been
  uploaded there yet, both `--mode infer` and the UI just fall back to "no
  pretrained model" and expect you to pass `--checkpoint` / click "Load..."
  yourself - this is a convenience feature, not something the tools
  require to work.


## 4. find_similar_grooves.py - index, then query

```bash
# a) index your MIDI library once
python find_similar_grooves.py --mode index --data_dir ./data \
       --cache ./cache/groove_index.pkl

# b) query: rank the library by similarity to one groove
python find_similar_grooves.py --mode query --cache cache/groove_index.pkl \
       --query "/path/to/some_groove.mid" --top_k 15
```

## 5. Running the UIs

```bash
# Groove Finder (needs a cache built with find_similar_grooves.py --mode index first)
python groove_finder_ui.py

# Drum + Bass Humanization Studio (needs trained checkpoints from
# drum_humanizer_v3.py and drum_theme_segmentation.py)
python drum_bass_studio.py
```

Both just open a window - drag/drop or use Browse to pick your MIDI
file(s), no command-line arguments needed. Studio's humanizer model doesn't
need a manual "Load..." either - it automatically downloads and selects the
bundled pretrained model the first time you launch it (see "Pretrained
model" in section 3 above).

## 6. parse_midi_library.py - cleaning up an external MIDI library

A standalone script for cleaning up a large external folder of purchased/
downloaded MIDI packs (an example path used during development:
`/media/kapost/Schemsis/data`, on an external drive - point it at wherever
your equivalent folder lives). It's not part of the humanizer pipeline
itself, it just keeps the raw source library tidy before feeding any of it
into `drum_humanizer_v3.py --mode cache`.

It has **four separate modes**, chosen with a flag - only one mode runs per
invocation. Every mode defaults to a **dry run**: it prints what it *would*
do without changing anything, until you add `--execute`. The
renaming/moving modes also support `--preview N`, which samples N random
results instead of listing everything.

**Arguments (all modes):**

| Argument | Meaning |
|---|---|
| `base_dir` (positional, required) | Path to the library root, e.g. `"/media/kapost/Schemsis/data"` |
| `--execute` | Actually delete/rename/move files. Without it, every mode is a dry run (preview only). |
| `--preview N` | (flatten / move-by-measures / move-g24 only) Show N random before -> after examples instead of a full listing. Implies dry run unless combined with `--execute`. |
| `--seed N` | Random seed for `--preview` sampling, so repeated previews show the same sample. |
| `--keep-format {sd3,ezd}` | (default mode only) When a groove was shipped for both Superior Drummer and EZdrummer/EZX, which format to keep. Default `sd3`. |
| `--flatten` | Switch to flatten mode (see below). |
| `--move-by-measures` | Switch to move-by-measures mode (see below). |
| `--move-g24` | Switch to move-g24 mode (see below). |

### Mode 1: default (no mode flag) - cleanup

```bash
python parse_midi_library.py "/media/kapost/Schemsis/data"            # dry run
python parse_midi_library.py "/media/kapost/Schemsis/data" --execute  # for real
```

Runs 7 steps in order: delete the ViR2 pack (its real-drummer origin
couldn't be confirmed), delete specific unwanted genres (punk, jungle,
rave, cha cha, marcha/rancho, afrobeat, NWOBHM, EDM, trance, industrial),
delete house-genre folders plus the Groove Monkee Electronic pack, remove
exact duplicate files (same content) while keeping the `--keep-format`
edition (or the canonical numbered folder, for plain redundant copies),
remove any now-empty folders, delete every `header` marker file, then
remove now-empty folders again. Marker files that plugins use to recognize
a valid pack (`header`, `Aversion`, `kitpieces`, `midiDB`, `.dummy`, and any
0-byte file) are protected from the duplicate-removal step, since
Toontrack/EZdrummer/BFD each need their own copy per pack folder.

### Mode 2: `--flatten` - rename into a Company/Genre structure

```bash
python parse_midi_library.py "/media/kapost/Schemsis/data" --flatten --preview 20
python parse_midi_library.py "/media/kapost/Schemsis/data" --flatten --execute
```

Rewrites every file from its deep, numbered, `@`-riddled original path into
a flat `Company/Genre/renamed_file.ext` structure, e.g.:

```
data/210@GROOVE_MONKEE_BLUES/21@078 SLOW BLUES A/078 Slow Blues Hats (8) F1 S.mid
  -> data/GROOVE/SLOW BLUES A/groove_Slow_Blues_Hats_(8)_F1S.mid
```

It folds as much of the original folder path into the new filename as it
can, without repeating information already implied elsewhere (limited to
the last 4 folder levels, with duplicate words removed and some words
abbreviated, e.g. `straight`->`s`, `variation`->`v`, `fills`->`f`). It never
overwrites a file - if two results would collide, it adds `_2`, `_3`, etc.
It also double-checks the total file count is unchanged after `--execute`.

### Mode 3: `--move-by-measures` - sort long files into _songs/ and _g48/

```bash
python parse_midi_library.py "/media/kapost/Schemsis/data" --move-by-measures --preview 20
python parse_midi_library.py "/media/kapost/Schemsis/data" --move-by-measures --execute
```

Measures every `.mid`/`.midi` file's length in bars (using `pretty_midi`'s
downbeat detection) and moves it into one of two new top-level folders,
first match wins:
1. `_songs/` - if "song" or "songs" appears anywhere in the file's old path
   (case-insensitive) AND it's longer than 64 bars.
2. `_g48/` - if it's longer than 48 bars (only checked if #1 didn't match).

The old path is folded into the new filename, so nothing about where a
file came from is lost once it's sitting in a flat folder.

### Mode 4: `--move-g24` - sort the remaining 25-48 bar files into _g24/

```bash
python parse_midi_library.py "/media/kapost/Schemsis/data" --move-g24 --preview 20
python parse_midi_library.py "/media/kapost/Schemsis/data" --move-g24 --execute
```

Same idea, simpler: every `.mid`/`.midi` file that isn't already under
`_songs/` or `_g48/` and is longer than 24 bars moves into a new top-level
`_g24/` folder. Since mode 3 already moved everything over 48 bars, this
only picks up the 25-48 bar range. Run mode 3 first if starting from
scratch - mode 4 always skips `_songs/` and `_g48/` regardless.

**Suggested order on a fresh copy of the library:** mode 1 (cleanup) ->
mode 2 (flatten) -> mode 3 (move-by-measures) -> mode 4 (move-g24). Every
mode defaults to a dry run, so it's safe to just run each one first and
read its output before adding `--execute`.

## Quick order of operations (starting from nothing)

1. Set up the venv and install dependencies (section 1 above).
2. Build a groove-similarity index (`find_similar_grooves.py --mode index`)
   if you want to use Groove Finder.
3. Build the humanizer cache and train it (`drum_humanizer_v3.py`: cache ->
   train) if you want fresh or better humanization.
4. Build the segmentation dataset and train it (`drum_theme_segmentation.py`:
   dataset -> train) if you want fresh or better auto-segmentation.
5. Open `drum_bass_studio.py` for the actual humanize-a-song workflow, or
   `groove_finder_ui.py` just to find similar grooves.

## Notes (things that aren't obvious from the commands alone)

**Caches vs. checkpoints - these are NOT interchangeable files:**

| Producer | File it makes | Who reads it |
|---|---|---|
| `drum_humanizer_v3.py --mode cache` | `cache/samples.pkl` (raw training data) | only `drum_humanizer_v3.py --mode train` |
| `drum_humanizer_v3.py --mode train` | `checkpoints/<run_name>/best.pt` (trained model) | `drum_humanizer_v3.py --mode infer` **and** `drum_bass_studio.py` |
| `find_similar_grooves.py --mode index` (or Groove Finder's "Build Index" button) | `cache/groove_index.pkl` | `find_similar_grooves.py --mode query` **and** `groove_finder_ui.py` |

- `drum_bass_studio.py` never builds or touches a cache. It only needs an
  already-trained **checkpoint** (`.pt` file) from `drum_humanizer_v3.py`
  and one from `drum_theme_segmentation.py`, picked with its "browse for
  checkpoint" buttons. If you haven't trained anything yet, Studio has
  nothing to load.
- `groove_finder_ui.py`'s "Build Index" button runs the exact same code as
  `find_similar_grooves.py --mode index` - it produces the identical `.pkl`
  format, just triggered from the window instead of the command line.
  Either one can build the index file, either one can load it.
- When you run `infer`, `drum_humanizer_v3.py` reads the model's
  architecture straight out of the checkpoint file, so you don't need to
  pass `--model_size` etc. again just to humanize something.

**Some flags are "baked in" at build time and won't apply later** if you
change them and just re-run a different mode - you have to rebuild:
- `drum_humanizer_v3.py --mode cache`: `--no_quality_filter`,
  `--min_velocity_std/range`, `--min_offset_std/range` only take effect
  while building the cache. Changing them later means rebuilding
  `cache/samples.pkl` from scratch.
- `find_similar_grooves.py --mode index`: `--min_notes` and the
  per-instrument velocity floors are baked into the index file. Changing
  them means rebuilding with `--mode index` again - just re-running
  `--mode query` won't pick up the change.

**Other things worth remembering:**
- `--synthetic` on `drum_humanizer_v3.py` lets you smoke-test training
  end-to-end with made-up data, no MIDI library needed - handy for
  checking a code change works before waiting on a real cache build.
  `drum_theme_segmentation.py` has no `--synthetic` equivalent (its
  training data is already generated from real loops, so it always needs a
  library) - there, the fast-iteration trick is a small `--num_samples` for
  a quick cache plus a low `--epochs`. Don't mistake a small
  `--num_samples` for the main quality lever though - see the segmentation
  tuning table above for what actually moves `val_F1`.
- `drum_humanizer_v3.py` also has a hidden `--mode grid_search` (not shown
  in its own built-in usage examples) for automatically trying different
  combinations of `--grid_batch_sizes` / `--grid_model_sizes` /
  `--grid_lrs` and comparing the results.
- `--resume <checkpoint>` on both trainers continues training from a saved
  checkpoint instead of starting from scratch.
- **`--max_seq_len` controls how much compute training uses, not just a
  ceiling on input size - and setting it too high can quietly wreck
  performance on a small graphics card.** Every training sample gets padded
  out to this length, and the "attention" mechanism's compute cost grows
  roughly with the *square* of this number, not linearly. Measured on a
  166k-sample library: the typical sample is only **38 notes**, 95% of
  samples are under 176 notes, and only 0.10% exceed 1024 notes - so the
  old default of 1024 meant **93.9% of every batch was pure padding**
  (wasted work: 16x more than needed in the simple layers, 123x more in
  attention). Worse, at `max_seq_len=1024, batch_size=32`, the model needs
  about 15.5GB of GPU memory to run - on a 4GB card, Windows doesn't fail
  loudly (an "out of memory" error), it silently spills the extra memory
  over to system RAM, so training still appears to run, just at 17 seconds
  per batch, with `nvidia-smi` misleadingly showing 100% GPU usage (it's
  thrashing, not actually computing). Benchmarked on a GTX 1650:

  | `max_seq_len` / `batch_size` | seconds/batch | samples/second | peak GPU memory |
  |---|---|---|---|
  | 1024 / 32 (old default) | 17.11 | 2 | 15,563 MB |
  | **256 / 32 (new default)** | **1.18** | **27** | **1,557 MB** |
  | 192 / 32 | 0.88 | 36 | 1,044 MB |
  | 128 / 64 | 1.02 | 63 | 1,153 MB |

  The tradeoff: at 256, 98.5% of samples fit without being cut short; at
  192 it's 96.2%; at 128 it drops to 88%. Samples longer than the window
  get randomly cropped during training and stitched back together with
  overlap-blending at inference time, so a smaller value is mostly a speed
  win rather than a quality loss - until the crop rate gets high enough to
  start cutting off real musical phrases. **This value gets saved into the
  cache file itself**, so an existing cache keeps whatever value it was
  built with - pass `--max_seq_len 256` explicitly at train time, or
  rebuild the cache, to actually change it.
- **`--model_size` (`tiny`/`small`/`base`/`deep`/`deeper`/`huge`) is a real
  speed lever, not just a quality knob - and the benefit of going bigger
  drops off fast past `small`.** Benchmarked on a GTX 1650 at
  `--batch_size 8 --max_seq_len 256`:

  | size | parameters | ms/batch | samples/second | vs `base` | peak GPU memory |
  |---|---|---|---|---|---|
  | tiny | 0.9M | 55 | 145.2 | 4.94x faster | 98MB |
  | **small** | **2.4M** | **124** | **64.6** | **2.20x faster** | **178MB** |
  | base (default) | 5.8M | 272 | 29.4 | 1.00x (baseline) | 326MB |
  | deep | 13.9M | 620 | 12.9 | 0.44x (slower) | 638MB |
  | deeper | 27.1M | 1205 | 6.6 | 0.23x (slower) | 1062MB |
  | very_deep | 37.7M | (not benchmarked on the 1650) | - | - | - |
  | huge | 60.6M | 2678 | 3.0 | 0.10x (slower) | 1871MB |

  **`very_deep` prioritizes depth over width, deliberately - it's not just
  "bigger."** It's the only preset that has *more layers* than `huge` (20
  vs 18) while each layer is *narrower* (384 vs 512 dimensions), landing at
  about 37.7M parameters instead of `huge`'s 60.6M. The reasoning: with
  only ~166k training samples (about 10.4M individual note-events total,
  median 38 notes/sample), a real comparison found `base` (5.8M params) and
  `deep` (13.9M params) scored almost identically (val_loss 8.2999 vs
  8.2949 - a gap so small it's basically noise). That suggests model size
  wasn't the bottleneck, so making the model *wider* (which grows roughly
  with the square of that width) mostly risks overfitting, while making it
  *deeper* (which only grows about linearly) buys more chances to relate
  distant notes to each other - closer to what's actually needed to judge
  "is this the bar before a fill?" Dropout is also raised to 0.25 to
  compensate. **Worth being honest here: the same evidence behind this idea
  also means it might not beat `deep` at all** - it's worth trying once in
  a sweep, not a safe default, and it's the most expensive option in any
  sweep it's part of.

  `small` is the one actually recommended above: more than 2x faster than
  `base` while staying roughly the same order of magnitude in size (unlike
  `tiny`, which is a genuinely smaller model at only 16% of `base`'s
  capacity - nearly 5x faster, but a real quality tradeoff, not just
  speed). None of these sizes come close to using up a 4GB card's memory
  at this batch size - if a smaller model isn't the right call, raising
  `--batch_size` is a separate lever you can still reach for. Whether
  `small` actually *sounds* as good as `base` is a question this benchmark
  can't answer - only listening to real output can tell you that.
- **A big cache plus DataLoader worker processes on Windows can cause a
  MemoryError before training even starts.** On Windows/macOS, each worker
  process is started fresh (`spawn`) and gets handed its own full *copy* of
  the dataset in memory. Training and validation each start
  `--num_workers` of these, so the real memory cost is roughly
  `(2 x num_workers + 1)` times the size of the cache. A 2.5GB cache with
  `--num_workers 4` would need about 21GB and crashes while starting the
  worker processes, with an error message that doesn't obviously point to
  the real cause. Both trainers now check for this ahead of time and fall
  back to `num_workers=0` (loading data in the main process, no copies) with
  an explanation printed to the console. Since data loading here is just
  numpy array slicing, extra worker processes weren't buying much speed
  anyway. Set the environment variable `DBH_FORCE_WORKERS=1` if you want to
  keep your configured worker count regardless. Linux doesn't have this
  problem - it starts workers with `fork`, which shares memory pages
  instead of copying them.
- `drum_theme_segmentation.py`'s **validation samples are fixed** (the same
  evenly-spaced crops every time), while training samples are randomly
  chosen each epoch. This is intentional: if validation used a different
  random slice every epoch, the `val_F1` score wouldn't be comparable from
  one epoch to the next, which would corrupt both "is this the best
  checkpoint so far?" decisions and early stopping - they'd end up reacting
  to random noise instead of real improvement. Don't change the validation
  loader back to random crops.
- `find_similar_grooves.py --mode query --exclude_same_family` filters out
  results that are just a near-duplicate or variation of the query file by
  name (e.g. "Fill 1" vs "Fill 14") - useful when the top match is
  trivially the same take as what you searched for.
- `groove_finder_ui.py`'s audition/playback only produces real sound on
  **Windows** (it plays through the built-in Microsoft GS Wavetable Synth
  via `mido`/`python-rtmidi`). It was built and tested in a headless Linux
  environment, so the window and matching logic work everywhere, but you
  need Windows to actually hear anything.
- `config.py` labels each constant as a `JUDGMENT CALL` (a developer's
  intuition - fine to adjust by feel) vs. a `VERIFIED FINDING` or
  `HARD TECHNICAL CONSTRAINT` (backed by measurement or a real limitation -
  don't casually change these without checking why they're set that way).
- Every training/inference command already picks the best available device
  on its own (`cuda` -> `mps` -> `cpu`) - you don't need to pass anything
  to use a GPU, it just happens automatically if
  `torch.cuda.is_available()` is `True`.
- `pip install -r requirements.txt` installs a **CPU-only** build of
  `torch` by default. If you have an NVIDIA GPU, install the CUDA build
  instead (match the CUDA version to your driver - check with
  `nvidia-smi`, then see https://pytorch.org/get-started/locally/ for the
  right install command), e.g.:
  ```bash
  pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
  ```
  Confirm it worked with: `python -c "import torch; print(torch.cuda.is_available())"`.
  On a 4GB-class card, lower `--batch_size` if training fails with a CUDA
  out-of-memory error.
- The `dbh` venv is already set as this workspace's default Python
  interpreter (see `drum_bass_human.code-workspace`), so a fresh VS Code
  terminal should have it active automatically - no need to run
  `source dbh/bin/activate` yourself unless you're using a plain terminal
  outside VS Code.

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
