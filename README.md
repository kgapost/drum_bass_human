# drum_bass_human

This project has tools that:
- Make drum MIDI sound more human (small changes in timing and volume, like a real drummer).
- Make a bass MIDI file match (sync to) the humanized drums.
- Find similar-sounding grooves in your MIDI library.
- Find song section changes (like verse, chorus) automatically.

## Files in this project

- `config.py` — Settings and default values used by `drum_bass_studio.py`.
- `drum_humanizer_v3.py` — A model that learns how a real drummer plays (timing and velocity) from your MIDI library. It can then add that human feel to a stiff (robotic) MIDI file. Modes: `cache`, `train`, `infer`.
- `drum_theme_segmentation.py` — Finds where a song changes section (like verse to chorus). Modes: `dataset`, `train`, `infer`.
- `find_similar_grooves.py` — Takes one groove (a short MIDI file) and searches your library for similar-sounding grooves (by rhythm, velocity, note density, tempo). Modes: `index`, `query`.
- `drum_bass_studio.py` — The main app, with a window (UI). It combines everything above: split a song into segments, humanize each segment, adjust the rush/drag feel, sync the bass, swap in a similar groove if you want, and render the final song. Sound only plays through Windows' built-in synth (see `MidiPlayer` in the file — Windows only).
- `parse_midi_library.py` — A separate tool that cleans up a big folder of MIDI files you downloaded or bought. It is not part of the main humanizer pipeline. It removes packs you don't want, deletes exact duplicate files, makes the folder structure flat and simple, and sorts long files by length.

## Step 1: Set up your computer

```bash
python3 -m venv dbh
source dbh/bin/activate
pip install -r requirements.txt
```

On Windows, use this instead:

```bash
python3 -m venv dbh
dbh\Scripts\activate.bat
pip install -r requirements.txt
```

Note about `tkinter`: `drum_bass_studio.py` needs `tkinter`, but you cannot install it with `pip`. It comes from your operating system.
- On Linux: if `import tkinter` fails, run `sudo apt install python3-tk`.
- On Windows: the normal Python installer from python.org (or `winget install Python.Python.3.12`) already includes `tkinter`. You don't need to do anything extra. (Only exception: Python from the Microsoft Store does NOT include it. If you use that, reinstall Python from python.org or winget, and check the box for "tcl/tk and IDLE" during install.)

## Step 2: Clean your MIDI library (optional, but a good first step)

If you have a big folder of downloaded or bought MIDI files, use `parse_midi_library.py` to clean it up before using it with the other tools. Full details are in Step 7 below. Quick example:

```bash
dbh\Scripts\activate.bat

# clean up (delete unwanted genres/duplicates) - dry run first, changes nothing yet
python parse_midi_library.py ".\data" --execute

# make the folder structure flat and simple
python parse_midi_library.py ".\data" --flatten --execute

# sort long files into _songs/ and _g48/
python parse_midi_library.py ".\data" --move-by-measures --execute

# sort the remaining 25-48 bar files into _g24/
python parse_midi_library.py ".\data" --move-g24 --execute

python parse_midi_library.py ".\data" --verify-midi --execute
python parse_midi_library.py ".\data" --erase-junk --execute
python parse_midi_library.py ".\data" --erase-ambiguous --execute
```

## Step 3: drum_theme_segmentation.py — find song sections

This tool has 3 modes. Run them in this order: `dataset`, then `train`, then `infer`.

```bash
# a) build training data from your MIDI library
python drum_theme_segmentation.py --mode dataset --data_dir ./data --cache ./cache/segments.pkl --num_samples 1000

# b) train the model
python drum_theme_segmentation.py --mode train --cache ./cache/segments.pkl --run_name seg_v1 --epochs 100 --windows_per_sample 8

# c) run on a real song and print where it thinks each section starts
#    (use the threshold number training suggests, not always 0.5)
python drum_theme_segmentation.py --mode infer --checkpoint ./checkpoints/seg_v1/best.pt --input my_song.mid --threshold 0.5
```

### How to get a better score (val_F1) from this model

Here is what helps the most, from biggest effect to smallest:

| Setting | Why it helps |
|---|---|
| `--windows_per_sample` (default 8) | One training example has about 9,000 notes, but the model only looks at `--max_seq_len` notes at a time. So it needs about 17 "looks" (windows) to see one whole example. With the old setting of 1 window per example, the model only saw about 5% of the data each epoch (training round). This setting fixes that for free — no need to rebuild the cache or use more disk space. |
| `--epochs` and `--early_stop_patience` (default 15) | The learning rate slowly goes down across all `--epochs`. Most of the improvement happens near the end, when the learning rate is low. If `--early_stop_patience` is too small, training stops too early, before that improvement happens. Set `--epochs` to the number you actually plan to run. |
| `--max_seq_len` (default 512) and `--batch_size` | To decide if a bar starts a new section, the model must compare it to the section before it. At 512 notes, the model can only see about 2 sections at once, so near a boundary it often only sees a small piece of what came before. Using 1024 lets it see about 4 sections — but this uses much more memory (the cost grows fast, not just double), so also lower `--batch_size` to 8 if you have a small graphics card (4GB). |
| `--d_model` and `--num_layers` | The default (128 / 3) is a small model (about 500,000 numbers) and trains fast on a GPU. You can safely try a bigger model: 256 / 6. |
| `--num_samples` (dataset mode) | Only raise this after trying the settings above — they help more for less cost. Every 1000 samples uses about 240MB of disk space. Whether more samples actually help depends on how many different song "families" are in your library (the tool prints this number when it builds the dataset). |
| `--pos_weight` (default 8.0) | This controls how much the model cares about rare "section start" notes. Check your training output: if precision is lower than recall, the model is guessing "section start" too often — try a lower number. |

After training finishes, the tool automatically tests many threshold values on the best checkpoint (model file) and tells you which one scores best. The score shown during training always uses a fixed threshold of 0.5, but that number is usually not the best choice, because "section start" notes are rare in the data. Use the number the tool suggests, not 0.5, when you run `--mode infer`.

## Step 4: drum_humanizer_v3.py — make MIDI sound human

3 modes, run in order: `cache`, then `train`, then `infer`.

```bash
# build a training cache from a folder of MIDI files (only needed once)
python drum_humanizer_v3.py --mode cache --data_dir "./data" --cache ./cache/samples.pkl

# train a model using that cache
# (--model_size small is recommended: 2.2x faster than the default "base" size,
#  with only a small drop in quality - see the size table below)
python drum_humanizer_v3.py --mode train --cache cache/samples.pkl --run_name v1 --model_size small --epochs 100

# quick test with fake data, no real MIDI files needed
python drum_humanizer_v3.py --mode train --synthetic --epochs 3 --run_name smoke

# use the trained model to humanize a MIDI loop
python drum_humanizer_v3.py --mode infer --checkpoint checkpoints/v1/best.pt --input my_loop.mid --output my_loop_human.mid --strength 0.85
```

### Pretrained model (try it now, without training)

You don't have to train your own model just to try this tool. A ready-made model is kept in the `pretrained/` folder:

- `pretrained/humanizer_best.pt` — the best model found so far. It only has the model's learned numbers (weights), not the full training state, so you can use it but not continue training it further. `pretrained/humanizer_metadata.json` has info about how it was trained.
- `--mode grid_search` automatically saves its best result here, replacing the old file each time.
- `--mode infer` and `drum_bass_studio.py` both use `pretrained/humanizer_best.pt` automatically if you don't give your own checkpoint with `--checkpoint` or the "Load..." button.

**This folder is NOT saved in git** — a file of about 250MB does not belong in git history (see `knowledge.md`). Instead, it is shared using Google Drive, and the script `download_pretrained.py` downloads it for you:

```bash
python download_pretrained.py             # download only if the file is missing
python download_pretrained.py --force     # download again, even if the file is already there
```

- `drum_bass_studio.py` runs this automatically when it starts. If `pretrained/humanizer_best.pt` is missing (for example, right after you download this project), it fetches the file before opening the window. After that first time, it finds the file already there and skips checking the internet — so only the very first startup is a little slower.
- This needs the `gdown` package (already listed in `requirements.txt`). A normal download request does not work well with Google Drive's big-file warning page — `gdown` handles that correctly.
- Shared folder link: https://drive.google.com/drive/folders/1Q7PnRZUZ5Xm1V9DnC1PX3jJHGW2d45Ye
  - The download matches the file by **name**, not by a fixed ID. So if someone uploads a better model with the same name, you get it automatically, with no code change needed.
  - Uploading a new file to this folder needs your own Google account with edit access. This is a manual step — nothing in this project does it for you.
- If `gdown` is not installed, or Google Drive cannot be reached, or nothing has been uploaded there yet: both `--mode infer` and the app just continue without a pretrained model. You then need to give your own checkpoint with `--checkpoint` or "Load...".

## Step 5: find_similar_grooves.py — find similar grooves

```bash
# a) build a search index from your MIDI library (do this once)
python find_similar_grooves.py --mode index --data_dir ./data \
       --cache ./cache/groove_index.pkl

# b) search: find grooves similar to one file
python find_similar_grooves.py --mode query --cache cache/groove_index.pkl \
       --query "/path/to/some_groove.mid" --top_k 15
```

## Step 6: Run the app (the main window)

```bash
# Drum + Bass Humanization Studio
# (needs trained checkpoints from drum_humanizer_v3.py and drum_theme_segmentation.py;
#  groove search needs an index from find_similar_grooves.py --mode index)
python drum_bass_studio.py
```

This just opens a window. Drag and drop your MIDI file(s) into it, or click to browse — no other steps needed. You also don't need to load the humanizer model by hand: the app downloads and picks the ready-made model automatically the first time you run it (see "Pretrained model" in Step 4).

## Step 7: parse_midi_library.py — clean up a big MIDI folder (full details)

This is a separate tool for cleaning a big folder of MIDI files you downloaded or bought. It is not part of the humanizer itself — it just keeps your raw MIDI files tidy before you use them with `drum_humanizer_v3.py --mode cache`. (Example folder from the original setup: `/media/kapost/Schemsis/data`, an external drive — use whatever folder path is correct on your own computer.)

It has **4 modes**. You pick one mode at a time using a flag.

By default, every mode is a **dry run**: it only prints what it *would* do and changes nothing. Add `--execute` to really change files. The rename/move modes also support `--preview N`, which shows N random example results instead of a full dry-run list.

**Options for every mode:**

| Option | What it means |
|---|---|
| `base_dir` (required, no flag needed) | The folder path to your MIDI library, for example `"/media/kapost/Schemsis/data"` |
| `--execute` | Really delete/rename/move files. Without this, nothing changes (dry run). |
| `--preview N` | (flatten / move-by-measures / move-g24 modes only) Show N random example results instead of a full list. This also means "dry run", unless you also add `--execute`. |
| `--seed N` | A number that controls the random preview results, so you can repeat the same preview again. |
| `--keep-format {sd3,ezd}` | (default mode only) When one groove exists in two plugin formats (Superior Drummer or EZdrummer/EZX), which one to keep. Default is `sd3`. |
| `--flatten` | Use flatten mode (see below). |
| `--move-by-measures` | Use move-by-measures mode (see below). |
| `--move-g24` | Use move-g24 mode (see below). |

### Mode 1: default (no extra flag) — cleanup

```bash
python parse_midi_library.py "/media/kapost/Schemsis/data"            # dry run
python parse_midi_library.py "/media/kapost/Schemsis/data" --execute  # really do it
```

Does 7 steps, in this order: delete the "ViR2" pack (not sure it's from a real drummer), delete some genres you don't want (punk, jungle, rave, cha cha, marcha/rancho, afrobeat, NWOBHM, EDM, trance, industrial), delete house-genre folders and the Groove Monkee Electronic pack, delete exact duplicate files (same content) — keeping the format you chose with `--keep-format` — remove folders that are now empty, delete every `header` marker file, then remove empty folders again. Special files like `header`, `Aversion`, `kitpieces`, `midiDB`, `.dummy`, and any 0-byte file are never removed by the duplicate check, because Toontrack/EZdrummer/BFD need their own copy in each pack folder to work correctly.

### Mode 2: `--flatten` — rename files into a simple Company/Genre structure

```bash
python parse_midi_library.py "/media/kapost/Schemsis/data" --flatten --preview 20
python parse_midi_library.py "/media/kapost/Schemsis/data" --flatten --execute
```

Moves every file from its old, deep, complicated path into a simple `Company/Genre/renamed_file.ext` path. Example:

```
data/210@GROOVE_MONKEE_BLUES/21@078 SLOW BLUES A/078 Slow Blues Hats (8) F1 S.mid
  -> data/GROOVE/SLOW BLUES A/groove_Slow_Blues_Hats_(8)_F1S.mid
```

It puts as much useful info from the old path into the new filename as it can, without repeating the same word twice (up to 4 folder levels, using short forms for common words, like `straight`->`s`, `variation`->`v`, `fills`->`f`). It never overwrites a file — if two files would get the same new name, it adds `_2`, `_3`, and so on. After `--execute`, it checks that the total number of files did not change.

### Mode 3: `--move-by-measures` — sort long files into `_songs/` and `_g48/`

```bash
python parse_midi_library.py "/media/kapost/Schemsis/data" --move-by-measures --preview 20
python parse_midi_library.py "/media/kapost/Schemsis/data" --move-by-measures --execute
```

Counts how many bars each `.mid`/`.midi` file has (using `pretty_midi`), then moves it to one of two new folders, checked in this order:
1. `_songs/` — if the old path anywhere contains the word "song" or "songs" (upper/lower case doesn't matter) AND the file is longer than 64 bars.
2. `_g48/` — if the file is longer than 48 bars (only checked if rule 1 did not match).

The old path is added into the new filename, so you don't lose the information about where the file came from.

### Mode 4: `--move-g24` — sort remaining 25-48 bar files into `_g24/`

```bash
python parse_midi_library.py "/media/kapost/Schemsis/data" --move-g24 --preview 20
python parse_midi_library.py "/media/kapost/Schemsis/data" --move-g24 --execute
```

Same idea, simpler: every `.mid`/`.midi` file that is **not already in `_songs/` or `_g48/`**, and is longer than 24 bars, moves into a new `_g24/` folder. Since mode 3 already moved everything over 48 bars, this mode only picks up files with 25 to 48 bars. If starting from scratch, run mode 3 first — but mode 4 always skips `_songs/` and `_g48/` anyway, just to be safe.

**Suggested order on a brand-new library:** mode 1 (cleanup) -> mode 2 (flatten) -> mode 3 (move-by-measures) -> mode 4 (move-g24). Every mode is a dry run by default, so it's safe to run each one first, read what it says, and add `--execute` only once you're happy with it.

## Quick summary: order of steps from zero

1. Set up the environment and install everything (Step 1).
2. Build a groove-search index (`find_similar_grooves.py --mode index`) if you want to use "find similar grooves".
3. Build the humanizer cache and train it (`drum_humanizer_v3.py` cache -> train) if you want fresh or better humanization — or just use the ready-made model (see Step 4).
4. Build the segmentation dataset and train it (`drum_theme_segmentation.py` dataset -> train) if you want fresh or better automatic section detection.
5. Open `drum_bass_studio.py` for the full workflow (it can also search for similar grooves), or use `find_similar_grooves.py --mode query` from the command line for a single one-off search.

## Extra notes (easy to miss, but useful)

**Caches and checkpoints are different things — don't mix them up:**

| Tool/mode that makes it | File it creates | What actually reads this file |
|---|---|---|
| `drum_humanizer_v3.py --mode cache` | `cache/samples.pkl` (raw training data) | only `drum_humanizer_v3.py --mode train` |
| `drum_humanizer_v3.py --mode train` | `checkpoints/<run_name>/best.pt` | `drum_humanizer_v3.py --mode infer` **and** `drum_bass_studio.py` |
| `find_similar_grooves.py --mode index` | `cache/groove_index.pkl` | `find_similar_grooves.py --mode query` **and** `drum_bass_studio.py`'s groove search |

- `drum_bass_studio.py` never builds a cache by itself. It only needs a trained **checkpoint** (a `.pt` file) from `drum_humanizer_v3.py`, and one from `drum_theme_segmentation.py`. You pick these with its "browse for checkpoint" buttons. If you haven't trained a model yet, there is nothing for Studio to load (unless you use the ready-made pretrained one — see Step 4).
- When you run `--mode infer`, `drum_humanizer_v3.py` reads the model's shape/size directly from the checkpoint file. You don't need to pass `--model_size` again.

**Some settings are locked in when you build the cache/index — changing them later does nothing until you rebuild:**
- `drum_humanizer_v3.py --mode cache`: `--no_quality_filter`, `--min_velocity_std/range`, `--min_offset_std/range` only apply when you build the cache. If you change them, you must rebuild `cache/samples.pkl`.
- `find_similar_grooves.py --mode index`: `--min_notes` and the per-instrument volume limits are locked into the index when you build it. To change them, run `--mode index` again — running `--mode query` again will NOT pick up the change.

**Other useful things to know:**
- `--synthetic` on `drum_humanizer_v3.py` lets you test that training works from start to finish, using fake data — no real MIDI library needed. This is useful to quickly check a code change before waiting for a real cache to build. `drum_theme_segmentation.py` has no `--synthetic` option (its data is already made from real MIDI loops, so it always needs a library). To test it quickly instead, use a small `--num_samples` and a low `--epochs`. Note: a small `--num_samples` is only for quick testing — it does not improve quality. See the table in Step 3 for what actually improves the score.
- `drum_humanizer_v3.py` also has a hidden mode, `--mode grid_search` (not shown in its own help text). It tries many combinations of `--grid_batch_sizes`, `--grid_model_sizes`, and `--grid_lrs` to find the best one.
- `--resume <checkpoint>` on both training tools continues training from a saved checkpoint file, instead of starting from zero.
- **`--max_seq_len` controls speed, not just a limit — picking too big a number can make training extremely slow.** Every training example is padded (filled with empty space) up to this length, and the cost of attention grows with the square of this number. On a library of 166,000 samples: the middle (median) sample has only 38 notes, 95% of samples have 176 notes or fewer, and only 0.10% have more than 1024 notes. So the old default of 1024 meant 93.9% of every batch was just empty padding — 16x wasted work in normal layers, and 123x wasted work in attention. It gets worse: with `max_seq_len=1024` and `--batch_size 32`, you need about 15.5GB of GPU memory. On a 4GB card, Windows does not clearly show an "out of memory" error — it silently spills over into regular computer memory (RAM), which is much slower. Training still runs, but each batch takes 17 seconds, and `nvidia-smi` wrongly shows 100% GPU use (it's stuck waiting, not doing real work). Measured on a GTX 1650 graphics card:

  | `max_seq_len` / `batch_size` | seconds per batch | samples per second | peak GPU memory |
  |---|---|---|---|
  | 1024 / 32 (old default) | 17.11 | 2 | 15,563 MB |
  | **256 / 32 (new default)** | **1.18** | **27** | **1,557 MB** |
  | 192 / 32 | 0.88 | 36 | 1,044 MB |
  | 128 / 64 | 1.02 | 63 | 1,153 MB |

  Trade-off: with 256, 98.5% of samples are used in full (not cut short). With 192, it's 96.2%. With 128, only 88%. Samples longer than the window get randomly shortened during training, and split into overlapping pieces during inference — so a smaller number is mostly a speed win, not a quality loss, as long as you don't cut too much. **This number is saved inside the cache file.** An existing cache keeps its old value. To use a new value, either pass `--max_seq_len 256` by hand, or rebuild the cache.

- **`--model_size` (`tiny`/`small`/`base`/`deep`/`deeper`/`huge`) controls speed a lot — but going bigger than `small` gives smaller and smaller extra benefit.** Measured on a GTX 1650 with `--batch_size 8 --max_seq_len 256`:

  | size | parameters (learned numbers) | ms per batch | samples per second | speed vs `base` | peak GPU memory |
  |---|---|---|---|---|---|
  | tiny | 0.9M | 55 | 145.2 | 4.94x faster | 98MB |
  | **small** | **2.4M** | **124** | **64.6** | **2.20x faster** | **178MB** |
  | base (default) | 5.8M | 272 | 29.4 | same speed | 326MB |
  | deep | 13.9M | 620 | 12.9 | 0.44x (slower) | 638MB |
  | deeper | 27.1M | 1205 | 6.6 | 0.23x (slower) | 1062MB |
  | very_deep | 37.7M | (not tested on this card) | - | - | - |
  | huge | 60.6M | 2678 | 3.0 | 0.10x (slower) | 1871MB |

  **`very_deep` is built to be deep, not just bigger.** It is the only option deeper than `huge` (20 layers vs 18 layers), while also being narrower (384 vs 512 "width"), so it has only about 37.7M parameters instead of 60.6M. Why: this library's ~166,000 training samples only have about 10.4M real note-events in total (median 38 notes per sample). A real test showed `base` (5.8M) at loss 8.2999 versus `deep` (13.9M) at 8.2949 — almost no difference (just noise). This means more parameters (width) was not the real problem — making the model *wider* mostly causes overfitting (memorizing instead of learning). Making it *deeper* instead lets it reason about things far apart in the song (like "is this the start of a build-up before a fill?") at a lower cost. Dropout is also raised to 0.25 to match. **Be honest: this same reasoning also means `very_deep` might not beat `deep` at all.** It is worth trying once in a sweep, but it is not a safe default choice, and it is the most expensive option in any sweep.

  `small` is the recommended one above: more than 2x faster than `base`, while still similar in size (unlike `tiny`, which is a real cut in ability — only 16% of `base`'s size, almost 5x faster, but a real quality trade-off, not just speed). None of these sizes come close to using all 4GB of GPU memory at this batch size — if you don't need a smaller model, you can instead just raise `--batch_size` to go faster. Whether `small`'s output actually *sounds* as good as `base`'s is something only your own ears can tell you — this test only measures speed and memory.

- **A big cache plus multiple DataLoader workers on Windows can crash with "MemoryError" before training even starts.** On Windows and macOS, each worker process gets its own full copy of the dataset, and that whole copy first has to be prepared in memory by the main process. Training and validation each use `--num_workers` workers, so the real memory cost is roughly `(2 x num_workers + 1)` times the cache size. Example: a 2.5GB cache with `--num_workers 4` would need about 21GB of memory, and crashes inside `w.start()` with an error message that does not clearly explain the real cause. Both training tools now check for this ahead of time and automatically switch to `num_workers=0` instead (load data in the main process, no copies needed), printing an explanation when they do. Since loading one item is already fast (simple array slicing), extra workers weren't helping much anyway. To force a specific worker count anyway, set `DBH_FORCE_WORKERS=1`. On Linux, this problem does not happen (it uses a cheaper way to share memory between processes, called `fork`).
- In `drum_theme_segmentation.py`, the **validation data windows are always the same** (fixed, evenly spaced), while training windows are random each time. This is on purpose: if validation also used random windows, the score (`val_F1`) would change randomly between epochs just from luck, not from the model actually getting better or worse. That would break both "save the best checkpoint" and "stop early" logic, because they would react to random noise instead of real improvement. Do not change the validation loader back to random windows.
- `find_similar_grooves.py --mode query --exclude_same_family` removes results that are just a small variation of your search file (like "Fill 1" vs "Fill 14"). Useful when the top result is basically the same file as what you searched for.
- `drum_bass_studio.py`'s play/preview sound only works on **Windows** (it uses the built-in Microsoft GS Wavetable Synth through `mido`/`python-rtmidi`). The window and all the matching/search logic work on any operating system — only the actual sound needs Windows.
- In `config.py`, each setting is labeled either `JUDGMENT CALL` (a guess based on experience — safe to change if you want) or `VERIFIED FINDING` / `HARD TECHNICAL CONSTRAINT` (based on real testing or a real limit — check why it's there before changing it).
- All training and inference commands automatically pick the best available hardware, in this order: `cuda` (NVIDIA GPU) -> `mps` (Apple GPU) -> `cpu`. You don't need to tell it to use your GPU — it happens automatically if `torch.cuda.is_available()` returns `True`.
- `pip install -r requirements.txt` installs the **CPU-only** version of `torch` by default. If you have an NVIDIA GPU, install the CUDA version instead (match the CUDA version to your graphics driver — check with `nvidia-smi`, then see https://pytorch.org/get-started/locally/ for the exact install command). For example:
  ```bash
  pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
  ```
  Check that it worked: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.
  On a 4GB-class GPU, lower `--batch_size` if training gives a CUDA "out of memory" error.
- If you use VS Code, you can set the `dbh` virtual environment as the default interpreter in your own local `.code-workspace` file, so a new VS Code terminal already has it active. This file is personal to your machine and is not saved in git.

## Example: full training commands

```bash
dbh\Scripts\activate.bat

tensorboard --logdir checkpoints
# then open this in your browser: http://localhost:6006

python drum_theme_segmentation.py --mode dataset --data_dir ./data --cache ./cache/segments.pkl --num_samples 15000
python drum_theme_segmentation.py --mode train --cache ./cache/segments.pkl --run_name seg_model --epochs 100 --windows_per_sample 8

python drum_humanizer_v3.py --mode cache --data_dir "./data" --cache ./cache/samples.pkl
python drum_humanizer_v3.py --mode train --cache cache/samples.pkl --run_name humanizer_final_v1 --model_size deep --batch_size 32 --lr 1.5e-4 --data_fraction 1.0 --epochs 100

python drum_humanizer_v3.py --mode grid_search --cache cache/samples.pkl --run_name grid --grid_model_sizes base,deep --grid_batch_sizes 32,16 --grid_lrs 7.5e-5,1.5e-4 --data_fraction 0.1 --epochs 3
```

## Fresh setup on a new Ubuntu machine

```bash
python3 -m venv dbh
source dbh/bin/activate
pip install -r requirements.txt

pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python drum_humanizer_v3.py --mode cache --data_dir ./data --cache cache/samples.pkl --hop_bars 16 --section_bars 16

python drum_humanizer_v3.py --mode grid_search --cache cache/samples.pkl --run_name grid --grid_model_sizes huge,very_deep --grid_batch_sizes 128,64 --grid_lrs 8e-4,4e-4 --data_fraction 0.25 --epochs 20 --grid_final_epoch_multiplier 5
```
