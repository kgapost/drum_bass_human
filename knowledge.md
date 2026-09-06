# Knowledge transfer: drum humanization project (drum_bass_human)

Written for a future Claude session working on a *different* problem, to carry
forward whatever generalizes from this one. Project-specific facts (MIDI, drums)
are kept short; the emphasis is on methodology, architecture patterns, and bugs/
gotchas that would recur on similar problems (sequence labelling, small-dataset
transformers, hyperparameter sweeps, long-running training pipelines).

---

## 1. The problem, in one paragraph

Take a *quantized* (robotic, on-grid, flat-velocity) MIDI drum performance and
predict how a human would actually play it: per-hit micro-timing offset and
velocity. This is **not** generation (the notes/pitches are fixed and never
touched) and **not** classification of a whole clip — it's dense, per-token
*regression-ish labelling* over a sequence, where every input token gets its own
two-part output (timing, velocity), conditioned on the whole surrounding pattern.
That framing — "every position needs a label, and the label depends on global
context" — is the part likely to recur in other problems (e.g. per-frame audio
labelling, per-character text correction, per-node graph labelling).

## 2. The architecture

**Encoder-only Transformer, bidirectional, one forward pass, no autoregression.**
The key early design decision (see `HISTORY OF OPTIMIZATIONS` item #1 in the
source) was realizing this is *labelling*, not *generation* — every hit needs a
label and all labels can be predicted in parallel from one pass over the whole
context. This is 10-100x cheaper than an autoregressive formulation and avoids
compounding-error drift, and it generalizes: **any time a task is "assign a label
to every position, using bidirectional context," check whether it actually needs
autoregression before reaching for it.**

- **Input embedding** (`DrumEventEmbedding`): each token embeds instrument,
  quantized velocity, absolute musical position (sinusoidal), bar number,
  inter-onset-interval (IOI) both globally and per-instrument, a *metric
  strength* class (downbeat/beat/8th/16th/off-grid — a categorical feature, not
  just raw position, so the model can learn "downbeats get X" directly), time
  signature id + position-within-measure, and flam/grace-note flags. All of
  these are concatenated then projected down (`Linear(d*11, d)`). Two more
  signals are injected as *residual additive* conditioning rather than
  concatenated: a global **intensity** scalar (overall loudness/genre context)
  and per-note **tempo** (micro-timing feel scales with tempo — a 15ms push is
  lazy at 80bpm, sloppy at 180bpm). **Lesson: decide per-feature whether it's a
  per-token identity feature (concatenate) or a global conditioning signal
  (inject as a residual/FiLM-style addition) — collapsing that distinction and
  concatenating everything wastes capacity on broadcasting a constant.**

- **Relative position bias, not absolute-only** (`RelativePositionBias`,
  T5-style): a learned scalar bias added to attention logits, keyed on the
  *bucketed signed distance* between query and key position (log-scaled buckets
  — fine resolution nearby, coarse far away). This generalizes to phrase
  lengths/start-offsets never seen during training, which absolute position
  embeddings alone don't. **Relevant any time inputs have variable length or
  content that's positionally translation-invariant** (a downbeat groove
  feels the same whether it starts at bar 0 or bar 40).

- **Fused attention**: replaced hand-rolled QK^T/softmax/@V with
  `F.scaled_dot_product_attention`, passing the relative-position bias as an
  additive float `attn_mask` (padding mask gets `masked_fill`'d into the same
  tensor). Measured 1.74x faster, ~45% less peak memory, numerically verified
  equivalent (max diff 1.8e-6) before trusting it. **Lesson: SDPA can absorb an
  arbitrary additive bias, not just a padding mask — don't assume a custom
  attention-bias term forces you to hand-roll attention.**

- **Two prediction paradigms, both kept, switchable via config**
  (`target_mode`): `classification` (binned output + **ordinal soft-target
  cross-entropy**: velocity 0-127 becomes a soft label distribution — a Gaussian
  centered on the true bin — so predicting bin 101 for a true value of 100 is
  barely penalized while bin 20 is heavily penalized; retains a full output
  distribution so sampling gives real variety) vs `regression` (continuous
  scalar heads, infinite resolution, but collapses to the mean on multi-modal
  data — "no timing" and "way behind" both round toward "half behind"). The
  code explicitly says to A/B both on your own data rather than assume one
  wins. **This ordinal-soft-target trick generalizes to any bucketed-but-really-
  continuous target** (age bins, price bins, angle bins) where adjacent bins
  are "almost right," not "equally wrong."

- **Model-size ladder is depth vs. width, not just "bigger."** Six presets
  (tiny→small→base→deep→deeper→very_deep→huge). `very_deep` deliberately goes
  *deeper* than `huge` (20 vs 18 layers) while staying *narrower* (d_model 384
  vs 512), because a real sweep here showed capacity wasn't the binding
  constraint (base 5.8M vs deep 13.9M: 0.006 val_loss gap, i.e. noise) — so
  extra *width* (params grow ~quadratically in d_model) mostly buys
  overfitting on a data-limited problem, while extra *depth* buys more rounds
  of relating distant tokens at ~linear cost. **When capacity isn't the
  bottleneck, prefer depth over width before concluding "needs a bigger
  model."** But the same source is honest that this reasoning "may not beat a
  plain deeper model at all" — it's a hypothesis worth one sweep slot, not a
  default.

## 3. The data / training-signal problems (arguably the hardest part)

These are not modeling problems — they're "is your supervision signal actually
teaching the right thing" problems, and they took more design effort than the
architecture:

- **Garbage-in filtering.** A flat, unhumanized MIDI section teaches the model
  that "human" means "mechanical." The cache builder rejects
  velocity-std/range and offset-std/range below thresholds before a sample
  ever reaches training. **Any humanization/style-transfer/denoising task needs
  this check**: verify your "human"/"target" examples actually contain the
  signal you want to learn, or you're training toward the average of your
  noise floor.
- **Song-splitting, not whole-file training.** Long files get split into
  overlapping section-samples (`--section_bars`/`--hop_bars`) so the model sees
  coherent verse/chorus/bridge-sized units instead of one giant, dilutedly-
  averaged crop.
- **Song-aware train/val split.** Splitting per-sample (not per-song) leaks
  near-duplicate sections of the same song across train/val, inflating val
  metrics. The fingerprinting system (`song_aware_split`) tracks whether a
  cache supports this and refuses to silently compare runs that don't.
- **Augmentation must not corrupt the target.** Bar-rotation (drop the first
  N bars so the model doesn't over-index on phrase openings) applies
  *identically* to input and target — described in the source as "SAFE because
  the result is still a real performance, just a rotated one." **Any
  augmentation for a labelling task needs this property**: transform input and
  label together, and verify the transformed pair is still a valid
  (input, label) example, not just a valid input.
- **Preserve what's already there, don't overwrite blindly.**
  `--preserve_grid_distance` reduces model influence on notes that are *already*
  off-grid in the input (already humanized) — recognizing "this input already
  has real signal, don't stomp it with the model's own average" is a distinct
  design axis from "how strong is the model," and easy to miss if you only
  think in terms of one global strength knob.
- **max_seq_len is a compute knob, not just a ceiling — and this is the single
  highest-leverage lesson in the whole project.** Every sample pads to
  max_seq_len and attention is O(n²). Measured on this library: median sample
  = 38 notes, p95 = 176, only 0.10% exceed 1024. The old default of 1024 made
  **93.9% of every batch pure padding** — 16x wasted linear-layer work, 123x
  wasted attention work. Benchmarked concretely (GTX 1650): 1024/bs32 → 17.1
  s/batch, 15.5GB VRAM; 256/bs32 → 1.18 s/batch, 1.6GB VRAM (14x faster, 10x
  less memory) while still covering 98.5% of samples uncropped. **Always
  histogram your actual sequence-length distribution before picking a padding
  ceiling — p50/p95/p99, not a round number that "should be enough."** Related
  and nastier: on a small card, an oversized max_seq_len can silently spill to
  system RAM over PCIe (WDDM) rather than hard-OOM, so training still "runs" at
  17s/batch with `nvidia-smi` showing a misleading 100% util (thrashing, not
  compute) — a slow run is not proof the model needs more capacity or the
  hardware needs upgrading; profile before concluding either.
- **DataLoader workers can be worse than none on `spawn`-based platforms.**
  Windows/macOS spawn workers by pickling the whole dataset per worker; with
  `num_workers=4` for train+val that's `(2*4+1)` = 9x the cache size in RAM,
  and it dies inside `w.start()` with a `multiprocessing/reduction.py`
  traceback that doesn't point at the real cause (cache size). Fix: measure
  projected RAM up front and fall back to `num_workers=0` with an explanation,
  rather than let the user hit a cryptic crash. Linux `fork` doesn't have this
  problem (copy-on-write). **Any large-in-memory-dataset + multi-worker
  DataLoader combination on Windows/macOS needs this check.**

## 4. The grid-search / experiment-management pattern (highly transferable)

This is the part most likely to be directly reusable on an unrelated problem —
it's a general pattern for "cheaply screen many hyperparameter combos, then
commit full compute to the winner":

1. **Screen cheap, commit expensive.** Sweep combos train on a *subset* of data
   for *few* epochs (`data_fraction`, auto-picked smaller as combo count grows,
   floored at 15%) — you only need each combo's *relative ranking*, not its
   final quality. Once ranked, automatically retrain the single winner on full
   data for many more epochs (`final_epoch_multiplier x` the sweep's epoch
   count) — that's the one model that's actually kept, so it's the only one
   worth paying full price for.
2. **Fingerprint what makes results comparable.** A dict of everything that
   defines the val task (loss weights, bin counts, recipe version, data
   fraction, split strategy, seed, dataset size...) is stamped onto every run.
   Before ranking a prior checkpoint alongside a new sweep, its fingerprint is
   diffed against the current one — mismatches get a **named reason** (not
   silently dropped, not silently included) so "this run's number isn't
   comparable" is visible, not a trap. **This generalizes to any recurring
   experiment-tracking setup**: never merge metrics across runs without
   checking they were measured the same way, and when you reject a comparison,
   say why.
3. **Cheap re-ranking without reloading heavyweight artifacts.** Best val_loss
   for ranking is read from a small per-epoch `log.jsonl`, not by
   `torch.load`-ing a 60M-parameter checkpoint (seconds + GB per run just to
   read one float). Persist the *cheap summary* separately from the *expensive
   artifact*.
4. **Persist the leaderboard, don't just print it.** A multi-hour sweep's
   whole ranked result used to live only in stdout — lose the scrollback, lose
   the ranking (recoverable by re-deriving from `log.jsonl` per run, but that's
   real, avoidable work). Fixed by writing a `grid_results.json` with
   leaderboard + rejected/uncomparable runs (with reasons) + the winning
   config + the final retrain result, written via **atomic temp-file + rename**
   so a reader never sees a half-written file, written *before* the long final
   training pass starts (so a crash there doesn't lose the sweep's ranking, which
   is the expensive part) and again after it finishes.
5. **A sweep must survive one bad combo.** Each combo's training is wrapped so
   an exception (including OOM, with `torch.cuda.empty_cache()`) reports and
   drops that one row rather than aborting every other combo's results —
   critical for an unattended overnight run.
6. **Preflight, don't discover resource limits mid-run.** Before launching N
   combos, the sweep estimates total disk usage (`best.pt` + `last.pt` per
   combo) against `shutil.disk_usage` and warns if it's tight, since running
   out of disk mid-sweep otherwise fails *late*, deep into wasted compute.

### A real bug this surfaced this session (worth remembering as a class of bug)

The sweep names each run directory from its hyperparameters, e.g.
`f"{prefix}_{model}_bs{batch}_lr{lr:.0e}"`. `.0e` = **zero decimal digits**.
That's fine when swept values are far apart (4e-4 vs 8e-4 never collide), but a
narrower, more exploratory lr axis (1e-3, 1.2e-3, 1.6e-3) collided: both 1e-3
and 1.2e-3 formatted to the identical string `"1e-03"`. The second run found
the first run's checkpoint already sitting at its target epoch count, **auto-
resumed, saw nothing left to do, and reported the first run's val_loss as its
own** — silently voiding an entire combo without any error, warning, or crash.
It only became visible because two "different" leaderboard rows had *byte-
identical* val_loss and pointed at the same checkpoint path.

**General lesson: any naming/hashing scheme that collapses configuration into a
string (run names, cache keys, experiment IDs) must be tested for collisions
whenever the swept values get closer together than the format's precision** —
and a resume/checkpoint-reuse feature that's convenient in the common case can
silently corrupt results in the collision case, precisely *because* it doesn't
error. When something can silently short-circuit instead of loudly failing,
that's the case to specifically design against. Fix applied: format with
enough precision (`.2e`) that swept values in the same regime can't collapse.
Also worth building, if this recurs: an explicit assert that all generated run
names in one sweep are unique, so a future precision gap fails loudly instead
of silently.

### Other checkpoint/resume gotchas worth carrying forward

- **A learning-rate schedule that bakes in total step count (`OneCycleLR`)
  cannot be naively resumed if anything that affects step count changed**
  (epochs, batch size, data fraction) — its whole warmup/anneal shape is fixed
  at construction. Loading an incompatible `state_dict` doesn't fail at load
  time; it silently corrupts internal bookkeeping and crashes much later on an
  ordinary `.step()` call, far from the real cause. Fix: detect the mismatch
  (compare saved vs. current `total_steps`) and fast-forward a *freshly
  constructed* scheduler to the equivalent epoch rather than loading old state.
  **Any stateful object whose behavior depends on a value fixed at
  construction time (schedules, some optimizers, iterators) needs this same
  guard on resume.**
- **Update your "best" bookkeeping before you write the checkpoint that
  encodes it, not after.** `best_val` used to update only inside `if is_best`
  *after* `last.pt` had already been written for this epoch — so resuming from
  `last.pt` (including auto-resume) silently restored a stale, one-epoch-old
  `best_val`, which also corrupted early-stopping's patience counter after any
  resume. Reordered so the value is current *before* the checkpoint that
  embeds it is saved.
- **Retry once at reduced batch size on OOM, don't just die**, especially for a
  long unattended final-training pass — halve batch size, scale lr with it
  (linear-scaling rule), use a *different* run name for the retry so it can't
  auto-resume from the OOM'd attempt's incompatible half-written checkpoint.

## 5. General workflow lessons from this session specifically

- **Measure GPU limits empirically instead of estimating from formulas.**
  When asked whether a larger batch size would fit, the right move was to
  actually launch a training step at that batch size and read the real
  `OutOfMemoryError` boundary, then binary-search neighboring values — not
  extrapolate from parameter counts. Concrete numbers (128→18.7GB, 160→23.1GB
  w/ ~450MB headroom, 192→OOM) are far more trustworthy than any a priori
  estimate, and they were cheap to obtain (a few one-epoch smoke runs) on
  otherwise-idle hardware.
- **Ground every "improve this" recommendation in the sweep's own numbers, not
  ML folklore.** E.g., checking whether the winning learning rate sat at the
  edge of its tested range (suggesting the true optimum lies outside it) rather
  than just citing the standard linear-scaling-rule number. When that follow-up
  sweep actually ran, both extended lr values performed *worse* than the prior
  winner — the "still climbing" hypothesis was wrong, and this should be
  reported plainly, not glossed over, once the real data came in.
- **When something looks anomalous (e.g. two processes seemingly clobbering
  one shared results file), investigate before assuming or reverting.** In
  this case: `ps aux`, `nvidia-smi --query-compute-apps`, and directory
  mtimes revealed it was one process the whole time, misidentified because of
  a wrong assumption about a CLI default (`--run_name` defaults to a stamped
  string, not `None`). Cheap, direct system inspection resolved the confusion
  faster than reasoning about it in the abstract.
- **A DESIGN-comment culture pays off.** Nearly every non-obvious constant or
  choice in this codebase has an inline `# DESIGN:` comment citing the actual
  measurement or incident that justifies it (e.g. "base 8.2999 vs deep 8.2949,
  a 0.006 gap = noise," or the max_seq_len padding table above). This makes
  the code self-documenting for exactly the question a future session (human
  or AI) will ask — "why was it built this way, and does that reasoning still
  hold?" **When making a data-driven default choice, write the number and the
  date/context next to the code, not just in a commit message or chat log —
  the code is what the next reader actually opens.**
- **Persist expensive-to-regenerate results (leaderboards, sweep summaries) as
  structured data files, not just stdout**, specifically because long-running
  jobs' terminal output is exactly the thing likely to be lost (session ends,
  scrollback truncates, SSH drops) at the moment it's most expensive to
  reproduce.
- **Long training jobs: launch detached (`nohup ... &`), redirect to a log
  file, confirm it's actually progressing (not just alive) before considering
  the launch done, and don't poll it in a tight loop** — check in when there's
  a real update to report, or when the user asks.

## 6. Config-tagging convention worth stealing

This project's sibling script (`config.py`) tags every constant as one of:
`JUDGMENT CALL` (developer intuition, fine to retune by feel), `VERIFIED
FINDING`, or `HARD TECHNICAL CONSTRAINT` (derived from something real — don't
casually change without re-reading why it's there). **A three-way tag like this
is a cheap way to encode "how much evidence backs this default" directly in
the code**, so a future editor knows whether tweaking a value is a free
judgment call or requires re-deriving a measurement first.

## 7. What's genuinely project-specific (won't transfer)

For completeness / avoiding false generalization: MIDI parsing specifics
(ticks-per-beat, flam-gap windows in ms, time-signature tables), the exact
velocity-bin count (128, chosen for "lossless" MIDI velocity resolution), the
specific instrument taxonomy, and the RTX-4090-specific VRAM numbers (128 fits,
192 doesn't) are all facts about *this* dataset and *this* hardware — re-measure
them fresh on a new problem/machine rather than assuming they carry over. What
carries over is the *methodology* used to obtain them (histogram your sequence
lengths, benchmark your own batch/model-size grid, measure your own VRAM
ceiling empirically) — sections 1-6 above.
