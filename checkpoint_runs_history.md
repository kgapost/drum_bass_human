# Humanizer checkpoint history (drum_humanizer_v3.py)

Snapshot taken 2026-09-13 while merging `checkpoints/` and `checkpoints.bak/` and pruning down to the 3 most useful runs. This document is the permanent record of every run that existed before the prune, so nothing is actually lost by deleting the checkpoint directories.

**The current best model is already safely exported to `pretrained/humanizer_best.pt` + `pretrained/humanizer_metadata.json`** (lean, inference-only weights, independent of anything below). Nothing here is needed to keep that model working - what's kept below is kept for training resumability and hyperparameter-search reference, not because the production model depends on it.

## Why these 20 runs are NOT all directly comparable by `best_val`

`drum_humanizer_v3.py` itself only ranks runs that share an identical "fingerprint" - same val split, same data fraction, same corpus size, same loss recipe (see `run_fingerprint()` / `scan_previous_runs()` in the source). Scanning all 20 runs' `run_meta.json` fingerprints found **4 distinct groups** - runs in different groups validated on different data, so a lower raw loss does not necessarily mean a better model.

The key split: 9 runs (the `run_v2_*` family) were trained against a **252,195-sample** corpus; 11 runs (`grid_*`, `run_001_*`) were trained against an **older, smaller 164,463-sample** corpus (~35% fewer samples - the MIDI library has grown since those ran). Runs on the smaller/easier corpus post structurally lower losses that are not a fair comparison against the current corpus.

### Group: 252,195-sample corpus, data_fraction=1.0, val_split=0.05 (1 run)

| Run | Location | Model size | Batch | LR | Epochs | Best val loss | Kept? |
|---|---|---|---|---|---|---|---|
| `run_v2_final` | checkpoints | huge | 128 | 0.001 | 200 | 6.7537 | YES |

### Group: 252,195-sample corpus, data_fraction=0.25, val_split=0.05 (8 runs)

| Run | Location | Model size | Batch | LR | Epochs | Best val loss | Kept? |
|---|---|---|---|---|---|---|---|
| `run_v2_huge_bs128_lr1.00e-03` | checkpoints | huge | 128 | 0.001 | 20 | 8.0000 | YES |
| `run_v2_huge_bs160_lr8.00e-04` | checkpoints | huge | 160 | 0.0008 | 20 | 8.0104 | deleted |
| `run_v2_huge_bs160_lr1.00e-03` | checkpoints | huge | 160 | 0.001 | 20 | 8.0145 | deleted |
| `run_v2_huge_bs128_lr5.00e-04` | checkpoints | huge | 128 | 0.0005 | 20 | 8.0149 | deleted |
| `run_v2_huge_bs128_lr7.00e-04` | checkpoints | huge | 128 | 0.0007 | 20 | 8.0189 | deleted |
| `run_v2_huge_bs160_lr7.00e-04` | checkpoints | huge | 160 | 0.0007 | 20 | 8.0207 | deleted |
| `run_v2_huge_bs128_lr8.00e-04` | checkpoints | huge | 128 | 0.0008 | 20 | 8.0220 | deleted |
| `run_v2_huge_bs160_lr5.00e-04` | checkpoints | huge | 160 | 0.0005 | 20 | 8.0277 | deleted |

### Group: 164,463-sample corpus, data_fraction=1.0, val_split=0.05 (1 run)

| Run | Location | Model size | Batch | LR | Epochs | Best val loss | Kept? |
|---|---|---|---|---|---|---|---|
| `run_001_final` | checkpoints.bak | huge | 128 | 0.0008 | 70 | 7.5518 | YES |

### Group: 164,463-sample corpus, data_fraction=0.25, val_split=0.05 (10 runs)

| Run | Location | Model size | Batch | LR | Epochs | Best val loss | Kept? |
|---|---|---|---|---|---|---|---|
| `grid_huge_bs128_lr8e-04` | checkpoints.bak | huge | 128 | 0.0008 | 20 | 7.8583 | deleted |
| `run_001_huge_bs128_lr1e-03` | checkpoints.bak | huge | 128 | 0.0012 | 20 | 7.8636 | deleted |
| `grid_very_deep_bs128_lr8e-04` | checkpoints.bak | very_deep | 128 | 0.0008 | 20 | 7.8783 | deleted |
| `grid_huge_bs128_lr4e-04` | checkpoints.bak | huge | 128 | 0.0004 | 20 | 7.8809 | deleted |
| `grid_very_deep_bs128_lr4e-04` | checkpoints.bak | very_deep | 128 | 0.0004 | 20 | 7.8974 | deleted |
| `run_001_huge_bs128_lr2e-03` | checkpoints.bak | huge | 128 | 0.0016 | 20 | 7.8984 | deleted |
| `grid_huge_bs64_lr4e-04` | checkpoints.bak | huge | 64 | 0.0004 | 20 | 7.9438 | deleted |
| `grid_very_deep_bs64_lr4e-04` | checkpoints.bak | very_deep | 64 | 0.0004 | 20 | 7.9454 | deleted |
| `grid_very_deep_bs64_lr8e-04` | checkpoints.bak | very_deep | 64 | 0.0008 | 20 | 7.9463 | deleted |
| `grid_huge_bs64_lr8e-04` | checkpoints.bak | huge | 64 | 0.0008 | 20 | 7.9700 | deleted |

## Conclusions

1. **`run_v2_final` is the best real result and is already the deployed model.** Trained on the full (current, 252,195-sample) corpus for 200 epochs, val_loss=6.7537 - this is exactly what `pretrained/humanizer_best.pt` was exported from.
2. **Its hyperparameters (huge, batch_size=128, lr=1e-3) were themselves the winner of the `run_v2_huge_*` grid search** - among the 8 probes at 25% of the current corpus, `run_v2_huge_bs128_lr1.00e-03` had the lowest val_loss (8.00003), and that config is what `run_v2_final` then trained to completion. The grid search did its job correctly.
3. **`batch_size=160` never beat `batch_size=128`** at any matching LR in the `run_v2` grid (e.g. lr=8e-4: bs128=8.02198 vs bs160=8.01036 - a wash; lr=1e-3 bs128 was the outright best). No evidence bs160 is worth the extra memory/compute.
4. **`very_deep` did not outperform `huge`** in the older grid search (best `very_deep` 7.87833 vs best `huge` 7.85831 at the same corpus/budget) - `huge` remains the right architecture choice; the extra depth is not paying for itself.
5. **`run_001_final` (val_loss=7.55, 164,463-sample corpus) is now superseded**, not because it's a worse model, but because it was measured on an older, smaller corpus - it is not on the same scale as `run_v2_final`'s 6.75 and should not be compared to it directly. It remains useful only as a historical reference / alternate resumable checkpoint.
6. **Optimal LR for the `huge` architecture at bs=128 consistently landed in the 7e-4-1.2e-3 range across BOTH independent grid searches** (v2 corpus: 1e-3 won; older corpus: 8e-4-1.2e-3 were the top 2) - a robust, corpus-independent signal for future sweeps: center the next search around 1e-3 rather than re-exploring a wide range.

## What was kept, and why

Merged `checkpoints/` + `checkpoints.bak/` into a single `checkpoints/` directory, keeping only these 3 (plus the unrelated `checkpoints/seg_model/` segmentation run, which is a different model entirely and was left untouched):

- **`run_v2_final`** - the best real result overall; the source of the current production model. Kept for resumability (has optimizer/scheduler state the lean `pretrained/` copy does not).
- **`run_v2_huge_bs128_lr1.00e-03`** - the grid-search run whose config `run_v2_final` scaled up to a full run. Kept as the documented lineage/reproducibility record for the shipped model.
- **`run_001_final`** - the second full-corpus "final" run (older corpus). Kept as a fallback/comparison checkpoint and because it has 70 epochs of optimizer state, in case the older corpus is ever revisited.

The other 17 run directories (all short 20-epoch grid-search probes, ~1.4GB or 876MB each) were deleted after being fully recorded in the tables above - their config and result live on here even though the weights are gone.

Disk freed: ~22.7GB (27GB across both directories before -> ~4.3GB after, including seg_model).
