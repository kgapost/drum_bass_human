"""
Fetch the bundled pretrained models (humanizer + segmentation) from the shared
Google Drive folder, if they aren't already present locally.

DESIGN: pretrained/ is gitignored (see knowledge.md/README - these checkpoints
don't belong in git history), so a fresh clone of this repo has NO models at all
until this runs once. drum_bass_studio.py calls this automatically at startup;
it's also runnable standalone:

    python download_pretrained.py            # fetch only whichever are missing locally
    python download_pretrained.py --force     # re-fetch both even if already present

Needs the 'gdown' package (pip install gdown) - it's what makes this reliable for
a large file, handling Google's virus-scan-warning interstitial and confirmation
tokens that a plain requests.get() on the file URL would choke on.
"""
import os
import sys
import argparse
from typing import List

try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drum_humanizer_v3 as dh
import drum_theme_segmentation as dts
from config import (
    PRETRAINED_DRIVE_FOLDER_ID, PRETRAINED_HUMANIZER_CKPT_NAME, PRETRAINED_HUMANIZER_META_NAME,
    PRETRAINED_SEGMENTATION_CKPT_NAME, PRETRAINED_SEGMENTATION_META_NAME,
)

PRETRAINED_DRIVE_FOLDER_URL = f'https://drive.google.com/drive/folders/{PRETRAINED_DRIVE_FOLDER_ID}'

_MODELS = [
    {'label': 'humanizer', 'ckpt_name': PRETRAINED_HUMANIZER_CKPT_NAME,
     'meta_name': PRETRAINED_HUMANIZER_META_NAME,
     'ckpt_path': dh.DEFAULT_PRETRAINED_CHECKPOINT, 'meta_path': dh.DEFAULT_PRETRAINED_METADATA,
     'pretrained_dir': dh.PRETRAINED_DIR},
    {'label': 'segmentation', 'ckpt_name': PRETRAINED_SEGMENTATION_CKPT_NAME,
     'meta_name': PRETRAINED_SEGMENTATION_META_NAME,
     'ckpt_path': dts.DEFAULT_PRETRAINED_CHECKPOINT, 'meta_path': dts.DEFAULT_PRETRAINED_METADATA,
     'pretrained_dir': dts.PRETRAINED_DIR},
]


def pretrained_models_missing() -> List[str]:
    """Labels (e.g. ['humanizer', 'segmentation']) of pretrained models not
    currently present locally. Pure local filesystem check - no network call -
    so callers can cheaply decide whether to prompt before fetching anything."""
    return [m['label'] for m in _MODELS if not os.path.exists(m['ckpt_path'])]


def ensure_pretrained_model(force: bool = False, quiet: bool = False) -> bool:
    """
    Make sure both bundled pretrained checkpoints (humanizer + segmentation) exist
    locally, downloading whichever are missing from the shared Drive folder.

    Returns True if every model is available locally after this call, False if any
    couldn't be obtained (no gdown, offline, nothing uploaded yet, Drive error).
    NEVER raises - this is a best-effort startup convenience, not a hard dependency.
    --mode infer/train and the UI all still work with an explicit --checkpoint even
    if this fails outright.
    """
    missing = [m for m in _MODELS if force or not os.path.exists(m['ckpt_path'])]
    if not missing:
        return True
    if not HAS_GDOWN:
        print("[pretrained] 'gdown' not installed (pip install gdown) - skipping "
              "auto-download. Pass --checkpoint explicitly, or install gdown and rerun.")
        return False
    tmp_paths = [p + '.tmp' for m in _MODELS for p in (m['ckpt_path'], m['meta_path'])]
    all_ok = True
    try:
        print(f"[pretrained] Checking the shared Drive folder "
              f"({PRETRAINED_DRIVE_FOLDER_URL}) for missing models "
              f"({', '.join(m['label'] for m in missing)}) ...")
        remote_files = gdown.download_folder(id=PRETRAINED_DRIVE_FOLDER_ID,
                                             skip_download=True, quiet=True)
        by_name = {os.path.basename(f.path): f for f in remote_files}
        for m in missing:
            if m['ckpt_name'] not in by_name:
                print(f"[pretrained] No '{m['ckpt_name']}' in the shared Drive folder yet "
                      f"- nothing to download for {m['label']}. Upload one there, or pass "
                      f"--checkpoint explicitly / train one.")
                all_ok = False
                continue
            os.makedirs(m['pretrained_dir'], exist_ok=True)
            for name, local_path in ((m['ckpt_name'], m['ckpt_path']), (m['meta_name'], m['meta_path'])):
                if name not in by_name:
                    continue   # metadata.json is nice-to-have, not required
                tmp = local_path + '.tmp'
                gdown.download(id=by_name[name].id, output=tmp, quiet=quiet)
                os.replace(tmp, local_path)   # atomic - never leaves a half-downloaded model
            size_mb = os.path.getsize(m['ckpt_path']) / 1e6
            print(f"[pretrained] Downloaded {m['ckpt_name']} ({size_mb:.0f}MB) -> {m['ckpt_path']}")
        return all_ok
    except Exception as exc:
        print(f"[pretrained] Could not fetch models from Google Drive "
              f"({type(exc).__name__}: {exc}). Continuing without them - pass "
              f"--checkpoint explicitly, or retry later.")
        return False
    finally:
        # a failed/interrupted download must not leave a partial file that a later
        # os.path.exists() check on the .tmp path could ever mistake for a real one
        for tmp in tmp_paths:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true',
                        help='re-download even if a local pretrained model already exists')
    parser.add_argument('--quiet', action='store_true',
                        help="suppress gdown's per-file progress bar")
    args = parser.parse_args()
    ok = ensure_pretrained_model(force=args.force, quiet=args.quiet)
    sys.exit(0 if ok else 1)
