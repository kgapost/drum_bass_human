"""
Fetch the bundled pretrained humanizer model from the shared Google Drive folder,
if it isn't already present locally.

DESIGN: pretrained/ is gitignored (see knowledge.md/README - a ~250MB checkpoint
doesn't belong in git history), so a fresh clone of this repo has NO model at all
until this runs once. drum_bass_studio.py calls this automatically at startup;
it's also runnable standalone:

    python download_pretrained.py            # fetch only if missing locally
    python download_pretrained.py --force     # re-fetch even if already present

Needs the 'gdown' package (pip install gdown) - it's what makes this reliable for
a large file, handling Google's virus-scan-warning interstitial and confirmation
tokens that a plain requests.get() on the file URL would choke on.
"""
import os
import sys
import argparse

try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drum_humanizer_v3 as dh

# DESIGN: this is the shared Drive FOLDER's id, not a specific file id. Matching by
# filename inside the folder - rather than hardcoding a file id - means re-uploading
# a replacement model in Drive is picked up automatically with no code change here.
PRETRAINED_DRIVE_FOLDER_ID = '1Q7PnRZUZ5Xm1V9DnC1PX3jJHGW2d45Ye'
PRETRAINED_DRIVE_FOLDER_URL = f'https://drive.google.com/drive/folders/{PRETRAINED_DRIVE_FOLDER_ID}'

_EXPECTED_FILES = {
    'humanizer_best.pt': dh.DEFAULT_PRETRAINED_CHECKPOINT,
    'humanizer_metadata.json': dh.DEFAULT_PRETRAINED_METADATA,
}


def ensure_pretrained_model(force: bool = False, quiet: bool = False) -> bool:
    """
    Make sure pretrained/humanizer_best.pt exists locally, downloading it from the
    shared Drive folder if not.

    Returns True if the checkpoint is available locally after this call, False if
    it couldn't be obtained (no gdown, offline, nothing uploaded yet, Drive error).
    NEVER raises - this is a best-effort startup convenience, not a hard dependency.
    --mode infer and the UI both still work with an explicit --checkpoint even if
    this fails outright.
    """
    if not force and os.path.exists(dh.DEFAULT_PRETRAINED_CHECKPOINT):
        return True
    if not HAS_GDOWN:
        print("[pretrained] 'gdown' not installed (pip install gdown) - skipping "
              "auto-download. Pass --checkpoint explicitly, or install gdown and rerun.")
        return False
    tmp_paths = [p + '.tmp' for p in _EXPECTED_FILES.values()]
    try:
        print(f"[pretrained] No local model found - checking the shared Drive folder "
              f"({PRETRAINED_DRIVE_FOLDER_URL}) ...")
        remote_files = gdown.download_folder(id=PRETRAINED_DRIVE_FOLDER_ID,
                                             skip_download=True, quiet=True)
        by_name = {os.path.basename(f.path): f for f in remote_files}
        if 'humanizer_best.pt' not in by_name:
            print(f"[pretrained] No 'humanizer_best.pt' in the shared Drive folder yet "
                  f"- nothing to download. Upload one there, or pass --checkpoint "
                  f"explicitly / run --mode grid_search to train one.")
            return False
        os.makedirs(dh.PRETRAINED_DIR, exist_ok=True)
        for name, local_path in _EXPECTED_FILES.items():
            if name not in by_name:
                continue   # metadata.json is nice-to-have, not required
            tmp = local_path + '.tmp'
            gdown.download(id=by_name[name].id, output=tmp, quiet=quiet)
            os.replace(tmp, local_path)   # atomic - never leaves a half-downloaded model
        size_mb = os.path.getsize(dh.DEFAULT_PRETRAINED_CHECKPOINT) / 1e6
        print(f"[pretrained] Downloaded humanizer_best.pt ({size_mb:.0f}MB) -> "
              f"{dh.DEFAULT_PRETRAINED_CHECKPOINT}")
        return True
    except Exception as exc:
        print(f"[pretrained] Could not fetch the model from Google Drive "
              f"({type(exc).__name__}: {exc}). Continuing without it - pass "
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
