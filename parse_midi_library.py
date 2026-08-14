#!/usr/bin/env python3
"""
==============================================================================
 MIDI LIBRARY CLEANUP  (genre pruning + exact-duplicate dedup + empty dirs)
==============================================================================
Reproduces, on any copy of the same MIDI library, the cleanup done by hand on
the original: /media/kapost/Schemsis/Midi - Copy

FOUR PHASES, IN ORDER
----------------------
  1) Delete the ViR2 pack entirely (provenance of its "real drummer" grooves
     could never be confirmed).
  2) Delete specific unwanted genres: punk, jungle (except the two Toontrack
     "Real Jazz" brushes/sticks technique folders, which use "jungle" as a
     jazz tempo term, not the electronic genre), rave, cha cha, marcha/rancho,
     afrobeat, NWOBHM, EDM, trance, industrial.
  3) Delete "house" genre folders + the entire Groove Monkee Electronic pack
     (both the regular and _EZD editions, including its Basic/Bonus/
     Percussion subfolders that aren't tied to a specific electronic
     subgenre).
  4) Find EXACT duplicate files by content hash (not filename) and delete the
     redundant copies, using two rules:
       - format-duality pairs (same groove shipped for two different plugin
         formats, e.g. "_EZD"/"EZX"-branded vs Superior-Drummer-native): keep
         whichever format KEEP_FORMAT below says you actually use.
       - genuinely redundant copies (same folder pasted under multiple
         unrelated top-level names): keep the "@"-numbered canonical folder,
         delete the plain-named duplicate(s).
     Library-metadata files (header/Aversion/kitpieces/midiDB/.dummy) are
     NEVER touched even when their hash collides across thousands of pack
     folders — those are supposed to be identical; every pack needs its own
     local copy for Toontrack/EZdrummer/BFD to recognize the folder as valid
     content. Deleting them would break the library, not just save space.
  5) Remove any directories left empty by the above (repeated until none
     remain, since removing one can empty its parent).

USAGE
-----
  python dedup_midi_library.py "/path/to/Midi - Copy"            # dry run
  python dedup_midi_library.py "/path/to/Midi - Copy" --execute  # for real

Defaults to a DRY RUN that only prints what it would do. Pass --execute to
actually delete anything.
"""

import argparse
import collections
import hashlib
import os
import re
import shutil
import sys

# Which plugin-format copy to KEEP when a groove was shipped for both
# Superior Drummer (native) and EZdrummer/EZX ("_EZD" suffix or "EZX" in the
# folder name). Set to 'ezd' if you use EZdrummer instead.
KEEP_FORMAT = 'sd3'   # 'sd3' or 'ezd'

# Filenames known to be tiny per-folder marker files (5 bytes to ~1.6KB) that
# Toontrack/EZdrummer/BFD use to recognize each pack folder as valid library
# content. They are SUPPOSED to hash identically across thousands of folders.
# Kept here for documentation; the actual safety net is the zero-byte rule
# below, since this named list is not exhaustive — this library also has
# ".head" files per tempo folder, ".dummyS2", ".EZPP", ".updatedummy", and an
# SVN pristine-cache file, all 0 bytes, discovered by the zero-byte rule
# rather than by name.
LIBRARY_METADATA_NAMES = {"header", ".dummy", "midiDB", "kitpieces", "Aversion"}


def human_mb(num_bytes):
    return f"{num_bytes / 1024 / 1024:.1f} MB"


def find_dirs(base, substrings, exclude_substrings=None, path_regex=None):
    """Recursively find directories whose NAME contains any of `substrings`
    (case-insensitive), optionally excluding matches whose full path contains
    any of `exclude_substrings`, or matching a regex instead of substrings."""
    matches = []
    for root, dirs, _files in os.walk(base):
        for d in dirs:
            name = d
            hit = False
            if path_regex is not None:
                hit = bool(path_regex.search(name))
            else:
                hit = any(s.lower() in name.lower() for s in substrings)
            if not hit:
                continue
            full = os.path.join(root, d)
            if exclude_substrings and any(x.lower() in full.lower() for x in exclude_substrings):
                continue
            matches.append(full)
    return matches


def dedupe_nested(paths):
    """Keep only the topmost directory when one match is nested inside another."""
    kept = []
    for p in sorted(paths):
        if not any(p == k or p.startswith(k + os.sep) for k in kept):
            kept.append(p)
    return kept


def count_files(path):
    total = 0
    for _root, _dirs, files in os.walk(path):
        total += len(files)
    return total


def delete_dir(path, dry_run, label=None):
    n = count_files(path)
    tag = f" [{label}]" if label else ""
    print(f"  {'[DRY RUN] would delete' if dry_run else 'deleting'}{tag}: {path}  ({n} files)")
    if not dry_run:
        shutil.rmtree(path, ignore_errors=True)
    return n


def find_top_level(base, substring):
    """Case-insensitive substring match against the DIRECT children of `base`
    only (not recursive). Used for folders whose numeric '@' prefix might
    differ — or be entirely absent — on a different copy of the library.
    Never raises: returns [] if `base` can't be listed at all."""
    try:
        entries = os.listdir(base)
    except OSError as e:
        print(f"  WARNING: could not list {base}: {e}")
        return []
    matches = []
    for name in entries:
        full = os.path.join(base, name)
        if os.path.isdir(full) and substring.lower() in name.lower():
            matches.append(full)
    return matches


# -----------------------------------------------------------------------
# PHASE 1 — ViR2
# -----------------------------------------------------------------------
def phase_delete_vir2(base, dry_run):
    print("\n=== Phase 1: ViR2 ===")
    matches = find_top_level(base, "vir2")
    if not matches:
        print("  ViR2 folder not found, skipping.")
        return 0
    total = 0
    for path in matches:
        total += delete_dir(path, dry_run, label="ViR2")
    return total


# -----------------------------------------------------------------------
# PHASE 2 — genre pruning
# -----------------------------------------------------------------------
GENRE_SPECS = [
    # (label, substrings, exclude_substrings, path_regex, exact_name_exclude)
    ("Punk", ["punk"], None, None, None),
    ("Jungle", ["jungle"], None, None, {"jungle brushes", "jungle sticks"}),
    ("Rave", ["rave"], ["traveller", "travelling", "brave", "raven"], None, None),
    ("Cha Cha", None, None, re.compile(r"cha.*cha", re.I), None),
    ("Marcha/Rancho", ["marcha", "rancho"], None, None, None),
    ("Afrobeat", ["afrobeat"], None, None, None),
    ("NWOBHM", None, None, re.compile(r"new.*wave.*british", re.I), None),
    ("EDM", ["edm"], None, None, None),
    ("Trance", ["trance"], None, None, None),
    ("Industrial", ["industrial"], None, None, None),
]


def phase_delete_genres(base, dry_run):
    print("\n=== Phase 2: genre pruning ===")
    total = 0
    for label, substrings, exclude_substrings, path_regex, exact_name_exclude in GENRE_SPECS:
        dirs = find_dirs(base, substrings or [], exclude_substrings, path_regex)
        if exact_name_exclude:
            dirs = [d for d in dirs if os.path.basename(d).lower() not in exact_name_exclude]
        dirs = dedupe_nested(dirs)
        n_files = 0
        for d in dirs:
            n_files += delete_dir(d, dry_run, label=label)
        print(f"  {label}: {len(dirs)} folders, {n_files} files")
        total += n_files
    print(f"  Phase 2 total: {total} files")
    return total


# -----------------------------------------------------------------------
# PHASE 3 — house + Groove Monkee Electronic pack
# -----------------------------------------------------------------------
def phase_delete_house_and_electronic(base, dry_run):
    print("\n=== Phase 3: house + Groove Monkee Electronic pack ===")
    total = 0

    # "house" only within these specific parent packs (avoids false positives
    # like "...WAREHOUSE", "Coach House", "Stone House", "...Houses..." tracks
    # elsewhere in the library that aren't the house genre). Matched by
    # substring, not exact name, since the numeric '@' prefix can differ (or
    # be missing) on a different copy of the library.
    house_parent_substrings = ["ezx_dance", "groove_monkee_funk_hip_hop_rb"]
    house_dirs = []
    any_parent_found = False
    for sub in house_parent_substrings:
        parents = find_top_level(base, sub)
        if not parents:
            print(f"  (parent pack containing '{sub}' not found, skipping its house check)")
            continue
        for parent in parents:
            any_parent_found = True
            try:
                children = os.listdir(parent)
            except OSError as e:
                print(f"  WARNING: could not list {parent}: {e}")
                continue
            for d in children:
                full = os.path.join(parent, d)
                if os.path.isdir(full) and "house" in d.lower():
                    house_dirs.append(full)
    house_dirs = dedupe_nested(house_dirs)
    n_files = 0
    for d in house_dirs:
        n_files += delete_dir(d, dry_run, label="House")
    if any_parent_found:
        print(f"  House: {len(house_dirs)} folders, {n_files} files")
    total += n_files

    # Entire Groove Monkee Electronic pack, both editions, wholesale.
    # Matched by substring for the same reason as above.
    electronic_dirs = find_top_level(base, "groove_monkee_electronic")
    if not electronic_dirs:
        print("  Groove Monkee Electronic pack not found, skipping.")
    for path in electronic_dirs:
        total += delete_dir(path, dry_run, label="Groove Monkee Electronic")

    print(f"  Phase 3 total: {total} files")
    return total


# -----------------------------------------------------------------------
# PHASE 4 — exact-duplicate detection by content hash
# -----------------------------------------------------------------------
def hash_file(path, chunk_size=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def top_folder_name(base, path):
    rel = os.path.relpath(path, base)
    return rel.split(os.sep)[0]


def classify_folder(folder_name):
    if folder_name.endswith("_EZD") or "EZX" in folder_name:
        return "EZD_EZX"
    if "@" in folder_name:
        return "NUMBERED"
    return "PLAIN"


def numbered_prefix(folder_name):
    m = re.match(r"^(\d+)@", folder_name)
    return int(m.group(1)) if m else float("inf")


def phase_dedupe_by_hash(base, dry_run, keep_format="sd3"):
    print("\n=== Phase 4: exact-duplicate dedup by content hash ===")
    print(f"  Hashing all files under: {base}")

    hashes = collections.defaultdict(list)
    n_hashed = 0
    for root, _dirs, files in os.walk(base):
        for fname in files:
            full = os.path.join(root, fname)
            if os.path.basename(full) in LIBRARY_METADATA_NAMES:
                continue  # never even consider these for dedup
            try:
                if os.path.getsize(full) == 0:
                    continue  # zero-byte files are always per-folder markers,
                              # never real content — and freeing 0 bytes isn't
                              # worth the risk of breaking a library-recognition marker
                h = hash_file(full)
            except OSError:
                continue
            hashes[h].append(full)
            n_hashed += 1
            if n_hashed % 20000 == 0:
                print(f"  ...hashed {n_hashed} files so far")

    print(f"  Hashed {n_hashed} files (library-metadata files excluded from hashing).")

    to_delete = []
    format_groups = 0
    redundant_groups = 0

    for _h, paths in hashes.items():
        if len(paths) < 2:
            continue

        tops = [top_folder_name(base, p) for p in paths]
        classes = [classify_folder(t) for t in tops]

        has_native = any(c in ("NUMBERED", "PLAIN") for c in classes)
        has_ezd = any(c == "EZD_EZX" for c in classes)

        if has_native and has_ezd:
            if keep_format == "sd3":
                keep_side = [i for i, c in enumerate(classes) if c != "EZD_EZX"]
            else:
                keep_side = [i for i, c in enumerate(classes) if c == "EZD_EZX"]
            numbered_keep = [i for i in keep_side if classes[i] == "NUMBERED"]
            keep_idx = (min(numbered_keep, key=lambda i: numbered_prefix(tops[i]))
                        if numbered_keep else sorted(keep_side, key=lambda i: paths[i])[0])
            format_groups += 1
        else:
            numbered_idxs = [i for i, c in enumerate(classes) if c == "NUMBERED"]
            keep_idx = (min(numbered_idxs, key=lambda i: numbered_prefix(tops[i]))
                        if numbered_idxs else sorted(range(len(paths)), key=lambda i: paths[i])[0])
            redundant_groups += 1

        for i in range(len(paths)):
            if i != keep_idx:
                to_delete.append(paths[i])

    total_size = 0
    for p in to_delete:
        try:
            total_size += os.path.getsize(p)
        except OSError:
            pass

    print(f"  Format-duality groups: {format_groups}")
    print(f"  Redundant-copy groups: {redundant_groups}")
    print(f"  Files to delete: {len(to_delete)}  (~{human_mb(total_size)})")

    if dry_run:
        print("  [DRY RUN] not deleting. Re-run with --execute to apply.")
        return 0

    deleted = 0
    for p in to_delete:
        try:
            os.remove(p)
            deleted += 1
        except OSError as e:
            print(f"  failed to delete {p}: {e}")
    print(f"  Deleted {deleted} files.")
    return deleted


# -----------------------------------------------------------------------
# PHASE 5 — empty directories
# -----------------------------------------------------------------------
def phase_remove_empty_dirs(base, dry_run):
    print("\n=== Phase 5: empty directories ===")
    total = 0
    rounds = 0
    while True:
        empty_dirs = []
        for root, dirs, files in os.walk(base, topdown=False):
            if root == base:
                continue
            if not dirs and not files:
                empty_dirs.append(root)
        if not empty_dirs:
            break
        rounds += 1
        print(f"  Round {rounds}: {'[DRY RUN] would remove' if dry_run else 'removing'} "
              f"{len(empty_dirs)} empty directories")
        if dry_run:
            total += len(empty_dirs)
            break  # dry run: don't simulate cascading, just report first pass
        for d in empty_dirs:
            try:
                os.rmdir(d)
                total += 1
            except OSError:
                pass
    print(f"  Phase 5 total: {total} empty directories removed")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_dir", help="Path to the MIDI library root (e.g. 'Midi - Copy')")
    parser.add_argument("--execute", action="store_true",
                         help="Actually delete files/folders. Without this flag, it's a dry run.")
    parser.add_argument("--keep-format", choices=["sd3", "ezd"], default=KEEP_FORMAT,
                         help="Which plugin-format copy to keep in format-duality pairs "
                              "(default: %(default)s)")
    args = parser.parse_args()

    base = os.path.abspath(args.base_dir)
    if not os.path.isdir(base):
        print(f"Error: not a directory: {base}")
        sys.exit(1)

    dry_run = not args.execute
    print(f"Base directory: {base}")
    print(f"Mode: {'DRY RUN (no changes will be made)' if dry_run else 'EXECUTE (files will be deleted)'}")
    print(f"Keep format: {args.keep_format}")

    phase_delete_vir2(base, dry_run)
    phase_delete_genres(base, dry_run)
    phase_delete_house_and_electronic(base, dry_run)
    phase_dedupe_by_hash(base, dry_run, keep_format=args.keep_format)
    phase_remove_empty_dirs(base, dry_run)

    if dry_run:
        print("\nDry run complete. Re-run with --execute to actually delete anything.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
