#!/usr/bin/env python3
"""
==============================================================================
 MIDI LIBRARY CLEANUP  (genre pruning + exact-duplicate dedup + empty dirs)
==============================================================================
Reproduces, on any copy of the same MIDI library, the cleanup done by hand on
the original: /media/kapost/Schemsis/data

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
  6) Delete every "header" marker file in the library, unconditionally. NOTE:
     these are the same per-folder marker files phase 4 deliberately protects
     from dedup because Toontrack/EZdrummer/BFD use them to recognize a pack
     folder as valid content — deleting them everywhere may stop those
     plugins from browsing/loading the affected packs. This phase exists
     because it was explicitly requested; it is NOT implied by phases 1-5.
  7) Remove newly-empty directories left behind by phase 6.

USAGE
-----
  python parse_midi_library.py "/path/to/data"            # dry run
  python parse_midi_library.py "/path/to/data" --execute  # for real

Defaults to a DRY RUN that only prints what it would do. Pass --execute to
actually delete anything.

FILENAME FLATTENING  (separate mode, run with --flatten)
----------------------------------------------------------
Rewrites every file from its deep, numbered, "@"-riddled original path into
a flat "Company/Genre/renamed_file.ext" structure:

  data/210@GROOVE_MONKEE_BLUES/21@078 SLOW BLUES A/078 Slow Blues Hats (8) F1 S.mid
    -> data/GROOVE/SLOW BLUES A/groove_078_Slow_Blues_Hats_(8)_F1_S.mid

  data/14@EZX_METALHEADS/098-S113@THEME/Variation_01.mid
    -> data/EZX/S113-THEME/ezx_metalheads_s113_theme_Variation_01.mid

Rules:
  - COMPANY = the first word of the top-level folder (after stripping its
    leading "NN@" enumeration), e.g. "GROOVE_MONKEE_BLUES" -> "GROOVE",
    "EZX_METALHEADS" -> "EZX". Always kept, always in the filename.
  - GENRE folder (a single subfolder under COMPANY) = every folder between
    the top level and the file, each stripped of its enumeration/"@"/leading
    tempo number and joined with " - " if there's more than one. Falls back
    to the top-level folder's remainder (e.g. "METALHEADS") if there are no
    folders in between.
  - The new filename is prefixed with the lineage that produced the GENRE
    folder (lowercased, "_"-joined) — but a lineage segment (the top-level
    remainder, or an individual in-between folder) is DROPPED from the
    filename prefix if any of its words already appear in the original
    filename, so the name doesn't repeat what it already says. COMPANY is
    never dropped. The original filename's own text is preserved verbatim
    (case and all) after the prefix, just with spaces turned into "_" and a
    leading standalone tempo/index number stripped.
  - Spaces anywhere in the new filename become "_". Characters illegal on
    Windows/macOS/Linux filenames are stripped.
  - Never overwrites: if the computed new path already exists (on disk, or
    was already claimed earlier in the same run), an incrementing "_2",
    "_3", ... is appended before the extension until it's unique.

USAGE
-----
  python parse_midi_library.py "/path/to/data" --flatten --preview 20
      # print 20 randomly sampled OLD -> NEW path pairs, changes nothing

  python parse_midi_library.py "/path/to/data" --flatten --execute
      # actually rename every file

MOVE-BY-MEASURES  (separate mode, run with --move-by-measures)
------------------------------------------------------------------
Scans every .mid/.midi file, counts its length in measures (bars) via
pretty_midi's downbeat detection, and moves it into one of two new
top-level folders under the library root — each file goes to AT MOST ONE
destination, decided in this priority order:

  1) "_songs/" — file's filename OR path contains "song" or "songs"
                 (case-insensitive) AND it's longer than 64 measures.
  2) "_g48/"   — (checked only if #1 didn't match) file is longer than
                 48 measures.
  3) otherwise — left where it is.

The new filename keeps the file's former directory path folded into it
(same "_"-joined, space-safe convention as the flatten mode above), so
nothing about where it came from is lost even though it's now sitting in a
flat folder. Never overwrites — colliding names get an incrementing
"_2", "_3", ... suffix.

USAGE
-----
  python parse_midi_library.py "/path/to/data" --move-by-measures --preview 20
      # dry run, print 20 sampled OLD -> NEW moves + destination counts

  python parse_midi_library.py "/path/to/data" --move-by-measures --execute
      # actually move everything

MOVE-G24  (separate mode, run with --move-g24)
-------------------------------------------------
Same idea as move-by-measures, but simpler: every .mid/.midi file NOT
already under _songs/ or _g48/ that's longer than 24 bars moves into a new
top-level _g24/ folder, same path-folded-filename convention. Since
move-by-measures already relocated everything over 48 bars, this only
picks up files in the 25-48 bar range.

USAGE
-----
  python parse_midi_library.py "/path/to/data" --move-g24 --preview 20
  python parse_midi_library.py "/path/to/data" --move-g24 --execute

VERIFY-MIDI  (separate mode, run with --verify-midi)
-------------------------------------------------------
Scans every .mid/.midi file under the library and tries to actually load it
with pretty_midi.PrettyMIDI(...) (the same class the training pipeline uses,
e.g. drum_humanizer_v3.py's cache builder). Files that raise ANY exception
while loading (corrupt tick counts, malformed headers, etc.) are considered
faulty and deleted — this catches things like:

  ValueError: MIDI file has a largest tick of 17587201, it is likely corrupt

USAGE
-----
  python parse_midi_library.py "/path/to/data" --verify-midi --preview 20
      # dry run, print 20 sampled faulty files + their errors, deletes nothing

  python parse_midi_library.py "/path/to/data" --verify-midi
      # dry run (default), scans everything and lists every faulty file

  python parse_midi_library.py "/path/to/data" --verify-midi --execute
      # actually deletes every file pretty_midi could not load

ERASE-JUNK  (separate mode, run with --erase-junk)
------------------------------------------------------
Scans every .mid/.midi file that pretty_midi CAN load (files it can't are
--verify-midi's job, not this mode's - they're skipped here, not deleted)
and deletes ones that are too sparse or too short to ever be useful training
material for drum_humanizer_v3.py's cache builder:

  - "no_notes_at_all"       - zero notes in the whole file, any track/channel.
  - "too_few_notes(<N)"     - fewer than N notes total (default N=10),
                               counting ALL notes on ANY track/channel, before
                               any drum-specific filtering.
  - "too_few_drum_events(<N)" - fewer than N notes SURVIVE
                               drum_humanizer_v3.py's own channel-selection
                               (see select_drum_notes() below, mirroring that
                               file's _get_drum_notes()) and GM_DRUM_MAP pitch
                               filtering (default N=10, matching its
                               'too_few_events' threshold exactly). This is
                               NOT a subset of too_few_notes above - a file
                               can have plenty of raw notes and still fail
                               here, e.g. a kit using extended cymbal/
                               articulation pitches outside GM_DRUM_MAP's
                               standard 35-62 range, so most of its notes
                               never reach the model. Skipped (not counted as
                               junk under this reason) for files whose track
                               selection is itself ambiguous or empty - see
                               "too_few_notes"/--erase-ambiguous for those.
  - "shorter_than_X_measure"- the file's duration (pretty_midi's end time) is
                               less than X of one measure at its own tempo/
                               time-signature (default X=0.5, i.e. under half
                               a bar) - too short to contain a usable groove.

A file can match more than one reason; it's only deleted once, but every
matching reason is counted in the summary breakdown.

USAGE
-----
  python parse_midi_library.py "/path/to/data" --erase-junk --preview 20
      # dry run, print 20 sampled junk files + their reasons, deletes nothing

  python parse_midi_library.py "/path/to/data" --erase-junk
      # dry run (default), scans everything and lists every junk file

  python parse_midi_library.py "/path/to/data" --erase-junk --execute
      # actually deletes every file matching a junk reason

  python parse_midi_library.py "/path/to/data" --erase-junk --min-notes 4 \\
         --min-drum-events 4 --min-measure-fraction 0.25 --execute
      # override the thresholds

ERASE-AMBIGUOUS  (separate mode, run with --erase-ambiguous)
------------------------------------------------------------
Scans every .mid/.midi file that pretty_midi CAN load (files it can't are
--verify-midi's job, not this mode's - they're skipped here, not deleted) and
deletes ones that drum_humanizer_v3.py's cache builder can never use because
it can't tell which track is the drums: multiple non-empty tracks, none
flagged is_drum (channel 10), and none of them name-matched as drums (see
that file's _get_drum_notes()) - counted there as the 'ambiguous_multi_track'
failure reason. This mode mirrors that exact heuristic by hand (kept in sync
manually - see select_drum_notes() below) so it deletes precisely the files
the cache builder would otherwise silently skip on every future run. Unlike
--erase-junk, this is NOT a subset check against a simpler count - a file
can have plenty of notes and still be permanently unusable here, so nothing
else already removes these.

USAGE
-----
  python parse_midi_library.py "/path/to/data" --erase-ambiguous --preview 20
      # dry run, print 20 sampled ambiguous files, deletes nothing

  python parse_midi_library.py "/path/to/data" --erase-ambiguous
      # dry run (default), scans everything and lists every ambiguous file

  python parse_midi_library.py "/path/to/data" --erase-ambiguous --execute
      # actually deletes every file matching ambiguous_multi_track
"""

import argparse
import collections
import hashlib
import os
import random
import re
import shutil
import sys
import warnings

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False

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
def phase_remove_empty_dirs(base, dry_run, label="Phase 5"):
    print(f"\n=== {label}: empty directories ===")
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
    print(f"  {label} total: {total} empty directories removed")
    return total


# -----------------------------------------------------------------------
# PHASE 6 — delete "header" marker files (unconditional)
# -----------------------------------------------------------------------
def phase_delete_header_files(base, dry_run):
    print("\n=== Phase 6: delete 'header' files ===")
    targets = []
    for root, _dirs, files in os.walk(base):
        for fname in files:
            if fname.lower() == "header":
                targets.append(os.path.join(root, fname))

    print(f"  Found {len(targets)} 'header' files.")
    deleted = 0
    for p in targets:
        print(f"  {'[DRY RUN] would delete' if dry_run else 'deleting'}: {p}")
        if not dry_run:
            try:
                os.remove(p)
                deleted += 1
            except OSError as e:
                print(f"  failed to delete {p}: {e}")
    if dry_run:
        deleted = len(targets)
    print(f"  Phase 6 total: {deleted} 'header' files deleted")
    return deleted


# -----------------------------------------------------------------------
# FILENAME FLATTENING  (separate mode — see module docstring for the rules)
# -----------------------------------------------------------------------
ILLEGAL_CHARS_RE = re.compile(r'[<>:"/\\|?*]')
LEADING_NUMBER_RE = re.compile(r'^\d+[\s_]+')
ENUM_PREFIX_RE = re.compile(r'^\d+@')
WORD_SPLIT_RE = re.compile(r'[^A-Za-z0-9]+')


def sanitize_piece(s):
    """For filename pieces: spaces become '_' (per the 'replace spaces from
    filename with _' rule)."""
    s = ILLEGAL_CHARS_RE.sub('', s)
    s = re.sub(r'\s+', '_', s.strip())
    return s.strip('_ -') or "Misc"


def sanitize_folder_piece(s):
    """For folder names: spaces are kept (only the FILENAME rule replaces
    them), just collapsed and illegal characters stripped."""
    s = ILLEGAL_CHARS_RE.sub('', s)
    s = re.sub(r'\s+', ' ', s.strip())
    return s.strip(' _-') or "Misc"


def words_of(s):
    """Word set used for redundancy checks. Also adds a naive singular form
    for words ending in 's' (len > 3), so e.g. a folder's 'VARIATIONS' is
    recognized as redundant with a filename that already says 'Variation' —
    this is what makes the '.../VARIATIONS_01/Variation 003.mid' case drop
    the folder-derived duplicate instead of keeping both "v01" and "v003"."""
    words = set()
    for w in WORD_SPLIT_RE.split(s):
        if not w:
            continue
        wl = w.lower()
        words.add(wl)
        if len(wl) > 3 and wl.endswith('s'):
            words.add(wl[:-1])
    return words


# Word/phrase abbreviations applied to the final filename text. Order matters:
# multi-word phrases (e.g. "pre chorus") must come before the single-word
# rule they contain (e.g. "chorus"), or the single-word rule would fire first
# and leave the phrase half-abbreviated. All case-insensitive (re.I).
#
# NOTE: regex \b treats '_' as a word character, so it does NOT see a
# boundary in "..._fills_..." (both neighbors of "fills" are \w chars) — it
# would silently fail to match almost everywhere in this pipeline, since
# everything gets underscore-joined. WB/WE below are custom boundaries that
# only require "not a letter/digit" on each side, so '_', '-', space, start
# and end of string all count as real separators.
WB = r'(?<![A-Za-z0-9])'
WE = r'(?![A-Za-z0-9])'
ABBREVIATIONS = [
    (re.compile(WB + r'pre[ _-]chorus' + WE, re.I), 'pch'),
    (re.compile(WB + r'chorus' + WE, re.I), 'ch'),
    (re.compile(WB + r'verse' + WE, re.I), 'vrs'),
    (re.compile(WB + r'variations?[ _]?', re.I), 'v'),
    (re.compile(WB + r'straight[ _-]?', re.I), 's'),
    (re.compile(WB + r'fills' + WE, re.I), 'f'),
    (re.compile(WB + r'backbeat' + WE, re.I), 'bb'),
    (re.compile(WB + r'groove[ _]monke[a-z]*' + WE, re.I), 'gm'),
    (re.compile(WB + r'grooves' + WE, re.I), ''),
    (re.compile(WB + r'jazzy' + WE, re.I), 'jazz'),
    (re.compile(WB + r'pop[ _-]rock' + WE, re.I), 'pop'),
    (re.compile(WB + r'theme' + WE, re.I), 'thm'),
    (re.compile(WB + r'bridge' + WE, re.I), 'brdg'),
    (re.compile(WB + r'midtempo' + WE, re.I), 'midtmp'),
    (re.compile(WB + r'uptempo' + WE, re.I), 'uptmp'),
    (re.compile(WB + r'swing' + WE, re.I), 'swg'),
    (re.compile(WB + r'essentials?' + WE, re.I), ''),
    # words below chosen from an actual frequency scan of this library's
    # folder-lineage words (not guessed) — counts as of the scan that added them:
    # snare 120, toms/tom 169, ride 322, crash 157, hats 682, cymbal 288,
    # shuffle 155, outro 91, percussion 680, pack 653, the 925
    (re.compile(WB + r'snare' + WE, re.I), 'sn'),
    (re.compile(WB + r'toms?' + WE, re.I), 'tm'),
    (re.compile(WB + r'ride' + WE, re.I), 'rd'),
    (re.compile(WB + r'crash' + WE, re.I), 'csh'),
    (re.compile(WB + r'hats' + WE, re.I), 'ht'),
    (re.compile(WB + r'cymbal' + WE, re.I), 'cym'),
    (re.compile(WB + r'shuffle' + WE, re.I), 'shf'),
    (re.compile(WB + r'outro' + WE, re.I), 'otr'),
    (re.compile(WB + r'percussion' + WE, re.I), 'perc'),
    (re.compile(WB + r'pack' + WE, re.I), ''),
    (re.compile(WB + r'the' + WE, re.I), ''),
]

MAX_LINEAGE_SEGMENTS = 4  # cap on how many folder-derived segments get folded into the filename


def apply_abbreviations(s):
    for pattern, repl in ABBREVIATIONS:
        s = pattern.sub(repl, s)
    return s


def collapse_repeats(s):
    """'--' -> '-', '__' -> '_', '  ' -> ' ' (any run length)."""
    s = re.sub(r'_{2,}', '_', s)
    s = re.sub(r'-{2,}', '-', s)
    s = re.sub(r' {2,}', ' ', s)
    return s


def normalize_spacing(s):
    """Space-cleanup rules, applied in this order:
    ' _ ' -> '_', ' - ' -> '-', '_-_' -> '-', then spaces directly between a
    letter and a digit are removed, spaces between letter-letter or
    digit-digit are turned into '_', and anything left over (e.g. a space
    next to punctuation like a parenthesis) falls back to the general
    'replace spaces with _' rule."""
    s = s.replace(' _ ', '_')
    s = s.replace(' - ', '-')
    s = s.replace('_-_', '-')
    s = re.sub(r'(?<=[A-Za-z]) (?=[0-9])', '', s)
    s = re.sub(r'(?<=[0-9]) (?=[A-Za-z])', '', s)
    s = re.sub(r'(?<=[A-Za-z]) (?=[A-Za-z])', '_', s)
    s = re.sub(r'(?<=[0-9]) (?=[0-9])', '_', s)
    s = re.sub(r'\s+', '_', s)  # anything left over
    return s


def extract_company(top_folder_name):
    """'210@GROOVE_MONKEE_BLUES' -> ('GROOVE', 'MONKEE_BLUES')
    '14@EZX_METALHEADS' -> ('EZX', 'METALHEADS')
    '02_@EZDRUMMER_3' -> ('02', 'EZDRUMMER_3')  (underscore-before-'@' variant)"""
    name = ENUM_PREFIX_RE.sub('', top_folder_name)
    parts = re.split(r'[_\-\s]+', name, maxsplit=1)
    company = parts[0].strip() if parts and parts[0].strip() else "MISC"
    remainder = parts[1].strip() if len(parts) > 1 else ""
    remainder = remainder.lstrip('@').strip()  # e.g. "02_@EZDRUMMER_3" splits into
                                                # ("02", "@EZDRUMMER_3") since ENUM_PREFIX_RE
                                                # only strips a DIRECTLY-adjacent "NN@" prefix
    return company, remainder


def clean_folder_component(name):
    """Strip enumeration/@-prefixes and leading tempo numbers.
    '21@078 SLOW BLUES A' -> 'SLOW BLUES A'
    '098-S113@THEME' -> 'S113-THEME'"""
    if "@" in name:
        prefix, suffix = name.split("@", 1)
        suffix = suffix.strip()
        if prefix.isdigit() or not prefix:
            code = ""
        else:
            segments = [s for s in prefix.split("-") if s]
            non_numeric = [s for s in segments if not s.isdigit()]
            code = non_numeric[-1] if non_numeric else ""
        name = f"{code}-{suffix}" if code else suffix
    name = LEADING_NUMBER_RE.sub('', name)
    return name.strip(' _-')


def compute_new_name(base, path):
    """Returns (new_dir_abs_path, new_filename) for `path` under `base`,
    per the rules in the module docstring."""
    rel = os.path.relpath(path, base)
    parts = rel.split(os.sep)
    filename = parts[-1]
    top = parts[0] if len(parts) > 1 else ""
    middle = parts[1:-1]

    name, ext = os.path.splitext(filename)

    if top:
        company, top_remainder = extract_company(top)
    else:
        company, top_remainder = "MISC", ""
    company = company.upper()

    cleaned_middle = [c for c in (clean_folder_component(m) for m in middle) if c]

    genre_parts = list(cleaned_middle)
    if not genre_parts and top_remainder:
        genre_parts = [top_remainder]
    genre = " - ".join(genre_parts) if genre_parts else "Misc"

    filename_words = words_of(name)

    # Step 1: drop whole segments (top-level remainder, or an in-between
    # folder) that are already redundant with the original filename.
    candidate_segments = []
    if top_remainder and not (words_of(top_remainder) & filename_words):
        candidate_segments.append(top_remainder)
    for cleaned in cleaned_middle:
        if words_of(cleaned) & filename_words:
            continue
        candidate_segments.append(cleaned)

    # Step 2: cap how many segments get folded into the filename, keeping
    # the deepest/most specific ones (closest to the file).
    if len(candidate_segments) > MAX_LINEAGE_SEGMENTS:
        candidate_segments = candidate_segments[-MAX_LINEAGE_SEGMENTS:]

    # Step 3: dedup words across segments (and against the company), so a
    # nested pack like ".../GROOVE_MONKEE_FUSION/GROOVE_MONKEE_FUSION_NASHVILLE/..."
    # doesn't repeat "groove monkee fusion" twice — only genuinely new words
    # from each segment, in order, survive.
    used_words = set(words_of(company))
    lineage_tokens = []
    for seg in candidate_segments:
        for w in re.split(r'[\s_\-]+', seg):
            if not w or (len(w) <= 1 and not w.isdigit()):
                continue  # drop stray single letters (noise like "A"), keep single digits (often meaningful, e.g. version numbers)
            if w.lower() in used_words:
                continue
            used_words.add(w.lower())
            lineage_tokens.append(w)

    prefix_words = [company.lower()] + [t.lower() for t in lineage_tokens if t]

    # original filename text, kept as-is (own casing, own spacing/punctuation)
    # aside from stripping a leading standalone tempo/index number — the
    # abbreviation + spacing rules run on the WHOLE combined string below.
    raw_original = LEADING_NUMBER_RE.sub('', name).strip()

    # Safety nets: make sure genre and "song" (if present anywhere in the
    # original path) survive into the filename, even if the redundancy
    # check above would otherwise have dropped them.
    def combined_lower():
        return ("_".join(prefix_words) + "_" + raw_original).lower()

    genre_words = words_of(genre)
    if genre_words and not (genre_words & words_of(combined_lower())):
        prefix_words.extend(w.lower() for w in re.split(r'[\s_\-]+', genre) if w)

    path_components = ([top] if top else []) + list(middle)
    if any('song' in c.lower() for c in path_components) and 'song' not in combined_lower():
        prefix_words.append('song')

    full_raw = "_".join(prefix_words) + "_" + raw_original if prefix_words else raw_original
    full_raw = apply_abbreviations(full_raw)
    full_raw = normalize_spacing(full_raw)
    full_raw = collapse_repeats(full_raw)
    full_raw = ILLEGAL_CHARS_RE.sub('', full_raw).strip('_ -') or "file"

    new_filename = f"{full_raw}{ext}"

    # keep filenames within common filesystem limits (255 bytes per component)
    max_stem = 255 - len(ext.encode('utf-8')) - 1
    stem, _ = os.path.splitext(new_filename)
    if len(stem.encode('utf-8')) > max_stem:
        stem = stem.encode('utf-8')[:max_stem].decode('utf-8', errors='ignore')
        new_filename = f"{stem}{ext}"

    new_dir = os.path.join(base, sanitize_folder_piece(company), sanitize_folder_piece(genre))
    return new_dir, new_filename


def make_unique(candidate, used_paths):
    if candidate not in used_paths and not os.path.exists(candidate):
        return candidate
    root, ext = os.path.splitext(candidate)
    i = 2
    while True:
        alt = f"{root}_{i}{ext}"
        if alt not in used_paths and not os.path.exists(alt):
            return alt
        i += 1


def phase_flatten_filenames(base, dry_run, preview_count=None, seed=None):
    print("\n=== Flatten: rename files into Company/Genre structure ===")
    all_files = []
    for root, _dirs, files in os.walk(base):
        for fname in files:
            all_files.append(os.path.join(root, fname))
    print(f"  Found {len(all_files)} files.")

    if preview_count:
        rng = random.Random(seed)
        sample = rng.sample(all_files, min(preview_count, len(all_files)))
        print(f"  Preview of {len(sample)} random renames (nothing will be changed):")
        for p in sample:
            new_dir, new_filename = compute_new_name(base, p)
            rel_old = os.path.relpath(p, base)
            rel_new = os.path.relpath(os.path.join(new_dir, new_filename), base)
            print(f"    {rel_old}")
            print(f"      -> {rel_new}")
        return 0

    used_paths = set()
    renamed = 0
    for p in all_files:
        new_dir, new_filename = compute_new_name(base, p)
        candidate = make_unique(os.path.join(new_dir, new_filename), used_paths)
        used_paths.add(candidate)
        if dry_run:
            continue
        try:
            os.makedirs(os.path.dirname(candidate), exist_ok=True)
            shutil.move(p, candidate)
            renamed += 1
        except OSError as e:
            print(f"  failed to move {p} -> {candidate}: {e}")

    if dry_run:
        print(f"  [DRY RUN] would rename {len(all_files)} files. Re-run with --execute to apply, "
              f"or use --preview N to sample results first.")
    else:
        print(f"  Renamed {renamed} files.")
        after_count = sum(len(files) for _root, _dirs, files in os.walk(base))
        if after_count == len(all_files):
            print(f"  Verified: file count unchanged ({len(all_files)} before, {after_count} after).")
        else:
            print(f"  WARNING: file count MISMATCH — {len(all_files)} before, {after_count} after. "
                  f"Investigate before trusting the result.")
        # every file just moved OUT of its old deep folder tree into COMPANY/GENRE/,
        # so the old numbered/"@" folder trees are now empty shells — clean them up.
        phase_remove_empty_dirs(base, dry_run=False, label="Flatten cleanup")
    return renamed


# -----------------------------------------------------------------------
# VERIFY-MIDI  (separate mode — see module docstring for the rules)
# -----------------------------------------------------------------------
def midi_load_error(path):
    """Try to load `path` with pretty_midi, the same way the training
    pipeline's cache builder does. Returns None if it loads fine, or a
    short string describing the error if it doesn't."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pretty_midi.PrettyMIDI(path)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def phase_verify_midi(base, dry_run, preview_count=None, seed=None):
    print("\n=== Verify MIDI: delete files pretty_midi cannot load ===")
    if not HAS_PRETTY_MIDI:
        print("  ERROR: pretty_midi is not installed — cannot verify files. "
              "Install it with: pip install pretty_midi")
        return 0

    candidates = []
    for root, _dirs, files in os.walk(base):
        for fname in files:
            if fname.lower().endswith((".mid", ".midi")):
                candidates.append(os.path.join(root, fname))

    print(f"  Scanning {len(candidates)} .mid/.midi files "
          f"(this loads every file with pretty_midi, so it takes a while)...")

    bad = []  # (path, error)
    n_scanned = 0
    for p in candidates:
        n_scanned += 1
        if n_scanned % 5000 == 0:
            print(f"  ...scanned {n_scanned}/{len(candidates)}")
        err = midi_load_error(p)
        if err is not None:
            bad.append((p, err))

    print(f"  Scanned {n_scanned} files. Faulty (unparseable): {len(bad)}")

    if preview_count:
        rng = random.Random(seed)
        sample = rng.sample(bad, min(preview_count, len(bad)))
        print(f"  Preview of {len(sample)} sampled faulty files (nothing will be deleted):")
        for p, err in sample:
            rel = os.path.relpath(p, base)
            print(f"    {rel}")
            print(f"      -> {err}")
        return 0

    for p, err in bad:
        rel = os.path.relpath(p, base)
        print(f"  {'[DRY RUN] would delete' if dry_run else 'deleting'}: {rel}  ({err})")

    if dry_run:
        print(f"  [DRY RUN] would delete {len(bad)} faulty files. Re-run with --execute to apply, "
              f"or use --preview N to sample results first.")
        return 0

    deleted = 0
    for p, _err in bad:
        try:
            os.remove(p)
            deleted += 1
        except OSError as e:
            print(f"  failed to delete {p}: {e}")
    print(f"  Deleted {deleted} faulty files.")
    return deleted


# -----------------------------------------------------------------------
# DRUM-TRACK SELECTION  (shared by --erase-junk's too_few_drum_events check
# and --erase-ambiguous)
# -----------------------------------------------------------------------
# Mirrors drum_humanizer_v3.py's _get_drum_notes() track-selection heuristic
# and its GM_DRUM_MAP pitch coverage, by hand. Keep these in sync with that
# file if either one ever changes.
_DRUM_NAME_RE = re.compile(r'drum|kit|perc|groove|beat', re.IGNORECASE)
_NON_DRUM_NAME_RE = re.compile(
    r'bass|guitar|piano|keys?|synth|vocal|lead|pad|string|brass|horn|organ|choir', re.IGNORECASE)

# Standard GM drum pitches drum_humanizer_v3.py's GM_DRUM_MAP recognizes
# (35-62 only - extended/custom articulation pitches outside this set are
# invisible to it, same caveat as that file's own comment on GM_DRUM_MAP).
GM_MAPPED_PITCHES = {35, 36, 38, 40, 37, 42, 44, 46, 41, 43, 45, 47, 48, 50,
                      49, 57, 51, 59, 53, 52, 55, 56, 58, 60, 62, 39}


def select_drum_notes(midi):
    """Same branching as drum_humanizer_v3.py's _get_drum_notes(): returns
    (notes, source) where source is 'channel10', 'single_track_fallback',
    'name_match_fallback', 'ambiguous_multi_track', or 'no_notes'."""
    drum_tracks = [t for t in midi.instruments if t.is_drum]
    if drum_tracks:
        return [n for t in drum_tracks for n in t.notes], 'channel10'
    non_empty = [t for t in midi.instruments if t.notes]
    if not non_empty:
        return [], 'no_notes'
    if len(non_empty) == 1:
        return list(non_empty[0].notes), 'single_track_fallback'
    named_drum = [t for t in non_empty
                  if _DRUM_NAME_RE.search(t.name or '') and not _NON_DRUM_NAME_RE.search(t.name or '')]
    if named_drum:
        return [n for t in named_drum for n in t.notes], 'name_match_fallback'
    return [], 'ambiguous_multi_track'


# -----------------------------------------------------------------------
# ERASE-JUNK  (separate mode — see module docstring for the rules)
# -----------------------------------------------------------------------
def measure_length_seconds(pm):
    """Seconds per measure at this file's first tempo/time-signature (good
    enough for a junk-length check; files with mid-song tempo/meter changes
    are rare in a groove-loop library and this only needs to be roughly
    right to catch genuinely tiny scraps)."""
    tempo = 120.0
    try:
        tc = pm.get_tempo_changes()
        if len(tc[1]) > 0:
            tempo = float(tc[1][0])
    except Exception:
        pass
    num, den = 4, 4
    try:
        if pm.time_signature_changes:
            ts0 = sorted(pm.time_signature_changes, key=lambda t: t.time)[0]
            num, den = ts0.numerator, ts0.denominator
    except Exception:
        pass
    beats_per_bar = float(num) * (4.0 / float(den)) if den else 4.0
    seconds_per_beat = 60.0 / max(1e-6, tempo)
    return max(1e-6, beats_per_bar * seconds_per_beat)


def analyze_junk(path, min_notes, min_drum_events, min_measure_fraction):
    """Load `path` once and return a list of junk reason-strings (empty list
    if it's fine). Returns None if pretty_midi can't load it at all - that's
    --verify-midi's job, not this mode's, so such files are just skipped
    here rather than counted as junk."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pm = pretty_midi.PrettyMIDI(path)
    except Exception:
        return None

    reasons = []
    total_notes = sum(len(inst.notes) for inst in pm.instruments)
    if total_notes == 0:
        reasons.append("no_notes_at_all")
    elif total_notes < min_notes:
        reasons.append(f"too_few_notes(<{min_notes})")

    # Separate, stricter check: how many notes would actually survive
    # drum_humanizer_v3.py's own channel-selection + GM_DRUM_MAP filtering
    # (its 'too_few_events' failure reason) - NOT a subset of total_notes
    # above, since a file can have plenty of raw notes but few (or none) on
    # the selected track that map to a standard GM drum pitch. Skipped for
    # 'ambiguous_multi_track' / 'no_notes' sources - those are --erase-
    # ambiguous's and "no_notes_at_all"'s job respectively, not this reason's.
    notes, source = select_drum_notes(pm)
    if source not in ('ambiguous_multi_track', 'no_notes'):
        mapped_count = sum(1 for n in notes if n.pitch in GM_MAPPED_PITCHES)
        if mapped_count < min_drum_events:
            reasons.append(f"too_few_drum_events(<{min_drum_events})")

    mlen = measure_length_seconds(pm)
    duration = pm.get_end_time()
    if duration / mlen < min_measure_fraction:
        reasons.append(f"shorter_than_{min_measure_fraction}_measure")

    return reasons


def phase_erase_junk_midi(base, dry_run, min_notes=10, min_drum_events=10, min_measure_fraction=0.5,
                           preview_count=None, seed=None):
    print(f"\n=== Erase junk MIDI: no notes / <{min_notes} notes / "
          f"<{min_drum_events} drum-mapped events / <{min_measure_fraction} of a measure ===")
    if not HAS_PRETTY_MIDI:
        print("  ERROR: pretty_midi is not installed — cannot analyze files. "
              "Install it with: pip install pretty_midi")
        return 0

    candidates = []
    for root, _dirs, files in os.walk(base):
        for fname in files:
            if fname.lower().endswith((".mid", ".midi")):
                candidates.append(os.path.join(root, fname))

    print(f"  Scanning {len(candidates)} .mid/.midi files "
          f"(this loads every file with pretty_midi, so it takes a while)...")

    junk = []  # (path, reasons)
    reason_counts = collections.Counter()
    n_scanned = 0
    n_unreadable = 0
    for p in candidates:
        n_scanned += 1
        if n_scanned % 10000 == 0:
            print(f"  ...scanned {n_scanned}/{len(candidates)}")
        reasons = analyze_junk(p, min_notes, min_drum_events, min_measure_fraction)
        if reasons is None:
            n_unreadable += 1
            continue
        if reasons:
            junk.append((p, reasons))
            for r in reasons:
                reason_counts[r] += 1

    print(f"  Scanned {n_scanned} files ({n_unreadable} unreadable, skipped — "
          f"see --verify-midi for those).")
    print(f"  Junk files: {len(junk)}")
    for reason, count in reason_counts.most_common():
        print(f"    {count:>8}  {reason}")

    if preview_count:
        rng = random.Random(seed)
        sample = rng.sample(junk, min(preview_count, len(junk)))
        print(f"  Preview of {len(sample)} sampled junk files (nothing will be deleted):")
        for p, reasons in sample:
            rel = os.path.relpath(p, base)
            print(f"    {rel}")
            print(f"      -> {', '.join(reasons)}")
        return 0

    for p, reasons in junk:
        rel = os.path.relpath(p, base)
        print(f"  {'[DRY RUN] would delete' if dry_run else 'deleting'}: {rel}  ({', '.join(reasons)})")

    if dry_run:
        print(f"  [DRY RUN] would delete {len(junk)} junk files. Re-run with --execute to apply, "
              f"or use --preview N to sample results first.")
        return 0

    deleted = 0
    for p, _reasons in junk:
        try:
            os.remove(p)
            deleted += 1
        except OSError as e:
            print(f"  failed to delete {p}: {e}")
    print(f"  Deleted {deleted} junk files.")
    return deleted


# -----------------------------------------------------------------------
# ERASE-AMBIGUOUS  (separate mode — see module docstring for the rules)
# -----------------------------------------------------------------------
def analyze_ambiguous(path):
    """Load `path` once and return True if it's ambiguous_multi_track, False
    otherwise. Returns None if pretty_midi can't load it at all - that's
    --verify-midi's job, not this mode's, so such files are skipped here."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pm = pretty_midi.PrettyMIDI(path)
    except Exception:
        return None
    _notes, source = select_drum_notes(pm)
    return source == 'ambiguous_multi_track'


def phase_erase_ambiguous_midi(base, dry_run, preview_count=None, seed=None):
    print("\n=== Erase ambiguous MIDI: multi-track files with no identifiable drum part ===")
    if not HAS_PRETTY_MIDI:
        print("  ERROR: pretty_midi is not installed — cannot analyze files. "
              "Install it with: pip install pretty_midi")
        return 0

    candidates = []
    for root, _dirs, files in os.walk(base):
        for fname in files:
            if fname.lower().endswith((".mid", ".midi")):
                candidates.append(os.path.join(root, fname))

    print(f"  Scanning {len(candidates)} .mid/.midi files "
          f"(this loads every file with pretty_midi, so it takes a while)...")

    ambiguous = []
    n_scanned = 0
    n_unreadable = 0
    for p in candidates:
        n_scanned += 1
        if n_scanned % 10000 == 0:
            print(f"  ...scanned {n_scanned}/{len(candidates)}")
        result = analyze_ambiguous(p)
        if result is None:
            n_unreadable += 1
            continue
        if result:
            ambiguous.append(p)

    print(f"  Scanned {n_scanned} files ({n_unreadable} unreadable, skipped — "
          f"see --verify-midi for those).")
    print(f"  Ambiguous (multi-track, no identifiable drum part): {len(ambiguous)}")

    if preview_count:
        rng = random.Random(seed)
        sample = rng.sample(ambiguous, min(preview_count, len(ambiguous)))
        print(f"  Preview of {len(sample)} sampled ambiguous files (nothing will be deleted):")
        for p in sample:
            rel = os.path.relpath(p, base)
            print(f"    {rel}")
        return 0

    for p in ambiguous:
        rel = os.path.relpath(p, base)
        print(f"  {'[DRY RUN] would delete' if dry_run else 'deleting'}: {rel}")

    if dry_run:
        print(f"  [DRY RUN] would delete {len(ambiguous)} ambiguous files. Re-run with --execute to "
              f"apply, or use --preview N to sample results first.")
        return 0

    deleted = 0
    for p in ambiguous:
        try:
            os.remove(p)
            deleted += 1
        except OSError as e:
            print(f"  failed to delete {p}: {e}")
    print(f"  Deleted {deleted} ambiguous files.")
    return deleted


# -----------------------------------------------------------------------
# MOVE-BY-MEASURES  (separate mode — see module docstring for the rules)
# -----------------------------------------------------------------------
def count_measures(path):
    """Approximate bar/measure count via pretty_midi's downbeat detection.
    Returns None if the file can't be parsed."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pm = pretty_midi.PrettyMIDI(path)
            return len(pm.get_downbeats())
    except Exception:
        return None


def build_path_folded_filename(base, path):
    """'GROOVE/SLOW BLUES A/groove_..._Hats.mid' -> 'GROOVE_SLOW_BLUES_A_groove_..._Hats.mid'
    Folds every directory level between `base` and the file into the
    filename, so moving it into a flat destination folder doesn't lose
    where it came from."""
    rel = os.path.relpath(path, base)
    parts = rel.split(os.sep)
    dirs, filename = parts[:-1], parts[-1]
    prefix = "_".join(sanitize_piece(d) for d in dirs if d)
    combined = f"{prefix}_{filename}" if prefix else filename
    return collapse_repeats(combined)


def phase_move_by_measures(base, dry_run, preview_count=None, seed=None):
    print("\n=== Move by measures: _songs/ (>64 bars, 'song' in path) + _g48/ (>48 bars) ===")
    if not HAS_PRETTY_MIDI:
        print("  ERROR: pretty_midi is not installed — cannot count measures. "
              "Install it with: pip install pretty_midi")
        return 0

    songs_dir = os.path.join(base, "_songs")
    g48_dir = os.path.join(base, "_g48")
    skip_tops = {"_songs", "_g48"}

    candidates = []
    for root, dirs, files in os.walk(base):
        rel_root = os.path.relpath(root, base)
        top = rel_root.split(os.sep)[0] if rel_root != "." else ""
        if top in skip_tops:
            dirs[:] = []  # don't descend into the destination folders
            continue
        for fname in files:
            if fname.lower().endswith((".mid", ".midi")):
                candidates.append(os.path.join(root, fname))

    print(f"  Scanning {len(candidates)} .mid/.midi files for measure counts "
          f"(this parses every file, so it takes a while)...")

    decisions = []  # (path, destination_dir, measures)
    n_scanned = 0
    n_failed = 0
    for p in candidates:
        n_scanned += 1
        if n_scanned % 10000 == 0:
            print(f"  ...scanned {n_scanned}/{len(candidates)}")
        measures = count_measures(p)
        if measures is None:
            n_failed += 1
            continue
        rel = os.path.relpath(p, base).lower()
        if "song" in rel and measures > 64:
            decisions.append((p, songs_dir, measures))
        elif measures > 48:
            decisions.append((p, g48_dir, measures))

    n_songs = sum(1 for _p, d, _m in decisions if d == songs_dir)
    n_g48 = sum(1 for _p, d, _m in decisions if d == g48_dir)
    print(f"  Scanned {n_scanned} files ({n_failed} unreadable, skipped).")
    print(f"  -> _songs/ : {n_songs} files (>64 bars, 'song' in path)")
    print(f"  -> _g48/   : {n_g48} files (>48 bars)")

    if preview_count:
        rng = random.Random(seed)
        sample = rng.sample(decisions, min(preview_count, len(decisions)))
        print(f"  Preview of {len(sample)} random moves (nothing will be changed):")
        for p, dest_dir, measures in sample:
            new_filename = build_path_folded_filename(base, p)
            rel_old = os.path.relpath(p, base)
            rel_new = os.path.relpath(os.path.join(dest_dir, new_filename), base)
            print(f"    [{measures} bars] {rel_old}")
            print(f"      -> {rel_new}")
        return 0

    if dry_run:
        print(f"  [DRY RUN] would move {len(decisions)} files. Re-run with --execute to apply, "
              f"or use --preview N to sample results first.")
        return 0

    before_total = sum(len(files) for _root, _dirs, files in os.walk(base))

    used_paths = set()
    moved = 0
    for p, dest_dir, _measures in decisions:
        new_filename = build_path_folded_filename(base, p)
        candidate = make_unique(os.path.join(dest_dir, new_filename), used_paths)
        used_paths.add(candidate)
        try:
            os.makedirs(os.path.dirname(candidate), exist_ok=True)
            shutil.move(p, candidate)
            moved += 1
        except OSError as e:
            print(f"  failed to move {p} -> {candidate}: {e}")

    print(f"  Moved {moved} files.")
    after_total = sum(len(files) for _root, _dirs, files in os.walk(base))
    if after_total == before_total:
        print(f"  Verified: total file count unchanged ({before_total} before, {after_total} after).")
    else:
        print(f"  WARNING: file count MISMATCH — {before_total} before, {after_total} after. "
              f"Investigate before trusting the result.")
    return moved


def phase_move_above_measures(base, dry_run, threshold, dest_name, exclude_tops=(),
                               preview_count=None, seed=None):
    """General single-criterion version of the _songs/_g48 mover above: move
    every .mid/.midi file longer than `threshold` bars into a new top-level
    `dest_name`/ folder, excluding files already under any of `exclude_tops`
    (and under `dest_name` itself, so reruns are safe). Same path-folded
    filename convention, same never-overwrite/uniqueness handling, same
    before/after file-count verification."""
    print(f"\n=== Move by measures: {dest_name}/ (>{threshold} bars) ===")
    if not HAS_PRETTY_MIDI:
        print("  ERROR: pretty_midi is not installed — cannot count measures. "
              "Install it with: pip install pretty_midi")
        return 0

    dest_dir = os.path.join(base, dest_name)
    skip_tops = set(exclude_tops) | {dest_name}

    candidates = []
    for root, dirs, files in os.walk(base):
        rel_root = os.path.relpath(root, base)
        top = rel_root.split(os.sep)[0] if rel_root != "." else ""
        if top in skip_tops:
            dirs[:] = []
            continue
        for fname in files:
            if fname.lower().endswith((".mid", ".midi")):
                candidates.append(os.path.join(root, fname))

    print(f"  Scanning {len(candidates)} .mid/.midi files "
          f"(excluding {', '.join(sorted(skip_tops))}/) for measure counts...")

    decisions = []  # (path, measures)
    n_scanned = 0
    n_failed = 0
    for p in candidates:
        n_scanned += 1
        if n_scanned % 10000 == 0:
            print(f"  ...scanned {n_scanned}/{len(candidates)}")
        measures = count_measures(p)
        if measures is None:
            n_failed += 1
            continue
        if measures > threshold:
            decisions.append((p, measures))

    print(f"  Scanned {n_scanned} files ({n_failed} unreadable, skipped).")
    print(f"  -> {dest_name}/ : {len(decisions)} files (>{threshold} bars)")

    if preview_count:
        rng = random.Random(seed)
        sample = rng.sample(decisions, min(preview_count, len(decisions)))
        print(f"  Preview of {len(sample)} random moves (nothing will be changed):")
        for p, measures in sample:
            new_filename = build_path_folded_filename(base, p)
            rel_old = os.path.relpath(p, base)
            rel_new = os.path.relpath(os.path.join(dest_dir, new_filename), base)
            print(f"    [{measures} bars] {rel_old}")
            print(f"      -> {rel_new}")
        return 0

    if dry_run:
        print(f"  [DRY RUN] would move {len(decisions)} files. Re-run with --execute to apply, "
              f"or use --preview N to sample results first.")
        return 0

    before_total = sum(len(files) for _root, _dirs, files in os.walk(base))

    used_paths = set()
    moved = 0
    for p, _measures in decisions:
        new_filename = build_path_folded_filename(base, p)
        candidate = make_unique(os.path.join(dest_dir, new_filename), used_paths)
        used_paths.add(candidate)
        try:
            os.makedirs(os.path.dirname(candidate), exist_ok=True)
            shutil.move(p, candidate)
            moved += 1
        except OSError as e:
            print(f"  failed to move {p} -> {candidate}: {e}")

    print(f"  Moved {moved} files.")
    after_total = sum(len(files) for _root, _dirs, files in os.walk(base))
    if after_total == before_total:
        print(f"  Verified: total file count unchanged ({before_total} before, {after_total} after).")
    else:
        print(f"  WARNING: file count MISMATCH — {before_total} before, {after_total} after. "
              f"Investigate before trusting the result.")
    return moved


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_dir", help="Path to the MIDI library root (e.g. 'data')")
    parser.add_argument("--execute", action="store_true",
                         help="Actually delete files/folders. Without this flag, it's a dry run.")
    parser.add_argument("--keep-format", choices=["sd3", "ezd"], default=KEEP_FORMAT,
                         help="Which plugin-format copy to keep in format-duality pairs "
                              "(default: %(default)s)")
    parser.add_argument("--flatten", action="store_true",
                         help="Run ONLY the filename-flattening rename (Company/Genre/renamed_file), "
                              "instead of the cleanup phases. Combine with --preview to sample "
                              "results first, or --execute to actually rename everything.")
    parser.add_argument("--preview", type=int, default=None, metavar="N",
                         help="With --flatten and without --execute: print N randomly sampled "
                              "proposed renames without changing anything.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for --preview sampling (reproducible previews).")
    parser.add_argument("--verify-midi", action="store_true",
                         help="Run ONLY a scan of every .mid/.midi file, loading it with "
                              "pretty_midi and deleting any file it can't parse (e.g. corrupt "
                              "tick counts), instead of the cleanup phases. Combine with "
                              "--preview to sample faulty files first, or --execute to "
                              "actually delete them.")
    parser.add_argument("--move-by-measures", action="store_true",
                         help="Run ONLY the measures-based move (_songs/ for >64-bar files with "
                              "'song' in their path, _g48/ for >48-bar files), instead of the "
                              "cleanup phases. Combine with --preview to sample results first, "
                              "or --execute to actually move everything.")
    parser.add_argument("--move-g24", action="store_true",
                         help="Run ONLY a move of every .mid/.midi file (not already under "
                              "_songs/ or _g48/) longer than 24 bars into a new top-level _g24/ "
                              "folder, same path-folded-filename convention. Combine with "
                              "--preview to sample results first, or --execute to actually "
                              "move everything.")
    parser.add_argument("--erase-junk", action="store_true",
                         help="Run ONLY a scan-and-delete of files with no notes at all, fewer "
                              "than --min-notes notes, fewer than --min-drum-events GM-mapped "
                              "drum events, or shorter than --min-measure-fraction of a measure, "
                              "instead of the cleanup phases. Combine with --preview to sample "
                              "results first, or --execute to actually delete them.")
    parser.add_argument("--min-notes", type=int, default=10, metavar="N",
                         help="--erase-junk: delete files with fewer than N notes total "
                              "(default: %(default)s).")
    parser.add_argument("--min-drum-events", type=int, default=10, metavar="N",
                         help="--erase-junk: delete files with fewer than N notes that survive "
                              "drum_humanizer_v3.py's own channel-selection + GM_DRUM_MAP "
                              "filtering (matches its 'too_few_events' threshold, default: "
                              "%(default)s).")
    parser.add_argument("--min-measure-fraction", type=float, default=0.5, metavar="X",
                         help="--erase-junk: delete files shorter than X of one measure at "
                              "their own tempo/time-signature (default: %(default)s, i.e. half "
                              "a bar).")
    parser.add_argument("--erase-ambiguous", action="store_true",
                         help="Run ONLY a scan-and-delete of multi-track files where no track is "
                              "flagged or named as drums (drum_humanizer_v3.py's cache builder "
                              "would skip these as 'ambiguous_multi_track' every run anyway), "
                              "instead of the cleanup phases. Combine with --preview to sample "
                              "results first, or --execute to actually delete them.")
    args = parser.parse_args()

    base = os.path.abspath(args.base_dir)
    if not os.path.isdir(base):
        print(f"Error: not a directory: {base}")
        sys.exit(1)

    dry_run = not args.execute
    print(f"Base directory: {base}")
    print(f"Mode: {'DRY RUN (no changes will be made)' if dry_run else 'EXECUTE (files will be deleted)'}")

    if args.flatten:
        if args.preview and not args.execute:
            phase_flatten_filenames(base, dry_run=True, preview_count=args.preview, seed=args.seed)
        else:
            phase_flatten_filenames(base, dry_run=dry_run)
            if dry_run:
                print("\nDry run complete. Re-run with --execute to actually rename anything.")
            else:
                print("\nDone.")
        return

    if args.verify_midi:
        if args.preview and not args.execute:
            phase_verify_midi(base, dry_run=True, preview_count=args.preview, seed=args.seed)
        else:
            phase_verify_midi(base, dry_run=dry_run)
            if dry_run:
                print("\nDry run complete. Re-run with --execute to actually delete anything.")
            else:
                print("\nDone.")
        return

    if args.move_by_measures:
        if args.preview and not args.execute:
            phase_move_by_measures(base, dry_run=True, preview_count=args.preview, seed=args.seed)
        else:
            phase_move_by_measures(base, dry_run=dry_run)
            if dry_run:
                print("\nDry run complete. Re-run with --execute to actually move anything.")
            else:
                print("\nDone.")
        return

    if args.move_g24:
        exclude = ("_songs", "_g48")
        if args.preview and not args.execute:
            phase_move_above_measures(base, dry_run=True, threshold=24, dest_name="_g24",
                                       exclude_tops=exclude, preview_count=args.preview, seed=args.seed)
        else:
            phase_move_above_measures(base, dry_run=dry_run, threshold=24, dest_name="_g24",
                                       exclude_tops=exclude)
            if dry_run:
                print("\nDry run complete. Re-run with --execute to actually move anything.")
            else:
                print("\nDone.")
        return

    if args.erase_junk:
        if args.preview and not args.execute:
            phase_erase_junk_midi(base, dry_run=True, min_notes=args.min_notes,
                                   min_drum_events=args.min_drum_events,
                                   min_measure_fraction=args.min_measure_fraction,
                                   preview_count=args.preview, seed=args.seed)
        else:
            phase_erase_junk_midi(base, dry_run=dry_run, min_notes=args.min_notes,
                                   min_drum_events=args.min_drum_events,
                                   min_measure_fraction=args.min_measure_fraction)
            if dry_run:
                print("\nDry run complete. Re-run with --execute to actually delete anything.")
            else:
                print("\nDone.")
        return

    if args.erase_ambiguous:
        if args.preview and not args.execute:
            phase_erase_ambiguous_midi(base, dry_run=True, preview_count=args.preview, seed=args.seed)
        else:
            phase_erase_ambiguous_midi(base, dry_run=dry_run)
            if dry_run:
                print("\nDry run complete. Re-run with --execute to actually delete anything.")
            else:
                print("\nDone.")
        return

    print(f"Keep format: {args.keep_format}")

    phase_delete_vir2(base, dry_run)
    phase_delete_genres(base, dry_run)
    phase_delete_house_and_electronic(base, dry_run)
    phase_dedupe_by_hash(base, dry_run, keep_format=args.keep_format)
    phase_remove_empty_dirs(base, dry_run)
    phase_delete_header_files(base, dry_run)
    phase_remove_empty_dirs(base, dry_run, label="Phase 7")

    if dry_run:
        print("\nDry run complete. Re-run with --execute to actually delete anything.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
