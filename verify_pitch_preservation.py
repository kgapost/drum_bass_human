#!/usr/bin/env python3
"""
==============================================================================
 VERIFY PITCH PRESERVATION  (drum_humanizer_v3.py --mode infer)
==============================================================================
Empirically confirms that humanize_file() never changes which note/pitch a
hit has — only its velocity and micro-timing. This backs up the guarantee
made in drum_humanizer_v3.py's own docstring and code comments (search that
file for "Pitch is ALWAYS the original note the user played"): the model's
architecture has no pitch/instrument output head at all (only velocity_head
and offset_head — see class HumanizationTransformer), so pitch can't be
predicted or altered even in principle.

This script compares an INPUT MIDI against its humanized OUTPUT MIDI
(produced by `drum_humanizer_v3.py --mode infer`) and asserts, note-for-note,
that every pitch is identical. It also reports the velocity/timing deltas
that DID change, as a sanity check that humanization actually did something.

USAGE
-----
  python verify_pitch_preservation.py input.mid humanized_output.mid
"""

import os
import sys

import pretty_midi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drum_humanizer_v3 import GM_DRUM_MAP  # same recognized-pitch filter the loader uses


def load_notes(path, filter_to_gm_map=False):
    """filter_to_gm_map=True mirrors _load_midi_events_impl(): notes whose
    pitch isn't in GM_DRUM_MAP are dropped BEFORE humanization ever sees
    them — that's a separate "unrecognized pitch" filter, not the model
    changing a pitch, so the INPUT side needs the same filter applied
    before comparing note-for-note against the OUTPUT."""
    pm = pretty_midi.PrettyMIDI(path)
    notes = [n for inst in pm.instruments for n in inst.notes]
    if filter_to_gm_map:
        notes = [n for n in notes if n.pitch in GM_DRUM_MAP]
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.mid humanized_output.mid")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    before_raw = load_notes(input_path)
    before = load_notes(input_path, filter_to_gm_map=True)
    after = load_notes(output_path)

    if len(before_raw) != len(before):
        print(f"Note: {len(before_raw) - len(before)} of {len(before_raw)} input notes have a "
              f"pitch not in GM_DRUM_MAP and are dropped before humanization ever runs — "
              f"that's a separate filter, not the model altering a pitch. Comparing against "
              f"the {len(before)} recognized notes only.\n")

    if len(before) != len(after):
        print(f"FAIL: note count changed — {len(before)} recognized input notes, "
              f"{len(after)} in output.")
        sys.exit(1)

    pitch_mismatches = []
    vel_deltas = []
    time_deltas_ms = []
    for i, (nb, na) in enumerate(zip(before, after)):
        if nb.pitch != na.pitch:
            pitch_mismatches.append((i, nb.pitch, na.pitch))
        vel_deltas.append(na.velocity - nb.velocity)
        time_deltas_ms.append((na.start - nb.start) * 1000)

    print(f"Compared {len(before)} notes between:\n  input:  {input_path}\n  output: {output_path}\n")

    if pitch_mismatches:
        print(f"FAIL: {len(pitch_mismatches)} pitch(es) changed:")
        for i, p_before, p_after in pitch_mismatches[:20]:
            print(f"  note {i}: pitch {p_before} -> {p_after}")
        sys.exit(1)

    print("PASS: every pitch is identical between input and output.\n")
    print(f"Velocity delta:  min={min(vel_deltas):+d}  max={max(vel_deltas):+d}  "
          f"mean={sum(vel_deltas)/len(vel_deltas):+.1f}")
    print(f"Timing delta:    min={min(time_deltas_ms):+.1f}ms  max={max(time_deltas_ms):+.1f}ms  "
          f"mean={sum(time_deltas_ms)/len(time_deltas_ms):+.1f}ms")
    print("\n(Non-zero velocity/timing deltas above are expected — that's the")
    print(" humanization working. Zero pitch mismatches is the guarantee being tested.)")


if __name__ == "__main__":
    main()
