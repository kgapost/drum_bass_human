#!/usr/bin/env python3
"""
==============================================================================
 CONFIG — every named constant used by drum_bass_studio.py
==============================================================================
This file holds NOTHING but constants: no functions, no classes, no imports
beyond what a value literally needs (none, here). It exists so every number
that reflects a DESIGN DECISION or a JUDGMENT CALL — as opposed to a pure UI
cosmetic choice like a window's pixel geometry — lives in exactly one place,
named, uppercase, with a comment explaining the reasoning behind the specific
value chosen. This includes the INITIAL VALUES of every slider/checkbox the
GUI exposes, not just the thresholds used inside the phase-processing logic —
so a programmer reviewing this file sees the complete set of decisions the
tool currently makes on the user's behalf, in one screen, without having to
hunt through widget-construction code to find out what a fresh segment starts
at.

Several of these were resolved by developer intuition during design (the
weak-beat damping ratio, the feel-variation-to-temperature mapping, the
segmentation confidence threshold, the sliders' own default positions, ...)
rather than derived from a hard constraint. Those are explicitly marked
"JUDGMENT CALL" below, as distinct from the few that are genuine hard
technical constraints (marked "VERIFIED FINDING" / "HARD TECHNICAL
CONSTRAINT") which should not be casually retuned without re-verifying
whatever they were derived from.
"""

# =============================================================================
# PHASE 2: RUSH / DRAG
# =============================================================================

# "1/128 note" == 1/32 of a beat (confirmed reading). This is the sliders' FULL
# travel at |slider|==1; the constant is expressed as a FRACTION OF A BEAT, not
# a fixed tick count, so it stays correct regardless of a file's tempo/PPQ.
PHASE2_MAX_RUSH_DRAG_BEATS = 1.0 / 32

# Beats 2 and 4 ("the main beats" minus 1 and 3) get LESS rush/drag effect than
# the e/a syncopation points — JUDGMENT CALL, set to 25% LOWER than full
# strength (i.e. multiplied by 0.75) per the design discussion: e/a positions
# are where a rushing/dragging FEEL is most audible, while 2 and 4 are meant to
# move along with that feel more subtly rather than as strongly. Retune this if
# a stronger or weaker contrast between "syncopation" and "weak beat" is wanted.
PHASE2_WEAK_BEAT_DAMPING = 0.75

# Vestigial: an earlier design considered varying WHICH positions get selected
# based on how many e/a hits a segment actually has (skip 2/4 if e/a alone are
# dense enough). The final decision was "always select e/a/2/4, unconditionally"
# — so this constant is currently UNUSED by classify_phase2_role(). Kept as a
# named placeholder in case a future revision wants density-conditional
# selection back; safe to delete if that's not wanted.
PHASE2_DENSITY_UNUSED_THRESHOLD = 2

# ── Phase 2 round-trip safety ────────────────────────────────────────────────
# VERIFIED FINDING: drum_humanizer_v3's load_midi_events() re-derives each
# note's (bar, grid_step, offset_ticks) from its ABSOLUTE TIME by snapping to
# the NEAREST grid cell. Directly probed the exact boundary (write a note at a
# known grid_step with increasing offset_ticks, reload, check whether
# grid_step survives): stable through 16 ticks, reassigned to the NEIGHBORING
# grid cell starting at 17 ticks (cfg.ticks_per_grid=30, so 17 > half the cell).
# PHASE2_SAFE_OFFSET_FRACTION is set to land the clip AT the slider's own
# nominal max (15 ticks = PHASE2_MAX_RUSH_DRAG_BEATS * ticks_per_beat), one
# tick inside the verified-safe boundary — so a full-strength (|slider|=1.0)
# rush/drag on a note with NO pre-existing offset reaches its full nominal
# effect uncapped, while a note that ALREADY carries a large offset (from
# Phase 1, or a previous Phase 2 pass) still gets safely capped before its
# combined total could cross into reassignment territory. This is a HARD
# TECHNICAL CONSTRAINT (verified empirically), not a taste-based judgment call
# — do not raise it above ~0.53 (16 ticks) without re-verifying the boundary.
PHASE2_SAFE_OFFSET_FRACTION = 0.5   # of cfg.ticks_per_grid = 15 ticks (verified safe to 16)

# ── Phase 2 slider default positions (JUDGMENT CALL: both start at "no change") ──
PHASE2_DEFAULT_RUSH_DRAG = 0.0   # a fresh segment's rush/drag sliders start untouched
PHASE2_DEFAULT_QUANTIZE = 0.0    # a fresh segment's quantize sliders start untouched

# ── Phase 2 slider travel ranges (shown in the GUI; also used as the ttk.Scale
# from_/to bounds) ────────────────────────────────────────────────────────────
# Rush/drag is signed (drag before the grid .. rush after it); quantize is a
# one-directional strength dial (0=untouched .. 1=fully snapped), so its range
# does NOT go negative — there's no meaningful "negative quantize."
PHASE2_RUSH_DRAG_RANGE = (-1.0, 1.0)
PHASE2_QUANTIZE_RANGE = (0.0, 1.0)


# =============================================================================
# PHASE 3: BASS SYNC
# =============================================================================

# "1/256 note" == 1/64 of a beat (same note-duration convention as Phase 2).
# JUDGMENT CALL: this is deliberately TIGHTER than Phase 2's rush/drag range
# (1/64 beat vs 1/32) — a bass note should only be considered "aimed at" a
# kick/snare hit if it's genuinely close, not loosely nearby; a wider threshold
# risks snapping a bass note that was actually intended as a passing tone
# between two drum hits (see the earlier discussion on kick-vs-passing-note
# bass behavior across genres).
PHASE3_SNAP_THRESHOLD_BEATS = 1.0 / 64

# Audibility-delay range endpoints (milliseconds). At slider=0 -> 0ms (off,
# exactly, as a special case — see run_phase3_sync). As the slider rises toward
# 1, the random draw range widens from [LOW_MIN, LOW_MAX] up to
# [HIGH_MIN, HIGH_MAX], interpolated linearly (_interpolated_delay_range_ms).
# JUDGMENT CALL: 1-2ms is roughly the smallest delay a listener can register at
# all; 8-10ms (widened from an original 2-5ms during design review) is large
# enough to clearly separate the bass attack from the drum transient without
# sounding like a timing error. Retune if the "separation" effect should be
# subtler or more pronounced.
PHASE3_DELAY_LOW_MIN_MS = 1.0
PHASE3_DELAY_LOW_MAX_MS = 2.0
PHASE3_DELAY_HIGH_MIN_MS = 8.0
PHASE3_DELAY_HIGH_MAX_MS = 10.0

# ── Phase 3 slider default positions (JUDGMENT CALL: both start at "no effect") ──
PHASE3_DEFAULT_SNAP_STRENGTH = 0.0   # a fresh segment's bass starts unsnapped
PHASE3_DEFAULT_DELAY_AMOUNT = 0.0    # a fresh segment's bass starts with no added delay

# ── Phase 3 slider travel ranges (both are one-directional strength dials —
# there's no meaningful "negative snap" or "negative delay") ───────────────────
PHASE3_SNAP_STRENGTH_RANGE = (0.0, 1.0)
PHASE3_DELAY_AMOUNT_RANGE = (0.0, 1.0)


# =============================================================================
# PHASE 1: DRUM HUMANIZE
# =============================================================================

PHASE1_DEFAULT_STRENGTH = 1.0            # 1.0 = exactly what the model predicts (see
                                          # drum_humanizer_v3's own --strength docs: 0=dry,
                                          # 1=model, >1=exaggerate past the model)
# JUDGMENT CALL: the strength slider's travel range. 0=dry input untouched,
# 1=exactly the model's output, up to 2=extrapolate twice as far past the
# model as it moved the note (drum_humanizer_v3 itself allows even higher, but
# 2.0 is a sane ceiling for a GUI slider before results get extreme).
PHASE1_STRENGTH_RANGE = (0.0, 2.0)

PHASE1_DEFAULT_INTENSITY = None          # None = auto-derive from the input file's own
                                          # average velocity (loud in -> loud out); this is
                                          # the SETTINGS default fed to humanize_file(),
                                          # distinct from the intensity SLIDER's own resting
                                          # value below (which only matters once "Auto" is
                                          # unchecked)
# JUDGMENT CALL: whether the "Auto intensity" checkbox starts CHECKED for a
# fresh segment. True by default — auto-deriving intensity from the input is
# the safer, more predictable starting point; a user who wants to override it
# to a specific target energy (see drum_humanizer_v3's own --intensity docs)
# unchecks this deliberately.
PHASE1_DEFAULT_INTENSITY_AUTO = True
# JUDGMENT CALL: the intensity SLIDER's own initial numeric position (only
# actually used once "Auto intensity" is unchecked). 0.5 = the exact midpoint
# of its 0..1 range — a neutral starting point that doesn't bias the user
# toward "loud" or "soft" before they've listened to anything.
PHASE1_DEFAULT_INTENSITY_VALUE = 0.5
PHASE1_INTENSITY_RANGE = (0.0, 1.0)

PHASE1_DEFAULT_FEEL_VARIATION = 0.7      # JUDGMENT CALL: slightly above the slider's
                                          # midpoint by default, so a fresh segment starts
                                          # with clearly audible (not overly cautious) but
                                          # not maximal randomness in the model's sampling
PHASE1_FEEL_VARIATION_RANGE = (0.0, 1.0)

# JUDGMENT CALL: maps the single 0..1 "feel variation" slider onto BOTH of the
# humanizer's own temperature knobs at once. Expressed as a base (drum_
# humanizer_v3's own stock default) plus a spread, so slider=0.5 exactly
# reproduces the humanizer's default feel, and the "centered on the model's
# own defaults" property can't silently drift out of sync with a comment.
PHASE1_BASE_TEMPERATURE_VEL = 0.8    # drum_humanizer_v3's own --temperature_vel default
PHASE1_BASE_TEMPERATURE_OFF = 0.6    # drum_humanizer_v3's own --temperature_off default
PHASE1_TEMP_VEL_SPREAD = 0.4         # feel_variation swings +/- this much around the base
PHASE1_TEMP_OFF_SPREAD = 0.3         # feel_variation swings +/- this much around the base
PHASE1_FEEL_VARIATION_TO_TEMP_VEL = (PHASE1_BASE_TEMPERATURE_VEL - PHASE1_TEMP_VEL_SPREAD,
                                     PHASE1_BASE_TEMPERATURE_VEL + PHASE1_TEMP_VEL_SPREAD)
PHASE1_FEEL_VARIATION_TO_TEMP_OFF = (PHASE1_BASE_TEMPERATURE_OFF - PHASE1_TEMP_OFF_SPREAD,
                                     PHASE1_BASE_TEMPERATURE_OFF + PHASE1_TEMP_OFF_SPREAD)

PHASE1_DEFAULT_FAST_HIT_CAP = False      # off by default -- a RULE (not learned), see
                                          # drum_humanizer_v3's own --fast_hit_cap docs;
                                          # only useful for very fast/dense passages


# =============================================================================
# SEGMENTATION  (drum_theme_segmentation.py model inference)
# =============================================================================

# JUDGMENT CALL: how confident the segmentation model must be (predicted
# probability) before a measure is called a theme boundary. 0.5 is the natural
# "more likely than not" midpoint; raise it to make the tool more conservative
# (fewer, more confident boundaries) or lower it to catch more subtle theme
# changes at the cost of more false positives.
SEGMENTATION_CONFIDENCE_THRESHOLD = 0.5

# How much consecutive analysis windows overlap when segmenting a song longer
# than the model's own max_seq_len, as a fraction of that window. Higher gives
# every measure more surrounding context (better boundary decisions) at the
# cost of more (redundant) model inference; see drum_theme_segmentation.py's
# own context_overlap discussion for the full rationale.
SEGMENTATION_CONTEXT_OVERLAP = 0.25


# =============================================================================
# INSTRUMENT GROUPS
# =============================================================================

# The 5-group taxonomy (kick / snare / toms / everything-else) used throughout
# Phase 2's per-instrument sliders, matching find_similar_grooves.py's grouping
# with cymbals+hihat+perc collapsed into one "other" bucket (per the design
# discussion — only kick/snare/toms were judged to need independent control).
# Values are drum_humanizer_v3's DRUM_CLASS_NAMES ids: 0=kick, 1=snare,
# 2/3=hihat closed/open, 4/5/6=low/mid/high tom, 7-10=crash/ride/bell/china,
# 11/12=perc/other.
GROUP_KICK  = (0,)
GROUP_SNARE = (1,)
GROUP_TOMS  = (4, 5, 6)
GROUP_OTHER = (2, 3, 7, 8, 9, 10, 11, 12)   # hihat + cymbals + perc + other


# =============================================================================
# SEGMENT TIMELINE (visual)
# =============================================================================

# JUDGMENT CALL: below this width, a segment rectangle would be too thin to
# reliably click on a long song with many short segments — floor it at 2px
# regardless of how small its true duration-proportional share would be. This
# means very short segments in a long song read as slightly WIDER than their
# exact proportional share (a deliberate, minor accuracy-for-usability trade).
SEGMENT_MIN_RECT_WIDTH_PX = 2
SEGMENT_OUTLINE_WIDTH_NORMAL = 2     # unselected segment border
SEGMENT_OUTLINE_WIDTH_SELECTED = 4   # selected segment border -- thicker, so the
                                      # currently-active segment is unambiguous at a glance

SEGMENT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
                  "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
                  # a fixed palette cycled by segment index -- 10 colors is enough that
                  # adjacent segments are essentially never the same color by chance
SELECTED_OUTLINE_COLOR = "#000000"    # the currently-selected segment's border color
CUSTOMIZED_MARKER_COLOR = "#FFD400"   # small dot drawn on any segment whose settings
                                      # differ from defaults in any of the 3 phases

# JUDGMENT CALL: whether each phase's collapsible section starts open (visible)
# or closed (collapsed) when the app launches. Phase 1 starts open since it's
# always the first thing a new segment needs; Phase 2/3 start closed both
# because they're gated (disabled until their prerequisite phase completes,
# see StudioApp._refresh_gating) and to keep the window's initial height down.
PHASE1_SECTION_STARTS_OPEN = True
PHASE2_SECTION_STARTS_OPEN = False
PHASE3_SECTION_STARTS_OPEN = False


# =============================================================================
# MISC
# =============================================================================

GLOBAL_SEED = 42   # single seed for every RNG in this script (Phase 3's random
                   # post-hit delay draw), so results are reproducible run-to-run
