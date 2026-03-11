#!/usr/bin/env python3
"""
test_recipes.py — Automated regression testing for Spark AI Tone Engineer.

Runs known-good songs through the full pipeline and asserts recipe values
match expected outputs. Catches regressions immediately when formulas change.

Usage:
    python test_recipes.py                    # run all tests
    python test_recipes.py cherubrock         # run specific test
    python test_recipes.py --stems /path/to   # specify stems directory

Requires: stems for test songs in the stems directory.
If a stem is missing, that test is skipped (not failed).
"""

import subprocess
import json
import sys
import os
import glob

# ==========================================
# KNOWN-GOOD RECIPES (ground truth from amp testing)
# ==========================================
# Each entry defines the expected recipe output for a song.
# Values come from confirmed amp-tested settings or validated recipe outputs.
# Tolerance: ±0.2 for most params, ±0.5 for EQ bands (more sensitive to DSP variation).

KNOWN_GOOD = {
    "cherubrock": {
        "stem_patterns": ["cherubrock.wav", "cherub_rock.wav", "*cherub*rock*.wav"],
        "description": "Cherub Rock — Muff → Plexi (DIALED IN)",
        "checks": {
            # Guitar
            "guitar.guitar": ("Mexican Strat (SSH)", "exact"),
            "guitar.pickup": ("Bridge Humbucker", "exact"),
            # Rig selections
            "rig.gate": ("Noise Gate", "exact"),
            "rig.drive": ("Guitar Muff", "exact"),
            "rig.amp": ("Plexiglas", "exact"),
            "rig.mod_eq": ("Guitar EQ", "exact"),
            "rig.reverb": ("Plate Short", "exact"),
            # Drive settings (tested at amp)
            "settings.drive.SUSTAIN": (6.5, 0.1),
            "settings.drive.TONE": (7.8, 0.1),
            "settings.drive.VOLUME": (5.5, 0.1),
            # Amp settings (tested at amp)
            "settings.amp.GAIN": (6.5, 0.1),
            "settings.amp.BASS": (5.0, 0.1),
            "settings.amp.MIDDLE": (4.0, 0.1),
            "settings.amp.TREBLE": (9.0, 0.1),
            "settings.amp.VOLUME": (8.2, 0.1),
            # EQ (tested at amp — these are the calibration anchors)
            "settings.mod_eq.100HZ": (-0.5, 0.5),
            "settings.mod_eq.200HZ": (-0.4, 0.5),
            "settings.mod_eq.400HZ": (-1.7, 0.5),
            "settings.mod_eq.800HZ": (-1.0, 0.5),
            "settings.mod_eq.1600HZ": (-3.8, 0.5),
            "settings.mod_eq.3200HZ": (4.7, 0.5),
        }
    },
    "eruption": {
        "stem_patterns": ["eruption.wav", "*eruption*.wav"],
        "description": "Eruption — Tube Drive → Plexi (DIALED IN)",
        "checks": {
            # Guitar
            "guitar.guitar": ("Mexican Strat (SSH)", "exact"),
            "guitar.pickup": ("Bridge Humbucker", "exact"),
            # Rig selections
            "rig.drive": ("Tube Drive", "exact"),
            "rig.amp": ("Plexiglas", "exact"),
            "rig.mod_eq": ("Guitar EQ", "exact"),
            # Drive settings (tested at amp)
            "settings.drive.OVERDRIVE": (1.7, 0.1),
            "settings.drive.TONE": (6.5, 0.1),
            "settings.drive.LEVEL": (8.0, 0.1),
            # Amp settings (tested at amp)
            "settings.amp.GAIN": (7.7, 0.1),
            "settings.amp.BASS": (6.0, 0.1),
            "settings.amp.MIDDLE": (4.1, 0.1),
            "settings.amp.TREBLE": (9.2, 0.1),
            "settings.amp.VOLUME": (7.8, 0.1),
            # EQ (tested at amp)
            "settings.mod_eq.100HZ": (0.5, 0.5),
            "settings.mod_eq.200HZ": (0.6, 0.5),
            "settings.mod_eq.400HZ": (-0.1, 0.5),
            "settings.mod_eq.800HZ": (0.4, 0.5),
            "settings.mod_eq.1600HZ": (-0.5, 0.5),
            "settings.mod_eq.3200HZ": (3.2, 0.5),
        }
    },
}


def find_stem(patterns, stems_dir):
    """Find a stem file matching any of the given patterns."""
    for pattern in patterns:
        matches = glob.glob(os.path.join(stems_dir, "**", pattern), recursive=True)
        if matches:
            return matches[0]
    return None


def get_nested(data, dotpath):
    """Get a value from nested dict using dot notation: 'settings.drive.TONE'"""
    keys = dotpath.split('.')
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def run_test(song_key, test_def, stems_dir):
    """Run a single regression test. Returns (passed, failed, skipped) counts."""
    desc = test_def["description"]
    patterns = test_def["stem_patterns"]
    checks = test_def["checks"]

    # Find stem
    stem_path = find_stem(patterns, stems_dir)
    if not stem_path:
        print(f"\n⏭️  {desc}")
        print(f"   Stem not found (searched: {', '.join(patterns)})")
        return 0, 0, 1

    print(f"\n🔬 {desc}")
    print(f"   Stem: {stem_path}")

    # Run main.py
    result = subprocess.run(
        [sys.executable, "main.py", stem_path],
        capture_output=True, text=True, timeout=120
    )

    if result.returncode != 0:
        print(f"   ❌ main.py failed:")
        print(f"   {result.stderr[:200]}")
        return 0, 1, 0

    # Find the JSON recipe
    stem_name = os.path.splitext(os.path.basename(stem_path))[0]
    clean_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in stem_name)
    json_path = os.path.join("recipes", f"{clean_name}.json")

    if not os.path.exists(json_path):
        # Try partial match
        candidates = glob.glob(os.path.join("recipes", f"*{song_key}*.json"))
        if candidates:
            json_path = candidates[0]
        else:
            print(f"   ❌ JSON recipe not found: {json_path}")
            return 0, 1, 0

    with open(json_path, 'r') as f:
        recipe = json.load(f)

    # Run checks
    passed = 0
    failed = 0

    for dotpath, expected in sorted(checks.items()):
        actual = get_nested(recipe, dotpath)

        if isinstance(expected, tuple) and expected[1] == "exact":
            # Exact string match
            expected_val = expected[0]
            if actual == expected_val:
                passed += 1
            else:
                failed += 1
                print(f"   ❌ {dotpath}: expected '{expected_val}', got '{actual}'")

        elif isinstance(expected, tuple):
            # Numeric with tolerance
            expected_val, tolerance = expected
            if actual is None:
                failed += 1
                print(f"   ❌ {dotpath}: expected {expected_val}, got None (missing)")
            else:
                try:
                    delta = abs(float(actual) - expected_val)
                    if delta <= tolerance:
                        passed += 1
                    else:
                        failed += 1
                        print(f"   ❌ {dotpath}: expected {expected_val} ±{tolerance}, got {actual} (Δ={delta:.2f})")
                except (ValueError, TypeError):
                    failed += 1
                    print(f"   ❌ {dotpath}: expected {expected_val}, got '{actual}' (not numeric)")

    if failed == 0:
        print(f"   ✅ All {passed} checks passed")
    else:
        print(f"   ⚠️ {passed} passed, {failed} FAILED")

    return passed, failed, 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Spark AI Tone Engineer — Regression Tests")
    parser.add_argument("songs", nargs="*", help="Specific song keys to test (default: all)")
    parser.add_argument("--stems", type=str, default=".", help="Directory containing stems")

    args = parser.parse_args()

    stems_dir = args.stems
    if not os.path.exists(stems_dir):
        print(f"❌ Stems directory not found: {stems_dir}")
        return

    # Select tests to run
    if args.songs:
        tests = {k: v for k, v in KNOWN_GOOD.items() if k in args.songs}
        if not tests:
            print(f"❌ Unknown song(s): {', '.join(args.songs)}")
            print(f"   Available: {', '.join(KNOWN_GOOD.keys())}")
            return
    else:
        tests = KNOWN_GOOD

    print(f"🎸 SPARK AI TONE ENGINEER — REGRESSION TESTS")
    print(f"   Songs: {len(tests)} | Stems dir: {stems_dir}")
    print(f"{'='*60}")

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for song_key, test_def in tests.items():
        p, f, s = run_test(song_key, test_def, stems_dir)
        total_passed += p
        total_failed += f
        total_skipped += s

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 RESULTS: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")

    if total_failed > 0:
        print(f"❌ REGRESSION DETECTED")
        sys.exit(1)
    elif total_skipped == len(tests):
        print(f"⚠️ All tests skipped (no stems found)")
    else:
        print(f"✅ ALL TESTS PASSED")


if __name__ == "__main__":
    main()
