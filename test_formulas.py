#!/usr/bin/env python3
"""
test_formulas.py — Unit tests for Spark AI formula math.

Tests the core calculations in isolation — no audio, no stems, runs instantly.
Catches formula regressions, broken constants, and edge cases.

Usage:
    python test_formulas.py
    python test_formulas.py -v    # verbose: show passing tests too
"""

import sys
import json
import math

# ── Constants (must match main.py) ──────────────────────────────────────────
AIR_MIN, AIR_MAX   = 0.328, 0.618
AIR_RANGE          = AIR_MAX - AIR_MIN          # 0.290
MIDS_MIN, MIDS_MAX = 0.30, 0.65
MIDS_RANGE         = MIDS_MAX - MIDS_MIN        # 0.35
BASS_MIN, BASS_MAX = 0.20, 0.50
BASS_RANGE         = BASS_MAX - BASS_MIN        # 0.30

AMP_KNOB_COVERAGE  = 0.35
K_air, K_mids = 42.0, 31.0


# ── Helpers ──────────────────────────────────────────────────────────────────
def eq_residuals(target_air, target_mids, target_bass, amp_bt, drive_sp=None):
    """Compute EQ band values using the calibrated formula from main.py.

    Bass (100/200Hz): drive compensation only — no residual.
      The amp BASS knob tracks target_low_energy directly, so the residual
      is near-zero for all non-preset songs.
    Mids (400/800/1600Hz): residual × K × per-band factor. No mid_comp.
      Per-band factors calibrated against Cherub Rock + Eruption:
        400Hz=0.46, 800Hz=0.27, 1600Hz=1.00
    Treble (3200Hz): residual × K − treble_comp.
      treble_comp sign: negative = boost (cut_bass drives), positive = reduce.
    """
    amp_predicted_air  = AIR_MIN  + amp_bt.get('treble', 0.5) * AIR_RANGE
    amp_predicted_mids = MIDS_MIN + amp_bt.get('mids',   0.5) * MIDS_RANGE

    air_r  = (target_air  - amp_predicted_air)  * (1.0 - AMP_KNOB_COVERAGE)
    mids_r = (target_mids - amp_predicted_mids) * (1.0 - AMP_KNOB_COVERAGE)

    bass_comp = treble_comp = 0.0

    if drive_sp:
        gb = drive_sp.get('gain_boost', 0)

        if drive_sp.get('cut_bass'):
            bass_comp    = -gb * 1.0
            treble_comp  = -gb * 1.26
        elif drive_sp.get('retains_low_end'):
            bass_comp    = gb * 0.80
            treble_comp  = gb * 0.14
        elif drive_sp.get('scooped') or drive_sp.get('wooly'):
            bass_comp    = gb * 0.56
            treble_comp  = gb * 0.14
        # else: both remain 0.0

    def clip(v): return max(-10.0, min(10.0, v))

    return {
        '100HZ':  round(clip(-bass_comp),              1),
        '200HZ':  round(clip(-bass_comp * 0.8),        1),
        '400HZ':  round(clip(mids_r * K_mids * 0.46),  1),
        '800HZ':  round(clip(mids_r * K_mids * 0.27),  1),
        '1600HZ': round(clip(mids_r * K_mids * 1.00),  1),
        '3200HZ': round(clip(air_r  * K_air - treble_comp), 1),
    }


def gain_sigmoid(g):
    return 1.0 / (1.0 + math.exp(-8.0 * (g - 0.45)))


# ── Test runner ──────────────────────────────────────────────────────────────
passed = failed = 0
verbose = '-v' in sys.argv

def check(name, actual, expected, tol=0.01):
    global passed, failed
    if isinstance(expected, str):
        ok = (actual == expected)
    else:
        ok = abs(actual - expected) <= tol
    if ok:
        passed += 1
        if verbose: print(f"  ✅ {name}: {actual}")
    else:
        failed += 1
        print(f"  ❌ {name}: expected {expected}, got {actual}")

def section(title):
    print(f"\n── {title} {'─'*(55-len(title))}")


# ════════════════════════════════════════════════════════════
# 1. NORMALIZATION CONSTANTS
# ════════════════════════════════════════════════════════════
section("Normalization constants")
check("AIR_RANGE",  AIR_RANGE,  0.290, tol=0.001)
check("MIDS_RANGE", MIDS_RANGE, 0.350, tol=0.001)
check("BASS_RANGE", BASS_RANGE, 0.300, tol=0.001)


# ════════════════════════════════════════════════════════════
# 2. EQ RESIDUAL CORE PROPERTY: perfect amp match → zero EQ
# ════════════════════════════════════════════════════════════
section("EQ residual: perfect amp match → near-zero EQ")

# AC Boost (Vox AC30): treble=0.80, mids=0.70, bass=0.35
ac30 = {'treble': 0.80, 'mids': 0.70, 'bass': 0.35}
# Target that exactly matches AC30's predicted output
target_air_match  = AIR_MIN  + 0.80 * AIR_RANGE   # 0.560
target_mids_match = MIDS_MIN + 0.70 * MIDS_RANGE  # 0.545
target_bass_match = BASS_MIN + 0.35 * BASS_RANGE  # 0.305

eq = eq_residuals(target_air_match, target_mids_match, target_bass_match, ac30)
check("Exact match (no drive): 3200Hz = 0", eq['3200HZ'], 0.0, tol=0.1)
check("Exact match (no drive): 1600Hz = 0", eq['1600HZ'], 0.0, tol=0.1)
# Bass bands: no drive means bass_comp=0, so 100Hz=0 regardless of target
check("Exact match (no drive): 100Hz = 0",  eq['100HZ'],  0.0, tol=0.1)


# ════════════════════════════════════════════════════════════
# 3. EQ DIRECTION: bright target with dark amp → positive 3200Hz
# ════════════════════════════════════════════════════════════
section("EQ direction: bright target + dark amp → boost treble")

marshall = {'treble': 0.40, 'mids': 0.60, 'bass': 0.50}  # Plexiglas
bright_target_air = 0.58  # Brian May territory

eq_dark_amp = eq_residuals(bright_target_air, 0.50, 0.30, marshall)
check("Dark amp + bright target: 3200Hz > 0", eq_dark_amp['3200HZ'] > 0, True)
check("Dark amp: 3200Hz >= 2 dB", eq_dark_amp['3200HZ'] >= 2.0, True)

# Inverse: dark target with bright amp → negative 3200Hz
eq_bright_amp = eq_residuals(AIR_MIN + 0.1, 0.40, 0.30, ac30)
check("Bright amp + dark target: 3200Hz < 0", eq_bright_amp['3200HZ'] < 0, True)


# ════════════════════════════════════════════════════════════
# 4. EQ DIRECTION: scooped mids → mid bands cut
# ════════════════════════════════════════════════════════════
section("EQ direction: scooped mids → negative mid bands")

eq_scooped = eq_residuals(0.48, 0.32, 0.35, ac30)  # Very scooped mids
check("Scooped target: 1600Hz < 0", eq_scooped['1600HZ'] < 0, True)
check("Scooped target: 800Hz < 0",  eq_scooped['800HZ']  < 0, True)

eq_forward = eq_residuals(0.48, 0.65, 0.35, marshall)  # Mid-forward target, scooped amp
check("Mid-forward target + scooped amp: 1600Hz > 0", eq_forward['1600HZ'] > 0, True)


# ════════════════════════════════════════════════════════════
# 5. DRIVE COMPENSATION: bass_comp direction per drive type
# ════════════════════════════════════════════════════════════
section("Drive compensation: bass and treble direction per drive")

muff_sp      = {'gain_boost': 0.9, 'mids': -0.5, 'treble': -0.2, 'scooped': True}
tube_drv_sp  = {'gain_boost': 0.5, 'mids':  0.4, 'treble': -0.1, 'cut_bass': True}
neutral_sp   = {'gain_boost': 0.5, 'mids':  0.1, 'treble':  0.0}

eq_no_drive   = eq_residuals(0.48, 0.38, 0.32, marshall)
eq_with_muff  = eq_residuals(0.48, 0.38, 0.32, marshall, drive_sp=muff_sp)
eq_with_tube  = eq_residuals(0.48, 0.38, 0.32, marshall, drive_sp=tube_drv_sp)
eq_neutral    = eq_residuals(0.48, 0.38, 0.32, marshall, drive_sp=neutral_sp)

# Muff (scooped, retains bass) → 100Hz CUTS (negative)
check("Muff: 100Hz < 0 (scooped drives retain bass → EQ cuts)",
      eq_with_muff['100HZ'] < 0, True)

# Tube Drive (cut_bass) → 100Hz BOOSTS (positive)
check("Tube Drive: 100Hz > 0 (drive cuts bass → EQ restores)",
      eq_with_tube['100HZ'] > 0, True)

# Muff cuts more bass than neutral drive
check("Muff 100Hz < neutral drive 100Hz",
      eq_with_muff['100HZ'] < eq_neutral['100HZ'], True)

# Tube Drive boosts 3200Hz beyond no-drive (treble_comp negative)
check("Tube Drive: 3200Hz > no-drive 3200Hz",
      eq_with_tube['3200HZ'] >= eq_no_drive['3200HZ'], True)

# Neutral drive leaves EQ unchanged (bass_comp=0, treble_comp=0)
check("Neutral drive: 100Hz = no-drive 100Hz",
      eq_neutral['100HZ'] == eq_no_drive['100HZ'], True)


# ════════════════════════════════════════════════════════════
# 6. TREBLE COMPENSATION: scooped/wooly drives have tiny positive
#    comp; cut_bass drives have negative comp (EQ boosts)
# ════════════════════════════════════════════════════════════
section("treble_comp: drive flags modulate 3200Hz")

booster_sp   = {'gain_boost': 0.2, 'mids': 0.0, 'treble':  0.1}
fuzz_face_sp = {'gain_boost': 0.7, 'mids': -0.1, 'treble': -0.2, 'wooly': True}

eq_booster   = eq_residuals(0.58, 0.50, 0.32, marshall, drive_sp=booster_sp)
eq_fuzz      = eq_residuals(0.58, 0.50, 0.32, marshall, drive_sp=fuzz_face_sp)

# Booster (no flags) → treble_comp=0; Fuzz Face (wooly) → treble_comp=+0.098
# Same residual → Booster gets slightly more 3200Hz boost
check("Booster leaves >= 3200Hz vs wooly Fuzz Face",
      eq_booster['3200HZ'] >= eq_fuzz['3200HZ'], True)

# cut_bass drive gives more 3200Hz boost than neutral (treble_comp is negative)
eq_cutbass = eq_residuals(0.58, 0.50, 0.32, marshall, drive_sp=tube_drv_sp)
eq_no_drv  = eq_residuals(0.58, 0.50, 0.32, marshall)
check("cut_bass drive: 3200Hz >= no-drive 3200Hz",
      eq_cutbass['3200HZ'] >= eq_no_drv['3200HZ'], True)


# ════════════════════════════════════════════════════════════
# 7. EQ BOUNDS: all bands stay within ±10 dB
# ════════════════════════════════════════════════════════════
section("EQ bounds: all bands within ±10 dB")

extreme_cases = [
    (AIR_MAX, 0.65, 0.50, {'treble': 0.20, 'mids': 0.30, 'bass': 0.85}),  # max bright, dark amp
    (AIR_MIN, 0.30, 0.20, {'treble': 0.90, 'mids': 0.70, 'bass': 0.35}),  # max dark, bright amp
    (AIR_MAX, 0.65, 0.50, {'treble': 0.20, 'mids': 0.30, 'bass': 0.85},
     {'gain_boost': 0.9, 'mids': -0.5, 'treble': -0.2, 'scooped': True}), # + heavy Muff
]
for i, case in enumerate(extreme_cases):
    drive = case[4] if len(case) > 4 else None
    eq = eq_residuals(case[0], case[1], case[2], case[3], drive)
    for band, val in eq.items():
        check(f"Case {i+1} {band} in bounds", -10.0 <= val <= 10.0, True)


# ════════════════════════════════════════════════════════════
# 8. AMP SELECTION DISTANCE: sigmoid + weighting
# ════════════════════════════════════════════════════════════
section("Amp selection: sigmoid gain distance")

check("gain_sigmoid(0.0) < 0.5", gain_sigmoid(0.0) < 0.5, True)
check("gain_sigmoid(0.45) ≈ 0.5", gain_sigmoid(0.45), 0.5, tol=0.01)
check("gain_sigmoid(1.0) > 0.5", gain_sigmoid(1.0) > 0.5, True)
check("Sigmoid: high-gain spread", gain_sigmoid(0.8) - gain_sigmoid(0.5) > 0.1, True)

# Clean amps (low gain) should have lower sigmoid than high-gain
check("Clean (0.1) < Crunch (0.6) in sigmoid",
      gain_sigmoid(0.1) < gain_sigmoid(0.6), True)


# ════════════════════════════════════════════════════════════
# 9. GEAR DATABASE INTEGRITY
# ════════════════════════════════════════════════════════════
section("Gear database integrity")

try:
    with open('spark_gear.json', 'r') as f:
        db = json.load(f)

    # All amps have required base_tone fields
    all_amps = [m for cat in db['amps'] for m in cat['models']]
    for amp in all_amps:
        bt = amp['base_tone']
        for field in ('gain', 'treble', 'mids', 'bass'):
            check(f"{amp['name']}: base_tone.{field} exists",
                  field in bt, True)
        # All values in [0,1]
        for field, val in bt.items():
            check(f"{amp['name']}: {field} in [0,1]",
                  0.0 <= val <= 1.0, True)

    # All drives have sonic_profile with required fields
    for drive in db['drive']:
        sp = drive.get('sonic_profile', {})
        check(f"{drive['name']}: has gain_boost",  'gain_boost' in sp, True)
        check(f"{drive['name']}: has mids",        'mids' in sp,       True)
        check(f"{drive['name']}: has treble",      'treble' in sp,     True)
        check(f"{drive['name']}: gain_boost in [0,1]",
              0.0 <= sp.get('gain_boost', 0) <= 1.0, True)

    check("All amps have bass field",
          all('bass' in m['base_tone'] for m in all_amps), True)

    check("Drive count = 14", len(db['drive']), 14)
    check("Amp count = 35",   len(all_amps),    35)

except FileNotFoundError:
    print("  ⏭️  spark_gear.json not found — skipping DB tests")


# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"📊 {passed} passed, {failed} failed")
if failed:
    print("❌ FAILURES DETECTED")
    sys.exit(1)
else:
    print("✅ ALL FORMULA TESTS PASSED")
