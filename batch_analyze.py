#!/usr/bin/env python3
"""
batch_analyze.py — Scan all WAV files and dump raw DSP features + v4 gain.

Usage:
    python batch_analyze.py                     # scans current directory
    python batch_analyze.py /path/to/wavs       # scans specific directory
    python batch_analyze.py file1.wav file2.wav  # specific files

Outputs a tab-separated table to terminal AND saves to batch_results.csv
"""

import sys
import os
import glob
import csv
import librosa
import numpy as np
from scipy.signal import butter, lfilter
from scipy.stats import kurtosis as calc_kurtosis

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def extract_features(file_path):
    """Extract ALL raw features from a wav file and compute v4 gain."""
    try:
        y_stereo, sr = librosa.load(file_path, mono=False)
        y_stereo = librosa.util.normalize(y_stereo)

        if y_stereo.ndim == 1:
            y_stereo = np.array([y_stereo, y_stereo])

        y_left, y_right = y_stereo[0], y_stereo[1]
        mid = (y_left + y_right) / 2
        side = (y_left - y_right) / 2
        width = np.clip((np.sqrt(np.mean(side**2)) + 1e-6) / (np.sqrt(np.mean(mid**2)) + 1e-6), 0, 1.0)

        y_raw = mid
        y_filtered = butter_bandpass_filter(y_raw, 450, 6800, sr)

        # === LEGACY FEATURES (retained for reference) ===
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y_filtered)))
        amplitude_env = librosa.feature.rms(y=y_filtered, frame_length=2048, hop_length=512)[0]
        crest_factor = float(np.max(amplitude_env) / (np.mean(amplitude_env) + 1e-6))
        normalized_crest = np.clip(crest_factor / 16.0, 0, 1.0)
        raw_gain = 1.0 - normalized_crest

        # Harmonic multiplier (v0 formula component, retained for raw_spark_gain reference)
        base_multiplier = 0.15 + (flatness * 350)
        harmonic_multiplier = float(np.clip(base_multiplier, 0.15, 0.95))
        raw_spark_gain = float(np.clip((raw_gain ** 1.2) * harmonic_multiplier, 0, 1.0))

        # Kurtosis (v4 feature)
        kurt = float(calc_kurtosis(y_filtered))

        # High-band flatness (legacy)
        high_band = butter_bandpass_filter(y_raw, 2000, 6800, sr)
        hbf = float(np.mean(librosa.feature.spectral_flatness(y=high_band)))

        # Energy bands (legacy)
        low_band = butter_bandpass_filter(y_raw, 80, 500, sr)
        mid_band = butter_bandpass_filter(y_raw, 500, 2000, sr)
        energy_low = float(np.mean(low_band**2))
        energy_mid = float(np.mean(mid_band**2))
        energy_high = float(np.mean(high_band**2))
        total_energy = energy_low + energy_mid + energy_high + 1e-10
        her = energy_high / total_energy

        # Zero-crossing rate (v4 feature)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y_filtered)))

        # Spectral centroid and mids
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_filtered, sr=sr)))
        treble = np.clip(centroid / 3500, 0, 1)

        S = np.abs(librosa.stft(y_filtered))
        freqs = librosa.fft_frequencies(sr=sr)
        total_e = np.sum(S) + 1e-6

        mids_range     = (freqs >= 500)  & (freqs <= 2000)
        presence_range = (freqs >= 1200) & (freqs <= 2400)
        air_range      = (freqs >= 2400) & (freqs <= 6800)

        # Global means (legacy, retained for comparison)
        mids     = float(np.clip((np.sum(S[mids_range,     :]) / total_e) * 1.5, 0, 1))
        presence = float(np.sum(S[presence_range, :]) / total_e)
        air      = float(np.sum(S[air_range,      :]) / total_e)

        # Energy-weighted loud-frame analysis (matches main.py)
        # Weights toward loud frames (top 50% by energy), solving the
        # multi-section averaging problem. A song like Comfortably Numb
        # (clean verse + screaming solo) gets measurements dominated by
        # the solo, not the average of both.
        frame_energy = np.sum(S, axis=0)
        energy_median = np.median(frame_energy)
        loud_mask = frame_energy > energy_median
        if np.sum(loud_mask) < 2:
            loud_mask = np.ones(S.shape[1], dtype=bool)

        S_loud = S[:, loud_mask]
        loud_energy = frame_energy[loud_mask]
        loud_weights = loud_energy / (np.sum(loud_energy) + 1e-10)

        frame_totals = np.sum(S_loud, axis=0) + 1e-10
        mids_per_frame     = np.sum(S_loud[mids_range, :], axis=0) / frame_totals
        presence_per_frame = np.sum(S_loud[presence_range, :], axis=0) / frame_totals
        air_per_frame      = np.sum(S_loud[air_range, :], axis=0) / frame_totals

        ew_mids     = float(np.clip(np.sum(mids_per_frame * loud_weights) * 1.5, 0, 1))
        ew_presence = float(np.sum(presence_per_frame * loud_weights))
        ew_air      = float(np.sum(air_per_frame * loud_weights))

        # === v4 NEW FEATURES ===

        # Spectral contrast — upper bands (v4 feature)
        contrast = librosa.feature.spectral_contrast(y=y_filtered, sr=sr, n_bands=6)
        contrast_mean = np.mean(contrast, axis=1)
        sc_high = float(np.mean(contrast_mean[3:6]))

        # Harmonic-to-percussive energy ratio (v4 feature)
        y_harmonic, y_percussive = librosa.effects.hpss(y_filtered)
        harmonic_energy = np.mean(y_harmonic**2)
        percussive_energy = np.mean(y_percussive**2) + 1e-10
        hp_ratio = float(harmonic_energy / percussive_energy)

        # RMS coefficient of variation (v4 feature)
        rms_cv = float(np.std(amplitude_env) / (np.mean(amplitude_env) + 1e-10))

        # === HARMONIC SERIES ANALYSIS ===
        # Measures even/odd harmonic balance to characterize distortion type.
        # Uses y_raw (unfiltered) for pitch tracking — y_filtered cuts fundamentals below A4.
        try:
            f0_harm, voiced_flag_harm, _ = librosa.pyin(
                y_raw, fmin=60, fmax=1400, sr=sr,
                frame_length=2048, hop_length=512
            )
            S_harm = np.abs(librosa.stft(y_raw, n_fft=4096, hop_length=512))
            freqs_harm = librosa.fft_frequencies(sr=sr, n_fft=4096)
            freq_res = freqs_harm[1] - freqs_harm[0]

            n_hf = min(len(f0_harm), S_harm.shape[1], len(amplitude_env))
            f0_harm = f0_harm[:n_hf]
            voiced_flag_harm = voiced_flag_harm[:n_hf]
            S_harm = S_harm[:, :n_hf]
            rms_harm = amplitude_env[:n_hf]

            rms_med = np.median(rms_harm)
            valid_harm = voiced_flag_harm & (rms_harm > rms_med * 0.5) & np.isfinite(f0_harm)
            harm_coverage = float(np.sum(valid_harm) / max(n_hf, 1))
            harm_ratio = 1.0
            harm_rolloff = 0.5

            if np.sum(valid_harm) >= 20:
                valid_idx = np.where(valid_harm)[0]
                f_even, f_odd, f_lower, f_upper, f_w = [], [], [], [], []

                for idx in valid_idx:
                    f0_val = f0_harm[idx]
                    spectrum = S_harm[:, idx]
                    h_energy = {}
                    for h in range(2, 7):
                        hf = f0_val * h
                        if hf > sr * 0.45:
                            break
                        bc = int(round(hf / freq_res))
                        if bc < 2 or bc >= len(freqs_harm) - 2:
                            continue
                        h_energy[h] = float(np.sum(spectrum[bc-2:bc+3]**2))

                    if len(h_energy) < 4:
                        continue

                    even_e = sum(h_energy.get(h, 0) for h in [2, 4, 6])
                    odd_e = sum(h_energy.get(h, 0) for h in [3, 5])
                    if odd_e > 1e-10 and even_e > 1e-10:
                        f_even.append(even_e)
                        f_odd.append(odd_e)
                        f_w.append(float(rms_harm[idx]))

                    lower_e = sum(h_energy.get(h, 0) for h in [2, 3])
                    upper_e = sum(h_energy.get(h, 0) for h in [4, 5, 6])
                    if lower_e > 1e-10:
                        f_lower.append(lower_e)
                        f_upper.append(upper_e)

                if len(f_w) >= 10:
                    w = np.array(f_w)
                    w = w / (np.sum(w) + 1e-10)
                    harm_ratio = float(np.clip(np.sum(np.array(f_even) * w) / (np.sum(np.array(f_odd) * w) + 1e-10), 0.1, 10.0))
                    if len(f_lower) >= 10:
                        w_ro = np.array(f_w[:len(f_lower)])
                        w_ro = w_ro / (np.sum(w_ro) + 1e-10)
                        harm_rolloff = float(np.clip(np.sum(np.array(f_upper) * w_ro) / (np.sum(np.array(f_lower) * w_ro) + 1e-10), 0.0, 5.0))

        except Exception:
            harm_ratio = 1.0
            harm_rolloff = 0.5
            harm_coverage = 0.0

        # === v4 UNIFIED GAIN ===
        v4_gain = float(np.clip(
            2.1855 * zcr
          - 0.0327 * hp_ratio
          - 0.2666 * kurt
          - 0.1359 * sc_high
          + 1.2657 * rms_cv
          + 2.3766
        , 0.0, 1.0))

        # === v3 GAIN (for comparison) ===
        is_clean_zone = (kurt > 0.5) and (hbf < 0.005)
        SAT_THRESH = 0.775

        if is_clean_zone:
            kurt_norm = np.clip(kurt / 2.5, 0, 1)
            ceiling = 0.495 + (1.0 - kurt_norm) * 0.13
            ceiling += her * 0.20
            ceiling -= max(0, flatness - 0.0015) * 25.0
            ceiling = np.clip(ceiling, 0.48, 0.80)
            corrected = min(raw_spark_gain, ceiling)
            gain_min, gain_max = 0.52, 0.82
            stretch = (0.95 - 0.20) / (gain_max - gain_min)
            v3_gain = float(np.clip((corrected - gain_min) * stretch + 0.20, 0.0, 1.0))
            tier = "T1-CLEAN"
        elif raw_spark_gain < SAT_THRESH:
            gain_min, gain_max = 0.52, 0.82
            stretch = (0.95 - 0.20) / (gain_max - gain_min)
            v3_gain = float(np.clip((raw_spark_gain - gain_min) * stretch + 0.20, 0.0, 1.0))
            tier = "T2-UNSAT"
        else:
            v3_gain = float(np.clip(raw_spark_gain * 0.6320 + flatness * 13.64 + 0.1031, 0.0, 1.0))
            tier = "T3-SAT"

        # Duration
        duration = len(y_raw) / sr

        return {
            'file': os.path.basename(file_path),
            'duration': round(duration, 1),
            # v4 input features
            'zcr': round(zcr, 4),
            'hp_ratio': round(hp_ratio, 2),
            'kurt': round(kurt, 2),
            'sc_high': round(sc_high, 2),
            'rms_cv': round(rms_cv, 4),
            # v4 output
            'v4_gain': round(v4_gain, 3),
            # v3 comparison
            'v3_gain': round(v3_gain, 3),
            'v3_tier': tier,
            # Legacy features (still useful for diagnostics)
            'raw': round(raw_spark_gain, 4),
            'flat': round(flatness, 5),
            'hbf': round(hbf, 5),
            'her': round(her, 4),
            'crest': round(crest_factor, 2),
            'treble': round(float(treble), 2),
            # Global means (legacy)
            'mids': round(mids, 2),
            'presence': round(presence, 3),
            'air': round(air, 3),
            # Energy-weighted (matches main.py recipe engine)
            'ew_mids': round(ew_mids, 3),
            'ew_presence': round(ew_presence, 3),
            'ew_air': round(ew_air, 3),
            # Harmonic series analysis
            'harm_ratio': round(harm_ratio, 3),
            'harm_rolloff': round(harm_rolloff, 3),
            'harm_coverage': round(harm_coverage, 3),
            'width': round(width, 2),
        }

    except Exception as e:
        return {'file': os.path.basename(file_path), 'error': str(e)}

def find_wav_files(paths):
    """Find all wav files from given paths (files or directories)."""
    wav_files = []
    for path in paths:
        if os.path.isfile(path) and path.lower().endswith('.wav'):
            wav_files.append(path)
        elif os.path.isdir(path):
            wav_files.extend(sorted(glob.glob(os.path.join(path, '**', '*.wav'), recursive=True)))
    return wav_files

def main():
    # Determine input
    if len(sys.argv) < 2:
        paths = ['.']
    else:
        paths = sys.argv[1:]

    wav_files = find_wav_files(paths)

    if not wav_files:
        print("No .wav files found.")
        print("Usage: python batch_analyze.py [directory or file paths]")
        return

    print(f"\n🔬 BATCH ANALYSIS (v4) — {len(wav_files)} files")
    print("=" * 120)

    results = []
    errors = []

    for i, f in enumerate(wav_files):
        print(f"  [{i+1}/{len(wav_files)}] {os.path.basename(f)}...", end=" ", flush=True)
        r = extract_features(f)
        if 'error' in r:
            print(f"❌ {r['error']}")
            errors.append(r)
        else:
            delta = r['v4_gain'] - r['v3_gain']
            arrow = "↑" if delta > 0.05 else "↓" if delta < -0.05 else "≈"
            print(f"✅ v4={r['v4_gain']:.2f} (v3={r['v3_gain']:.2f} {arrow}) | zcr={r['zcr']:.3f} hp={r['hp_ratio']:.1f} kurt={r['kurt']:.2f} sc={r['sc_high']:.1f} cv={r['rms_cv']:.3f}")
            results.append(r)

    if not results:
        print("\nNo files processed successfully.")
        return

    # Sort by v4_gain
    results.sort(key=lambda x: x['v4_gain'])

    # =============================================
    # MAIN TABLE: v4 vs v3 comparison
    # =============================================
    print(f"\n{'='*130}")
    print("RESULTS — sorted by v4 gain (v3 shown for comparison)")
    print(f"{'='*130}")

    print(f"  {'File':<40} {'v4':>5} {'v3':>5} {'Δ':>6} {'v3tier':<10} {'zcr':>6} {'hp':>5} {'kurt':>6} {'sc_hi':>6} {'cv':>6} {'mids*':>6} {'air*':>6} {'width':>5}")
    print("  " + "-" * 130)
    print("  (* = energy-weighted, matches recipe engine)")

    for r in results:
        delta = r['v4_gain'] - r['v3_gain']
        flag = ""
        if abs(delta) > 0.20:
            flag = " ⚡"  # Major shift
        elif abs(delta) > 0.10:
            flag = " △"   # Notable shift

        fname = r['file'][:38]
        print(f"  {fname:<40} {r['v4_gain']:>5.2f} {r['v3_gain']:>5.2f} {delta:>+6.2f} {r['v3_tier']:<10} {r['zcr']:>6.3f} {r['hp_ratio']:>5.1f} {r['kurt']:>6.2f} {r['sc_high']:>6.1f} {r['rms_cv']:>6.3f} {r['ew_mids']:>6.3f} {r['ew_air']:>6.3f} {r['width']:>5.2f}{flag}")

    # =============================================
    # SUMMARY STATISTICS
    # =============================================
    print(f"\n{'='*130}")
    print("SUMMARY STATISTICS")
    print(f"{'='*130}")

    v4_gains = [r['v4_gain'] for r in results]
    v3_gains = [r['v3_gain'] for r in results]

    print(f"\n  {'':>15} {'v4 Gain':>10} {'v3 Gain':>10}")
    print("  " + "-" * 35)
    for label, fn in [('Min', np.min), ('Mean', np.mean), ('Median', np.median), ('Max', np.max), ('Std', np.std)]:
        print(f"  {label:>15} {fn(v4_gains):>10.3f} {fn(v3_gains):>10.3f}")

    # v4 feature stats
    v4_feats = ['zcr', 'hp_ratio', 'kurt', 'sc_high', 'rms_cv']
    print(f"\n  v4 Feature Ranges:")
    print(f"  {'Feature':<12} {'Min':>8} {'Mean':>8} {'Max':>8} {'Std':>8}")
    print("  " + "-" * 45)
    for feat in v4_feats:
        vals = [r[feat] for r in results]
        print(f"  {feat:<12} {np.min(vals):>8.3f} {np.mean(vals):>8.3f} {np.max(vals):>8.3f} {np.std(vals):>8.3f}")

    # =============================================
    # SHIFT ANALYSIS: What moved most?
    # =============================================
    print(f"\n{'='*130}")
    print("v4 vs v3 SHIFT ANALYSIS")
    print(f"{'='*130}")

    shifts = [(r['file'], r['v4_gain'], r['v3_gain'], r['v4_gain'] - r['v3_gain']) for r in results]
    shifts.sort(key=lambda x: x[3])  # sort by delta

    print(f"\n  Songs that moved DOWN (v4 reads lower than v3):")
    for name, v4, v3, delta in shifts:
        if delta < -0.05:
            print(f"    {name[:40]:<42} v3={v3:.2f} → v4={v4:.2f} ({delta:+.2f})")

    print(f"\n  Songs that moved UP (v4 reads higher than v3):")
    for name, v4, v3, delta in shifts:
        if delta > 0.05:
            print(f"    {name[:40]:<42} v3={v3:.2f} → v4={v4:.2f} ({delta:+.2f})")

    print(f"\n  Songs that stayed stable (|Δ| ≤ 0.05):")
    for name, v4, v3, delta in shifts:
        if abs(delta) <= 0.05:
            print(f"    {name[:40]:<42} v3={v3:.2f} → v4={v4:.2f} ({delta:+.2f})")

    # =============================================
    # SAVE CSV
    # =============================================
    csv_cols = ['file', 'v4_gain', 'v3_gain', 'v3_tier',
                'zcr', 'hp_ratio', 'kurt', 'sc_high', 'rms_cv',
                'raw', 'flat', 'hbf', 'her', 'crest',
                'treble', 'mids', 'presence', 'air',
                'ew_mids', 'ew_presence', 'ew_air',
                'harm_ratio', 'harm_rolloff', 'harm_coverage',
                'width', 'duration']

    csv_path = "batch_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in results:
            writer.writerow({c: r.get(c, '') for c in csv_cols})

    print(f"\n💾 Results saved to {csv_path}")

    if errors:
        print(f"\n❌ {len(errors)} files failed to process:")
        for e in errors:
            print(f"  {e['file']}: {e['error']}")

if __name__ == "__main__":
    main()
