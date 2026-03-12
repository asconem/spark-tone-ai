import sys
import os
import librosa
import numpy as np
import json
import math
import datetime
import re
from scipy.signal import butter, lfilter
from scipy.stats import kurtosis as calc_kurtosis

# Load .env file if present (for API keys without needing `export`)
def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    value = value.strip().strip("'").strip('"')
                    os.environ.setdefault(key.strip(), value)
_load_dotenv()

# ==========================================
# PART 1: THE EARS (Tone Analysis)
# ==========================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


# ==========================================
# BPM DETECTION WITH OCTAVE CORRECTION
# ==========================================

def detect_bpm(y, sr):
    """
    Detect BPM with octave-error correction.
    
    librosa's beat_track commonly detects double the actual tempo on slow
    songs (60-100 BPM) because the eighth-note pulse is stronger than the
    quarter-note downbeat in the onset envelope. This produces 140-200 BPM
    readings for songs that are actually 70-100 BPM.
    
    Correction approach: when detected BPM > 120, compare onset strength
    autocorrelation at the detected tempo vs half tempo. If the half-tempo
    autocorrelation is comparable or stronger, the detected BPM is likely
    an octave error.
    
    Returns dict with:
        bpm: corrected BPM
        bpm_raw: original librosa detection
        halved: bool, whether octave correction fired
        ac_full: autocorrelation at detected BPM lag (None if not checked)
        ac_half: autocorrelation at half BPM lag (None if not checked)
    """
    try:
        tempo_result = librosa.beat.beat_track(y=y, sr=sr)
        bpm_raw_val = tempo_result[0]
        bpm_raw = float(bpm_raw_val[0]) if hasattr(bpm_raw_val, '__len__') else float(bpm_raw_val)
        bpm_raw = max(40.0, min(240.0, bpm_raw))
    except Exception as e:
        print(f"   ⚠️ BPM detection failed: {e} (defaulting to 120)")
        return {'bpm': 120.0, 'bpm_raw': 120.0, 'halved': False, 'ac_full': None, 'ac_half': None}
    
    if bpm_raw <= 120:
        print(f"   📊 Tempo: {bpm_raw:.0f} BPM")
        return {'bpm': bpm_raw, 'bpm_raw': bpm_raw, 'halved': False, 'ac_full': None, 'ac_half': None}
    
    # --- Octave error check ---
    # Compute onset strength envelope and its autocorrelation.
    # Compare AC peak at detected tempo lag vs half-tempo lag.
    # If half-tempo is at least as strong, likely an octave error.
    try:
        hop_length = 512  # librosa onset_strength default
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        ac = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
        
        if ac[0] > 0:
            ac = ac / ac[0]  # normalize
        
        def bpm_to_lag(bpm_val):
            return int(60.0 * sr / (bpm_val * hop_length))
        
        half_bpm = bpm_raw / 2.0
        lag_full = bpm_to_lag(bpm_raw)
        lag_half = bpm_to_lag(half_bpm)
        
        # Windowed peak: account for slight tempo drift by checking ±3 frames
        window = 3
        def peak_ac(lag):
            s = max(0, lag - window)
            e = min(len(ac), lag + window + 1)
            return float(np.max(ac[s:e])) if s < len(ac) and s < e else 0.0
        
        ac_full = peak_ac(lag_full)
        ac_half = peak_ac(lag_half)
        
        # Decision: if half-tempo AC is at least 95% as strong as full-tempo AC,
        # the downbeat pulse is as strong as the detected pulse, suggesting the
        # detected tempo is tracking eighth notes, not quarter notes.
        #
        # Threshold TBD — disabled pending batch calibration on real stems.
        # AC ratio is logged for every song > 120 BPM so the correct threshold
        # can be determined from the full dataset.
        # halved = ac_half >= ac_full * THRESHOLD
        halved = False  # DISABLED — collecting data only
        bpm = bpm_raw
        
        ac_ratio = ac_half / max(ac_full, 1e-10)
        print(f"   📊 Tempo: {bpm_raw:.0f} BPM (octave check: AC@{bpm_raw:.0f}={ac_full:.3f}, AC@{half_bpm:.0f}={ac_half:.3f}, ratio={ac_ratio:.2f})")
        
        return {'bpm': bpm, 'bpm_raw': bpm_raw, 'halved': halved, 'ac_full': ac_full, 'ac_half': ac_half}
    
    except Exception as e:
        # Autocorrelation failed — use raw BPM
        print(f"   📊 Tempo: {bpm_raw:.0f} BPM (octave check failed: {e})")
        return {'bpm': bpm_raw, 'bpm_raw': bpm_raw, 'halved': False, 'ac_full': None, 'ac_half': None}


# ==========================================
# BPM API LOOKUP (GetSongBPM.com)
# ==========================================

def lookup_bpm_api(song_name):
    """
    Look up the canonical BPM for a song via the Anthropic API.
    
    Extracts a search query from the filename and asks Claude for the BPM.
    Returns None if no API key, unrecognizable filename, or error.
    
    Args:
        song_name: filename like "hotelcalifornia.wav" or "Hotel California - Eagles.wav"
    
    Returns:
        dict with 'bpm', 'title', 'artist' on success, or None on failure
    """
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None
    
    # Clean filename into search query
    name = os.path.splitext(os.path.basename(song_name))[0]
    name = re.sub(r'[_\-]', ' ', name)
    name = re.sub(r'\b(guitar|vocals|drums|bass|piano|other|stem|wav|flac|mp3)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+', ' ', name).strip()
    
    if not name or len(name) < 3:
        return None
    
    try:
        import anthropic
    except ImportError:
        return None
    
    prompt = (
        f'What is the BPM (tempo) of the song "{name}"? '
        f'Reply with ONLY a JSON object in this exact format, nothing else:\n'
        f'{{"bpm": 120, "title": "Song Title", "artist": "Artist Name"}}\n'
        f'Use the standard studio recording tempo. If you are not confident '
        f'about the exact song, reply with just: {{"error": "unknown"}}'
    )
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        
        # Strip markdown code fences if present
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        
        data = json.loads(text)
        
        if 'error' in data:
            return None
        
        bpm = data.get('bpm')
        if not bpm or not isinstance(bpm, (int, float)):
            return None
        
        return {
            'bpm': float(bpm),
            'title': data.get('title', ''),
            'artist': data.get('artist', ''),
        }
    
    except Exception as e:
        print(f"   ⚠️ BPM API lookup failed: {e}")
        return None


def correct_bpm_with_api(features, song_name):
    """
    Cross-reference librosa's BPM against the GetSongBPM database.
    
    If the API returns a BPM and librosa's detection is approximately double,
    halve the librosa value. This fixes the octave-doubling problem that
    autocorrelation analysis couldn't solve (AC ratio was useless — 0.91-1.01
    for both correct and doubled BPMs).
    
    Modifies features['bpm'] in-place if correction applied.
    
    Returns:
        str: status message for logging ('corrected', 'confirmed', 'no_match', 'no_key')
    """
    librosa_bpm = features.get('bpm', 120.0)
    
    api_result = lookup_bpm_api(song_name)
    if api_result is None:
        return 'no_match'
    
    api_bpm = api_result['bpm']
    title = api_result.get('title', '')
    artist = api_result.get('artist', '')
    
    # Check if librosa is approximately double the API value (within 15%)
    ratio = librosa_bpm / api_bpm if api_bpm > 0 else 0
    
    if 1.7 < ratio < 2.3:
        # Octave error — librosa detected double the actual tempo
        corrected = librosa_bpm / 2.0
        print(f"   🎵 BPM API: {title} by {artist} = {api_bpm:.0f} BPM")
        print(f"   🎵 BPM corrected: {librosa_bpm:.0f} → {corrected:.0f} (librosa was 2× actual)")
        features['bpm'] = corrected
        return 'corrected'
    elif 0.85 < ratio < 1.15:
        # Librosa and API agree — confirmed
        print(f"   🎵 BPM API: {title} by {artist} = {api_bpm:.0f} BPM (confirmed ✓)")
        return 'confirmed'
    elif 0.43 < ratio < 0.57:
        # Librosa detected half the API value — unusual but possible
        # Don't correct — librosa probably found a half-time feel
        print(f"   🎵 BPM API: {title} by {artist} = {api_bpm:.0f} BPM (librosa={librosa_bpm:.0f}, half-time feel?)")
        return 'half_time'
    else:
        # Big mismatch — API might have found wrong song, don't touch
        print(f"   🎵 BPM API: {title} by {artist} = {api_bpm:.0f} BPM (librosa={librosa_bpm:.0f}, mismatch — keeping librosa)")
        return 'mismatch'


# ==========================================
# MULTI-SECTION DETECTION
# ==========================================
def detect_sections(y, sr, min_section_duration=15.0, dynamic_threshold=3.0):
    """
    Detect distinct energy sections in audio.
    
    Uses RMS energy in 2-second windows to identify bimodal energy distribution.
    If the 90th percentile RMS is significantly higher than the 10th percentile,
    the track likely has distinct quiet and loud sections worth splitting.
    
    Args:
        y: audio signal (mono)
        sr: sample rate
        min_section_duration: minimum section length in seconds (shorter = merged)
        dynamic_threshold: ratio of p90/p10 RMS required to trigger splitting
    
    Returns:
        List of section dicts with 'start_sample', 'end_sample', 'type' ('quiet'/'loud'),
        or None if no clear section distinction detected.
    """
    # Compute RMS in 2-second windows with 0.5s hops
    frame_length = int(2.0 * sr)
    hop_length = int(0.5 * sr)
    
    if len(y) < frame_length:
        return None  # Track too short
    
    rms_values = []
    frame_positions = []
    
    for i in range(0, len(y) - frame_length, hop_length):
        rms = np.sqrt(np.mean(y[i:i+frame_length]**2))
        rms_values.append(rms)
        frame_positions.append(i)
    
    if len(rms_values) < 4:
        return None  # Not enough frames
    
    rms_values = np.array(rms_values)
    frame_positions = np.array(frame_positions)
    
    # Check for bimodal energy distribution
    rms_p90 = np.percentile(rms_values, 90)
    rms_p10 = np.percentile(rms_values, 10) + 1e-10
    rms_median = np.median(rms_values)
    
    dynamic_ratio = rms_p90 / rms_p10
    
    if dynamic_ratio < dynamic_threshold:
        # Not enough dynamic contrast for distinct sections
        return None
    
    # Threshold: midpoint between median and p90, biased toward median
    # This ensures we catch the "loud" sections without being too sensitive
    threshold = rms_median + (rms_p90 - rms_median) * 0.4
    
    # Classify each frame as loud or quiet
    is_loud = rms_values > threshold
    
    # Build raw sections by finding contiguous regions
    raw_sections = []
    current_state = is_loud[0]
    section_start_idx = 0
    
    for i in range(1, len(is_loud)):
        if is_loud[i] != current_state:
            raw_sections.append({
                'start_sample': frame_positions[section_start_idx],
                'end_sample': frame_positions[i],
                'type': 'loud' if current_state else 'quiet'
            })
            section_start_idx = i
            current_state = is_loud[i]
    
    # Final section extends to end of audio
    raw_sections.append({
        'start_sample': frame_positions[section_start_idx],
        'end_sample': len(y),
        'type': 'loud' if current_state else 'quiet'
    })
    
    # Merge short sections into adjacent ones
    min_samples = int(min_section_duration * sr)
    merged_sections = []
    
    for section in raw_sections:
        duration_samples = section['end_sample'] - section['start_sample']
        
        if duration_samples < min_samples and merged_sections:
            # Merge into previous section (extend its end)
            merged_sections[-1]['end_sample'] = section['end_sample']
        else:
            merged_sections.append(section)
    
    # Re-merge if we still have short sections after first pass
    final_sections = []
    for section in merged_sections:
        duration_samples = section['end_sample'] - section['start_sample']
        
        if duration_samples < min_samples and final_sections:
            final_sections[-1]['end_sample'] = section['end_sample']
        else:
            final_sections.append(section)
    
    # Need at least 2 distinct sections with different types
    if len(final_sections) < 2:
        return None
    
    types = set(s['type'] for s in final_sections)
    if len(types) < 2:
        return None
    
    return final_sections


def get_section_label(gain):
    """
    Label a section based on its gain level.
    
    Returns human-readable label for recipe output.
    """
    if gain < 0.35:
        return "Clean"
    elif gain < 0.55:
        return "Crunch"
    elif gain < 0.75:
        return "Drive"
    else:
        return "Lead"


def format_timestamp(seconds):
    """Format seconds as M:SS for display."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def analyze_with_sections(file_path):
    """
    Analyze a guitar stem with multi-section detection.
    
    For tracks with distinct energy sections (quiet intro + loud solo, etc.),
    analyzes each section separately and returns multiple feature sets.
    
    Args:
        file_path: path to audio file
    
    Returns:
        dict with:
            'is_multi_section': bool
            'sections': list of feature dicts, each with added keys:
                'section_start': start time in seconds
                'section_end': end time in seconds  
                'section_label': human-readable label (Clean/Crunch/Drive/Lead)
                'section_type': 'quiet' or 'loud'
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return None
    
    # Load audio for section detection BEFORE normalization
    # Normalization compresses dynamics (quiet parts boosted to [-1,1]),
    # which destroys the P90/P10 ratio needed for section detection.
    try:
        y_raw_mono, sr = librosa.load(file_path, mono=True)
    except Exception as e:
        print(f"❌ Audio Load Error: {e}")
        return None
    
    # Detect sections on raw (non-normalized) audio
    sections = detect_sections(y_raw_mono, sr)
    
    if sections is None:
        # Single section — run normal analysis
        features = analyze_tone(file_path)
        if features is None:
            return None
        
        duration = len(y_raw_mono) / sr
        features['section_start'] = 0.0
        features['section_end'] = duration
        features['section_label'] = get_section_label(features['gain'])
        features['section_type'] = 'full'
        
        return {
            'is_multi_section': False,
            'sections': [features]
        }
    
    # Multi-section detected — load stereo for per-section analysis
    try:
        y_stereo, sr = librosa.load(file_path, mono=False)
        y_stereo = librosa.util.normalize(y_stereo)
        
        if y_stereo.ndim == 1:
            y_stereo = np.array([y_stereo, y_stereo])
        
        y_left, y_right = y_stereo[0], y_stereo[1]
    except Exception as e:
        print(f"❌ Audio Load Error: {e}")
        return None
    
    print(f"   ⚡ Multi-section detected: {len(sections)} distinct sections")
    for s in sections:
        start_ts = format_timestamp(s['start_sample'] / sr)
        end_ts = format_timestamp(s['end_sample'] / sr)
        print(f"      {s['type'].upper():<6} {start_ts} — {end_ts}")
    
    # Analyze each section separately
    section_features = []
    
    for i, section in enumerate(sections):
        start_sample = section['start_sample']
        end_sample = section['end_sample']
        section_type = section['type']
        
        start_time = start_sample / sr
        end_time = end_sample / sr
        
        print(f"\n   🔬 Analyzing section {i+1}/{len(sections)}: {section_type.upper()} ({format_timestamp(start_time)} — {format_timestamp(end_time)})")
        
        # Extract audio segment
        y_section_left = y_left[start_sample:end_sample]
        y_section_right = y_right[start_sample:end_sample]
        
        # Analyze this segment
        features = analyze_audio_segment(y_section_left, y_section_right, sr)
        
        if features:
            features['section_start'] = start_time
            features['section_end'] = end_time
            features['section_label'] = get_section_label(features['gain'])
            features['section_type'] = section_type
            section_features.append(features)
    
    if not section_features:
        return None
    
    return {
        'is_multi_section': True,
        'sections': section_features
    }


def analyze_audio_segment(y_left, y_right, sr):
    """
    Run full DSP analysis on an audio segment (already loaded).
    
    This is the core of analyze_tone() but takes pre-loaded audio arrays
    instead of a file path. Used by analyze_with_sections() for per-section analysis.
    
    Args:
        y_left: left channel audio (normalized)
        y_right: right channel audio (normalized)
        sr: sample rate
    
    Returns:
        Feature dict (same structure as analyze_tone) or None on error.
    """
    try:
        mid = (y_left + y_right) / 2
        side = (y_left - y_right) / 2
        spark_width = np.clip((np.sqrt(np.mean(side**2)) + 1e-6) / (np.sqrt(np.mean(mid**2)) + 1e-6), 0, 1.0)
        
        y_raw = mid
        y_filtered = butter_bandpass_filter(y_raw, 450, 6800, sr)
        
        # === UNIFIED GAIN DETECTION (v4) ===
        zcr_frames = librosa.feature.zero_crossing_rate(y=y_filtered)[0]
        zcr = float(np.mean(zcr_frames))
        
        kurt = float(calc_kurtosis(y_filtered))
        
        contrast = librosa.feature.spectral_contrast(y=y_filtered, sr=sr, n_bands=6)
        contrast_mean = np.mean(contrast, axis=1)
        sc_high = float(np.mean(contrast_mean[3:6]))
        
        y_harmonic, y_percussive = librosa.effects.hpss(y_filtered)
        harmonic_energy = np.mean(y_harmonic**2)
        percussive_energy = np.mean(y_percussive**2) + 1e-10
        hp_ratio = float(harmonic_energy / percussive_energy)
        
        amplitude_env = librosa.feature.rms(y=y_filtered, frame_length=2048, hop_length=512)[0]
        rms_cv = float(np.std(amplitude_env) / (np.mean(amplitude_env) + 1e-10))
        
        # v4 gain formula
        spark_gain = np.clip(
            2.1855 * zcr
          - 0.0327 * hp_ratio
          - 0.2666 * kurt
          - 0.1359 * sc_high
          + 1.2657 * rms_cv
          + 2.3766
        , 0.0, 1.0)
        
        # P75 gain (diagnostic)
        rms_weights = amplitude_env / (np.sum(amplitude_env) + 1e-10)
        min_frames = min(len(zcr_frames), len(amplitude_env), contrast.shape[1])
        zcr_trimmed = zcr_frames[:min_frames]
        rms_trimmed = amplitude_env[:min_frames]
        contrast_trimmed = contrast[:, :min_frames]
        rms_w = rms_trimmed / (np.sum(rms_trimmed) + 1e-10)
        
        zcr_sorted_idx = np.argsort(zcr_trimmed)
        zcr_cumweight = np.cumsum(rms_w[zcr_sorted_idx])
        zcr_p75 = float(zcr_trimmed[zcr_sorted_idx[np.searchsorted(zcr_cumweight, 0.75)]])
        
        sc_high_per_frame = np.mean(contrast_trimmed[3:6, :], axis=0)
        sc_sorted_idx = np.argsort(sc_high_per_frame)
        sc_cumweight = np.cumsum(rms_w[sc_sorted_idx])
        sc_high_p75 = float(sc_high_per_frame[sc_sorted_idx[np.searchsorted(sc_cumweight, 0.75)]])
        
        spark_gain_p75 = float(np.clip(
            2.1855 * zcr_p75
          - 0.0327 * hp_ratio
          - 0.2666 * kurt
          - 0.1359 * sc_high_p75
          + 1.2657 * rms_cv
          + 2.3766
        , 0.0, 1.0))
        
        gain_delta = spark_gain_p75 - spark_gain
        p75_indicator = f" | P75={spark_gain_p75:.3f} ({gain_delta:+.3f})" if abs(gain_delta) > 0.02 else ""
        print(f"   📊 v4 Diagnostics: ZCR={zcr:.4f} | HP={hp_ratio:.2f} | Kurt={kurt:.2f} | SC_hi={sc_high:.2f} | RMS_cv={rms_cv:.4f} → Gain={spark_gain:.3f}{p75_indicator}")
        
        # === HARMONIC SERIES ANALYSIS ===
        try:
            f0_harm, voiced_flag_harm, voiced_prob_harm = librosa.pyin(
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
            
            harmonic_coverage = float(np.sum(valid_harm) / max(n_hf, 1))
            harmonic_ratio = 1.0
            harmonic_rolloff = 0.5
            
            if np.sum(valid_harm) >= 20:
                valid_idx = np.where(valid_harm)[0]
                frame_even = []
                frame_odd = []
                frame_lower = []
                frame_upper = []
                frame_weights = []
                
                for idx in valid_idx:
                    f0_val = f0_harm[idx]
                    spectrum = S_harm[:, idx]
                    
                    h_energy = {}
                    for h in range(2, 7):
                        harm_freq = f0_val * h
                        if harm_freq > sr * 0.45:
                            break
                        bin_center = int(round(harm_freq / freq_res))
                        if bin_center < 2 or bin_center >= len(freqs_harm) - 2:
                            continue
                        h_energy[h] = float(np.sum(spectrum[bin_center-2:bin_center+3]**2))
                    
                    if len(h_energy) < 4:
                        continue
                    
                    even_e = sum(h_energy.get(h, 0) for h in [2, 4, 6])
                    odd_e = sum(h_energy.get(h, 0) for h in [3, 5])
                    
                    if odd_e > 1e-10 and even_e > 1e-10:
                        frame_even.append(even_e)
                        frame_odd.append(odd_e)
                        frame_weights.append(float(rms_harm[idx]))
                    
                    lower_e = sum(h_energy.get(h, 0) for h in [2, 3])
                    upper_e = sum(h_energy.get(h, 0) for h in [4, 5, 6])
                    if lower_e > 1e-10:
                        frame_lower.append(lower_e)
                        frame_upper.append(upper_e)
                
                if len(frame_weights) >= 10:
                    w = np.array(frame_weights)
                    w = w / (np.sum(w) + 1e-10)
                    
                    even_total = float(np.sum(np.array(frame_even) * w))
                    odd_total = float(np.sum(np.array(frame_odd) * w))
                    harmonic_ratio = float(np.clip(even_total / (odd_total + 1e-10), 0.1, 10.0))
                    
                    if len(frame_lower) >= 10:
                        w_ro = np.array(frame_weights[:len(frame_lower)])
                        w_ro = w_ro / (np.sum(w_ro) + 1e-10)
                        lower_total = float(np.sum(np.array(frame_lower) * w_ro))
                        upper_total = float(np.sum(np.array(frame_upper) * w_ro))
                        harmonic_rolloff = float(np.clip(upper_total / (lower_total + 1e-10), 0.0, 5.0))
            
            if harmonic_ratio > 1.3:
                harm_char = "soft-clip"
            elif harmonic_ratio < 0.7:
                harm_char = "hard-clip"
            else:
                harm_char = "neutral"
            print(f"   📊 Saturation: ratio={harmonic_ratio:.2f} ({harm_char}) | rolloff={harmonic_rolloff:.2f} | coverage={harmonic_coverage:.0%}")
            
        except Exception as e:
            harmonic_ratio = 1.0
            harmonic_rolloff = 0.5
            harmonic_coverage = 0.0
            print(f"   ⚠️ Harmonic analysis skipped: {e}")
        
        # === MIDS, PRESENCE, AIR (energy-weighted) ===
        S = np.abs(librosa.stft(y_filtered))
        freqs = librosa.fft_frequencies(sr=sr)
        
        frame_energy = np.sum(S, axis=0)
        energy_median = np.median(frame_energy)
        loud_mask = frame_energy > energy_median
        
        if np.sum(loud_mask) < 2:
            loud_mask = np.ones(S.shape[1], dtype=bool)
        
        S_loud = S[:, loud_mask]
        loud_energy = frame_energy[loud_mask]
        loud_weights = loud_energy / (np.sum(loud_energy) + 1e-10)
        
        mids_range     = (freqs >= 500)  & (freqs <= 2000)
        presence_range = (freqs >= 1200) & (freqs <= 2400)
        air_range      = (freqs >= 2400) & (freqs <= 6800)
        
        frame_totals = np.sum(S_loud, axis=0) + 1e-10
        mids_per_frame     = np.sum(S_loud[mids_range, :], axis=0) / frame_totals
        presence_per_frame = np.sum(S_loud[presence_range, :], axis=0) / frame_totals
        air_per_frame      = np.sum(S_loud[air_range, :], axis=0) / frame_totals
        
        spark_mids     = float(np.clip(np.sum(mids_per_frame * loud_weights) * 1.5, 0, 1))
        spark_presence = float(np.sum(presence_per_frame * loud_weights))
        spark_air      = float(np.sum(air_per_frame * loud_weights))
        
        print(f"   📊 Tonal: Mids={spark_mids:.3f} | Presence={spark_presence:.3f} | Air={spark_air:.3f}")
        
        # === BPM DETECTION (B3) ===
        bpm_result = detect_bpm(y_raw, sr)
        bpm = bpm_result['bpm']

        return {
            'gain': spark_gain,
            'mids': spark_mids,
            'presence': spark_presence,
            'air': spark_air,
            'width': spark_width,
            'rms_cv': rms_cv,
            'zcr': zcr,
            'hp_ratio': hp_ratio,
            'kurt': kurt,
            'sc_high': sc_high,
            'gain_p75': spark_gain_p75,
            'harmonic_ratio': harmonic_ratio,
            'harmonic_rolloff': harmonic_rolloff,
            'harmonic_coverage': harmonic_coverage,
            'bpm': bpm,
        }
        
    except Exception as e:
        print(f"   ❌ Segment analysis error: {e}")
        return None


def analyze_tone(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return None

    print(f"   👂 Listening to track (UI-Calibrated Analysis)...")

    try:
        y_stereo, sr = librosa.load(file_path, mono=False)
        y_stereo = librosa.util.normalize(y_stereo)

        if y_stereo.ndim == 1:
            y_stereo = np.array([y_stereo, y_stereo])

        y_left, y_right = y_stereo[0], y_stereo[1]

        mid = (y_left + y_right) / 2
        side = (y_left - y_right) / 2
        spark_width = np.clip((np.sqrt(np.mean(side**2)) + 1e-6) / (np.sqrt(np.mean(mid**2)) + 1e-6), 0, 1.0)

        y_raw = mid
    except Exception as e:
        print(f"❌ Audio Load Error: {e}")
        return None

    y_filtered = butter_bandpass_filter(y_raw, 450, 6800, sr)

    # === UNIFIED GAIN DETECTION (v4) ===
    #
    # v4 replaces the three-tier architecture (v3) with a single unified
    # regression model. No gates, no tiers, no saturation thresholds.
    #
    # WHY: The v3 clean-zone gate (kurt > 0.5 AND hbf < 0.005) caused
    # catastrophic misroutes on unfamiliar songs — palm-muted metal
    # (RATM: -0.39), octave effects (Seven Nation Army: -0.50), and
    # bright clean tones (Gravity: +0.44) all broke the gate assumptions.
    #
    # v4 uses 5 features in a direct linear model trained on 24 songs:
    #   - zcr:     Zero-crossing rate (higher = more high-freq content/distortion)
    #   - hp_ratio: Harmonic-to-percussive energy ratio (lower = more distortion)
    #   - kurt:    Kurtosis of amplitude distribution (lower = more compressed)
    #   - sc_high: Spectral contrast in upper bands (lower = valleys filled by distortion)
    #   - rms_cv:  RMS coefficient of variation (dynamics compression indicator)
    #
    # Performance vs v3 on 24 songs:
    #   v3: MAE=0.150 | MaxErr=0.500 | ±0.10: 14/24 | ±0.15: 16/24 | Catastrophic(>0.20): 7
    #   v4: MAE=0.094 | MaxErr=0.188 | ±0.10: 12/24 | ±0.15: 20/24 | Catastrophic(>0.20): 0
    #
    # Tradeoff: v4 is slightly less precise on songs v3 nailed (classic rock
    # cleans read ~0.10-0.15 high), but eliminates all catastrophic failures.
    # Max error dropped from 0.500 to 0.188. Zero songs over 0.20 error.

    # --- Step 1: Extract features ---
    # Zero-crossing rate
    zcr_frames = librosa.feature.zero_crossing_rate(y=y_filtered)[0]
    zcr = float(np.mean(zcr_frames))

    # Kurtosis
    kurt = float(calc_kurtosis(y_filtered))

    # Spectral contrast — upper bands (bands 3-5 of 7)
    contrast = librosa.feature.spectral_contrast(y=y_filtered, sr=sr, n_bands=6)
    contrast_mean = np.mean(contrast, axis=1)  # mean across time per band
    sc_high = float(np.mean(contrast_mean[3:6]))  # upper frequency bands

    # Harmonic-to-percussive energy ratio
    y_harmonic, y_percussive = librosa.effects.hpss(y_filtered)
    harmonic_energy = np.mean(y_harmonic**2)
    percussive_energy = np.mean(y_percussive**2) + 1e-10
    hp_ratio = float(harmonic_energy / percussive_energy)

    # RMS coefficient of variation (dynamics)
    amplitude_env = librosa.feature.rms(y=y_filtered, frame_length=2048, hop_length=512)[0]
    rms_cv = float(np.std(amplitude_env) / (np.mean(amplitude_env) + 1e-10))

    # --- Step 2: Unified gain formula (uses means — trained on means) ---
    spark_gain = np.clip(
        2.1855 * zcr
      - 0.0327 * hp_ratio
      - 0.2666 * kurt
      - 0.1359 * sc_high
      + 1.2657 * rms_cv
      + 2.3766
    , 0.0, 1.0)

    # --- Step 2b: Percentile gain diagnostics (for future retraining) ---
    # Compute energy-weighted 75th percentile for frame-level gain features.
    # NOT used in the gain formula yet — regression needs retraining on
    # percentile features first. But logged for comparison.
    rms_weights = amplitude_env / (np.sum(amplitude_env) + 1e-10)
    # Align frame counts (zcr and contrast may differ slightly from rms)
    min_frames = min(len(zcr_frames), len(amplitude_env), contrast.shape[1])
    zcr_trimmed = zcr_frames[:min_frames]
    rms_trimmed = amplitude_env[:min_frames]
    contrast_trimmed = contrast[:, :min_frames]
    rms_w = rms_trimmed / (np.sum(rms_trimmed) + 1e-10)

    # Energy-weighted sort for percentile calculation
    zcr_sorted_idx = np.argsort(zcr_trimmed)
    zcr_cumweight = np.cumsum(rms_w[zcr_sorted_idx])
    zcr_p75 = float(zcr_trimmed[zcr_sorted_idx[np.searchsorted(zcr_cumweight, 0.75)]])

    sc_high_per_frame = np.mean(contrast_trimmed[3:6, :], axis=0)
    sc_sorted_idx = np.argsort(sc_high_per_frame)
    sc_cumweight = np.cumsum(rms_w[sc_sorted_idx])
    sc_high_p75 = float(sc_high_per_frame[sc_sorted_idx[np.searchsorted(sc_cumweight, 0.75)]])

    spark_gain_p75 = float(np.clip(
        2.1855 * zcr_p75
      - 0.0327 * hp_ratio
      - 0.2666 * kurt
      - 0.1359 * sc_high_p75
      + 1.2657 * rms_cv
      + 2.3766
    , 0.0, 1.0))

    # --- Step 3: Legacy features for diagnostics and other uses ---
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y_filtered)))
    crest_factor = float(np.max(amplitude_env) / (np.mean(amplitude_env) + 1e-6))
    high_band = butter_bandpass_filter(y_raw, 2000, 6800, sr)
    high_band_flatness = float(np.mean(librosa.feature.spectral_flatness(y=high_band)))
    low_band = butter_bandpass_filter(y_raw, 80, 500, sr)
    mid_band = butter_bandpass_filter(y_raw, 500, 2000, sr)
    energy_low = np.mean(low_band**2)
    energy_mid = np.mean(mid_band**2)
    energy_high = np.mean(high_band**2)
    total_energy = energy_low + energy_mid + energy_high + 1e-10
    high_energy_ratio = energy_high / total_energy

    # --- Debug output ---
    gain_delta = spark_gain_p75 - spark_gain
    p75_indicator = f" | P75={spark_gain_p75:.3f} ({gain_delta:+.3f})" if abs(gain_delta) > 0.02 else ""
    print(f"   📊 v4 Diagnostics: ZCR={zcr:.4f} | HP={hp_ratio:.2f} | Kurt={kurt:.2f} | SC_hi={sc_high:.2f} | RMS_cv={rms_cv:.4f} → Gain={spark_gain:.3f}{p75_indicator}")

    # --- Step 4: Harmonic Series Analysis ---
    #
    # Measures the CHARACTER of distortion by analyzing harmonic content.
    # This is a dimension that gain level alone can't capture — two songs
    # at the same gain can have completely different distortion feel:
    #   - Tube amp distortion: soft clipping, even harmonics (2nd, 4th, 6th)
    #     are prominent. Warm, musical, compresses gracefully.
    #   - Solid-state/transistor: hard clipping, odd harmonics (3rd, 5th)
    #     are prominent. Harsh, buzzy, clips abruptly.
    #
    # Approach: pitch-track the unfiltered signal using pyin, then for
    # each confidently-pitched frame, measure energy at harmonic multiples
    # (2×f0 through 6×f0) in the STFT. Energy-weight toward loud frames.
    #
    # Uses y_raw (unfiltered mid signal) instead of y_filtered because the
    # 450Hz bandpass filter in y_filtered cuts off guitar fundamentals
    # below A4. Open low E is 82Hz — need the full range for pitch tracking.
    #
    # Output features:
    #   harmonic_ratio:    even/odd harmonic energy (>1.3 = soft-clip, <0.7 = hard-clip)
    #   harmonic_rolloff:  upper vs lower harmonic energy
    #                      (high = hard clipping preserves upper harmonics,
    #                       low = soft clipping rolls them off)
    #   harmonic_coverage: fraction of frames with reliable pitch detection
    #                      (diagnostic: low coverage = mostly chords/noise)

    # Pitch tracking on unfiltered signal
    try:
        f0_harm, voiced_flag_harm, voiced_prob_harm = librosa.pyin(
            y_raw, fmin=60, fmax=1400, sr=sr,
            frame_length=2048, hop_length=512
        )

        # STFT with 4096-point FFT for high frequency resolution (~5.4 Hz/bin at 22050 sr).
        # Needed to distinguish closely-spaced harmonics at low fundamentals.
        S_harm = np.abs(librosa.stft(y_raw, n_fft=4096, hop_length=512))
        freqs_harm = librosa.fft_frequencies(sr=sr, n_fft=4096)
        freq_res = freqs_harm[1] - freqs_harm[0]  # Hz per bin

        # Align frame counts (pyin, STFT, and RMS may differ by 1-2 frames)
        n_hf = min(len(f0_harm), S_harm.shape[1], len(amplitude_env))
        f0_harm = f0_harm[:n_hf]
        voiced_flag_harm = voiced_flag_harm[:n_hf]
        S_harm = S_harm[:, :n_hf]
        rms_harm = amplitude_env[:n_hf]

        # Valid frames: voiced + loud enough + finite f0
        # Loud threshold: 50% of median RMS (includes moderate passages, not just peaks)
        rms_med = np.median(rms_harm)
        valid_harm = voiced_flag_harm & (rms_harm > rms_med * 0.5) & np.isfinite(f0_harm)

        harmonic_coverage = float(np.sum(valid_harm) / max(n_hf, 1))

        # Default neutral values (used if insufficient pitched frames)
        harmonic_ratio = 1.0
        harmonic_rolloff = 0.5

        if np.sum(valid_harm) >= 20:
            valid_idx = np.where(valid_harm)[0]
            frame_even = []
            frame_odd = []
            frame_lower = []
            frame_upper = []
            frame_weights = []

            for idx in valid_idx:
                f0_val = f0_harm[idx]
                spectrum = S_harm[:, idx]

                # Measure energy at harmonics 2nd through 6th
                # (1st = fundamental, not useful for even/odd classification)
                h_energy = {}
                for h in range(2, 7):
                    harm_freq = f0_val * h
                    if harm_freq > sr * 0.45:  # stay below Nyquist with margin
                        break
                    bin_center = int(round(harm_freq / freq_res))
                    if bin_center < 2 or bin_center >= len(freqs_harm) - 2:
                        continue
                    # ±2 bins window captures spectral leakage from windowing
                    h_energy[h] = float(np.sum(spectrum[bin_center-2:bin_center+3]**2))

                if len(h_energy) < 4:  # need at least 4 harmonics for reliable ratio
                    continue

                # Even harmonics (2nd, 4th, 6th) — soft-clip character
                even_e = sum(h_energy.get(h, 0) for h in [2, 4, 6])
                # Odd harmonics (3rd, 5th) — hard-clip character
                odd_e = sum(h_energy.get(h, 0) for h in [3, 5])

                if odd_e > 1e-10 and even_e > 1e-10:
                    frame_even.append(even_e)
                    frame_odd.append(odd_e)
                    frame_weights.append(float(rms_harm[idx]))

                # Lower harmonics (2nd, 3rd) vs upper (4th, 5th, 6th) for rolloff
                lower_e = sum(h_energy.get(h, 0) for h in [2, 3])
                upper_e = sum(h_energy.get(h, 0) for h in [4, 5, 6])
                if lower_e > 1e-10:
                    frame_lower.append(lower_e)
                    frame_upper.append(upper_e)

            if len(frame_weights) >= 10:
                w = np.array(frame_weights)
                w = w / (np.sum(w) + 1e-10)

                # Energy-weighted even/odd ratio
                even_total = float(np.sum(np.array(frame_even) * w))
                odd_total = float(np.sum(np.array(frame_odd) * w))
                harmonic_ratio = float(np.clip(even_total / (odd_total + 1e-10), 0.1, 10.0))

                # Harmonic rolloff: upper/lower ratio
                # High = upper harmonics preserved (hard clipping signature)
                # Low = upper harmonics decay quickly (soft clipping / tube)
                if len(frame_lower) >= 10:
                    w_ro = np.array(frame_weights[:len(frame_lower)])
                    w_ro = w_ro / (np.sum(w_ro) + 1e-10)
                    lower_total = float(np.sum(np.array(frame_lower) * w_ro))
                    upper_total = float(np.sum(np.array(frame_upper) * w_ro))
                    harmonic_rolloff = float(np.clip(upper_total / (lower_total + 1e-10), 0.0, 5.0))

        # Diagnostic output
        if harmonic_ratio > 1.3:
            harm_char = "soft-clip"
        elif harmonic_ratio < 0.7:
            harm_char = "hard-clip"
        else:
            harm_char = "neutral"
        print(f"   📊 Saturation: ratio={harmonic_ratio:.2f} ({harm_char}) | rolloff={harmonic_rolloff:.2f} | coverage={harmonic_coverage:.0%}")

    except Exception as e:
        # pyin or harmonic analysis failed — continue with neutral defaults
        harmonic_ratio = 1.0
        harmonic_rolloff = 0.5
        harmonic_coverage = 0.0
        print(f"   ⚠️ Harmonic analysis skipped: {e}")

    # --- Mids, Presence, and Air (energy-weighted loud-section analysis) ---
    #
    # Previous versions computed tonal features as global means across the
    # entire STFT. This averages quiet and loud sections equally — a song
    # like Comfortably Numb (clean verse + screaming solo) gets a mid-range
    # reading that matches neither section.
    #
    # Now uses energy-weighted analysis: each STFT frame's contribution is
    # weighted by its energy (sum of magnitudes). Louder frames — the ones
    # that define the song's core tone — dominate the measurement.
    #
    # Additionally, a "loud mask" filters to frames above the median energy,
    # completely excluding quiet passages, intros, and fadeouts that would
    # otherwise dilute the tonal signature.
    #
    # Measurements:
    #   spark_mids     (500-2000Hz):  Body / midrange fullness
    #   spark_presence (1200-2400Hz): Upper mids / bite / vocal character
    #   spark_air      (2400-6800Hz): Presence / sparkle / Rangemaster territory

    S = np.abs(librosa.stft(y_filtered))
    freqs = librosa.fft_frequencies(sr=sr)

    # Per-frame energy and loud-frame mask
    frame_energy = np.sum(S, axis=0)  # total magnitude per frame
    energy_median = np.median(frame_energy)
    loud_mask = frame_energy > energy_median  # top 50% of frames by energy

    # Bail out if somehow no loud frames (shouldn't happen)
    if np.sum(loud_mask) < 2:
        loud_mask = np.ones(S.shape[1], dtype=bool)

    # Extract loud frames and their energy weights
    S_loud = S[:, loud_mask]
    loud_energy = frame_energy[loud_mask]
    loud_weights = loud_energy / (np.sum(loud_energy) + 1e-10)

    mids_range     = (freqs >= 500)  & (freqs <= 2000)
    presence_range = (freqs >= 1200) & (freqs <= 2400)
    air_range      = (freqs >= 2400) & (freqs <= 6800)

    # Per-frame ratios for loud frames
    frame_totals = np.sum(S_loud, axis=0) + 1e-10  # per-frame total energy
    mids_per_frame     = np.sum(S_loud[mids_range, :], axis=0) / frame_totals
    presence_per_frame = np.sum(S_loud[presence_range, :], axis=0) / frame_totals
    air_per_frame      = np.sum(S_loud[air_range, :], axis=0) / frame_totals

    # Energy-weighted average of loud frames
    spark_mids     = float(np.clip(np.sum(mids_per_frame * loud_weights) * 1.5, 0, 1))
    spark_presence = float(np.sum(presence_per_frame * loud_weights))
    spark_air      = float(np.sum(air_per_frame * loud_weights))

    # Also compute global means for diagnostics (to see the shift)
    total_e = np.sum(S) + 1e-6
    mids_global     = float(np.clip((np.sum(S[mids_range, :]) / total_e) * 1.5, 0, 1))
    presence_global = float(np.sum(S[presence_range, :]) / total_e)
    air_global      = float(np.sum(S[air_range, :]) / total_e)

    # Report shifts when significant
    mids_shift = spark_mids - mids_global
    air_shift = spark_air - air_global
    shift_note = ""
    if abs(mids_shift) > 0.02 or abs(air_shift) > 0.02:
        shift_note = f"  (Δ from global mean: mids {mids_shift:+.3f}, air {air_shift:+.3f})"

    print(f"   📊 Tonal: Mids={spark_mids:.3f} | Presence={spark_presence:.3f} | Air={spark_air:.3f}{shift_note}")
    print(f"   📊 Loud frames: {np.sum(loud_mask)}/{S.shape[1]} ({np.sum(loud_mask)/S.shape[1]*100:.0f}%)")

    # === BPM DETECTION (B3) ===
    bpm_result = detect_bpm(y_raw, sr)
    bpm = bpm_result['bpm']

    return {
        'gain': spark_gain,
        'mids': spark_mids,
        'presence': spark_presence,
        'air': spark_air,
        'width': spark_width,
        'rms_cv': rms_cv,
        'zcr': zcr,
        'hp_ratio': hp_ratio,
        'kurt': kurt,
        'sc_high': sc_high,
        'gain_p75': spark_gain_p75,
        'harmonic_ratio': harmonic_ratio,
        'harmonic_rolloff': harmonic_rolloff,
        'harmonic_coverage': harmonic_coverage,
        'bpm': bpm,
    }

# ==========================================
# PART 2: THE BRAIN (Rig Builder - HYBRID MODE)
# ==========================================
def load_gear_db():
    try:
        with open('spark_gear.json', 'r') as f: return json.load(f)
    except: return None

def load_artist_presets():
    try:
        with open('artist_presets.json', 'r') as f: return json.load(f)
    except: return None

def check_artist_override(song_name, db):
    """
    HYBRID MODE: Returns signature effects and optionally forced amp.
    If no amp is forced, DSP analysis determines the amp based on audio measurements.

    Returns:
        (override_rig, forced_settings, reverb_suggestion, preset_source)
        preset_source: None if hand-curated preset, "auto" if API-generated, or None if no match.
    """
    presets = load_artist_presets()
    if not presets: return None, None, None, None
    song_clean = re.sub(r'[^a-zA-Z0-9]', '', song_name.lower())

    # Stopwords that cause false keyword matches when used as substrings.
    # "thereisalight" contains "the" and "is", matching unrelated presets.
    # These are common English words that appear inside many filenames by
    # coincidence. Legitimate short keywords like "srv" or "yyz" are preserved.
    _keyword_stopwords = {'the', 'is', 'a', 'an', 'in', 'on', 'of', 'to', 'it', 'at', 'by', 'or', 'no', 'do', 'my', 'me', 'so', 'up', 'am', 'if', 'be', 'we', 'us', 'he'}

    for _, data in presets.items():
        for keyword in data['keywords']:
            keyword_clean = re.sub(r'[^a-zA-Z0-9]', '', keyword.lower())
            if keyword_clean in _keyword_stopwords:
                continue
            if keyword_clean in song_clean:
                forced_amp_msg = " (amp forced)" if 'amp' in data['forced_gear'] else " (amp chosen by analysis)"
                print(f"   🌟 ARTIST MATCH: {data['description']}")
                print(f"   📌 Applying signature effects{forced_amp_msg}")

                override_rig, forced = {}, data['forced_gear']
                # HYBRID MODE: Map all gear including amp if present
                mapping = {'drive': 'drive', 'mod_eq': 'modulation', 'comp_wah': 'compressor', 'delay': 'delay', 'gate': 'gate', 'amp': 'amps'}

                for slot, db_cat in mapping.items():
                    if slot in forced:
                        if slot == 'comp_wah':
                            target = next((c for c in db['compressor'] if c['name'] == forced[slot]), None)
                            if not target: target = next((w for w in db['wah'] if w['name'] == forced[slot]), None)
                        elif slot == 'mod_eq':
                            # Mod/EQ slot: search modulation first, then eq category
                            target = next((i for i in db['modulation'] if i['name'] == forced[slot]), None)
                            if not target and 'eq' in db:
                                target = next((i for i in db['eq'] if i['name'] == forced[slot]), None)
                        elif slot == 'amp':
                            # Amp: search across all amp categories
                            target = next((m for c in db['amps'] for m in c['models'] if m['name'] == forced[slot]), None)
                        else:
                            target = next((i for i in db[db_cat] if i['name'] == forced[slot]), None)
                        if target: override_rig[slot] = target

                # Carry amp_settings override if present (prevents DSP from overwriting
                # knob values when a forced amp has a very different base_tone profile)
                if 'amp_settings' in forced:
                    forced = dict(forced)  # ensure mutable copy
                    # amp_settings will be picked up by the settings loop below

                # Return reverb TYPE suggestion (not the reverb object itself)
                reverb_suggestion = forced.get('reverb', None)
                preset_source = data.get('source')  # None = curated, "auto" = API-generated
                return override_rig, forced, reverb_suggestion, preset_source
    return None, None, None, None

def format_settings(slot_name, item, overrides={}, source=""):
    if not item: return ""

    # Add source indicator to slot name
    header = f"[{slot_name}] {item['name'].upper()}"
    if source:
        header += f" {source}"

    output = [header]
    params = item.get('parameters', {})

    # Create a case-insensitive copy of overrides for matching
    ci_overrides = {str(k).upper(): v for k, v in overrides.items()}

    # Identify which keys to display
    active_params = []
    if params:
        active_params = sorted(params.keys())
    elif slot_name == "AMP":
        active_params = ["BASS", "GAIN", "MIDDLE", "TREBLE", "VOLUME"]

    row_buffer = ""
    for i, key in enumerate(active_params):
        # Look for the uppercase version of the key
        lookup_key = key.upper()
        param_def = params.get(key, {}) if params else {}
        param_type = param_def.get('type') if isinstance(param_def, dict) else None

        # Suppress MODE on Digital Delay when BPM_MODE is On.
        # In BPM mode, MODE (range selector: 50ms/200ms/500ms/1s) is irrelevant —
        # timing is fully determined by BPM + D_TIME subdivision. Showing it
        # produces a confusing and meaningless "MODE: 50ms" line in every recipe.
        if lookup_key == 'MODE' and ci_overrides.get('BPM_MODE') == 'On':
            continue

        if lookup_key in ci_overrides:
            val = ci_overrides[lookup_key]
        elif param_type in ('switch', 'selector'):
            # Switch/selector: use first value as default, or "Off" for switches
            values = param_def.get('values', [])
            val = values[-1] if param_type == 'switch' and values else (values[0] if values else '?')
        elif isinstance(param_def, dict) and 'min' in param_def:
            val = float(param_def.get('default', 5.0))
        elif isinstance(param_def, dict) and ('bpm_on' in param_def or 'bpm_off' in param_def):
            # Dual-mode param (behavior changes with BPM switch) — skip if no override
            continue
        elif params and key in params:
            val = float(param_def.get('default', 5.0)) if isinstance(param_def, dict) else 5.0
        else:
            val = 5.0

        if isinstance(val, str):
            entry = f"   • {lookup_key}: {val}"
        else:
            val = float(val)
            entry = f"   • {lookup_key}: {val:.1f}"
        row_buffer += f"{entry:<25}"
        if (i + 1) % 2 == 0:
            output.append(row_buffer); row_buffer = ""

    if row_buffer: output.append(row_buffer)
    output.append("-" * 50)
    return "\n".join(output)

def build_rig(features, song_name="Unknown", skip_research=False, save_presets=True, section_info=None):
    """
    Build a Spark 2 rig from DSP features.
    
    Args:
        features: dict from analyze_tone() or analyze_audio_segment()
        song_name: filename for preset matching and recipe saving
        skip_research: skip API research for unknown songs
        save_presets: save auto-generated presets to disk
        section_info: optional dict with multi-section metadata:
            - is_multi_section: bool
            - section_index: int (0-based)
            - section_count: int (total sections)
            - section_start: float (seconds)
            - section_end: float (seconds)
            - section_label: str (Clean/Crunch/Drive/Lead)
            - section_type: str (quiet/loud/full)
    """
    # Unpack feature dict from analyze_tone()
    target_gain = features['gain']
    target_mids = features['mids']
    target_presence = features['presence']
    target_air = features['air']
    target_width = features['width']
    # Additional features now available for downstream use:
    #   features['rms_cv']              — dynamics (reverb TIME, compressor scaling)
    #   features['zcr']                 — zero-crossing rate
    #   features['hp_ratio']            — harmonic-to-percussive ratio
    #   features['kurt']                — kurtosis
    #   features['sc_high']             — spectral contrast (upper bands)
    #   features['gain_p75']            — energy-weighted 75th percentile gain
    #   features['harmonic_ratio']      — even/odd harmonic balance (saturation character)
    #   features['harmonic_rolloff']    — upper vs lower harmonic decay (clipping type)
    #   features['harmonic_coverage']   — fraction of frames with reliable pitch

    db = load_gear_db()
    if not db: return

    # HYBRID MODE: Check for artist presets (effects only)
    override_rig, override_settings, reverb_suggestion, preset_source = check_artist_override(song_name, db)

    # AUTO-RESEARCH: If no preset matched and API key is available, research the song
    if not override_rig and not skip_research:
        try:
            import api_research
            if os.environ.get('ANTHROPIC_API_KEY'):
                preset, report = api_research.research_song(song_name, db, save=save_presets)
                if preset:
                    if save_presets:
                        # Re-run artist check — the new preset is on disk now
                        override_rig, override_settings, reverb_suggestion, preset_source = check_artist_override(song_name, db)
                    else:
                        # Batch mode: apply the preset directly without saving to disk.
                        # This prevents auto-presets from colliding with subsequent songs
                        # in the same batch run, while still giving this song a full recipe.
                        forced = preset.get('forced_gear', {})
                        override_rig = {}
                        override_settings = forced
                        mapping = {'drive': 'drive', 'mod_eq': 'modulation', 'comp_wah': 'compressor', 'delay': 'delay', 'gate': 'gate', 'amp': 'amps'}
                        for slot, db_cat in mapping.items():
                            if slot in forced and isinstance(forced[slot], str):
                                if slot == 'comp_wah':
                                    target = next((c for c in db['compressor'] if c['name'] == forced[slot]), None)
                                    if not target: target = next((w for w in db['wah'] if w['name'] == forced[slot]), None)
                                elif slot == 'mod_eq':
                                    target = next((i for i in db['modulation'] if i['name'] == forced[slot]), None)
                                    if not target and 'eq' in db:
                                        target = next((i for i in db['eq'] if i['name'] == forced[slot]), None)
                                elif slot == 'amp':
                                    target = next((m for c in db['amps'] for m in c['models'] if m['name'] == forced[slot]), None)
                                else:
                                    target = next((i for i in db[db_cat] if i['name'] == forced[slot]), None)
                                if target: override_rig[slot] = target
                        reverb_suggestion = forced.get('reverb', None)
                        preset_source = "auto"  # API-generated, not hand-curated
                        print(f"   📌 Applying API research (batch mode — not saved to disk)")
        except ImportError:
            pass  # api_research.py not available, continue with DSP-only
        except Exception as e:
            print(f"   ⚠️ API research failed: {e}. Continuing with DSP-only.")

    rig = { "gate": None, "comp_wah": None, "drive": None, "amp": None, "mod_eq": None, "delay": None, "reverb": None }
    settings = { "gate": {}, "comp_wah": {}, "drive": {}, "amp": {}, "mod_eq": {}, "delay": {}, "reverb": {} }
    sources = { "gate": "", "comp_wah": "", "drive": "", "amp": "", "mod_eq": "", "delay": "", "reverb": "" }

    # Apply artist preset EFFECTS and optionally AMP
    if override_rig:
        rig.update(override_rig)
        for k in ['gate', 'drive', 'mod_eq', 'comp_wah', 'delay', 'amp']:
            if rig[k]:
                sources[k] = "🎨"  # Artist signature effect marker
            # Search for settings with various naming patterns
            settings_keys = []
            if k == 'comp_wah':
                settings_keys = [f'{k}_settings', 'comp_settings']
            elif k == 'mod_eq':
                settings_keys = ['mod_settings', 'mod_eq_settings']
            elif k == 'amp':
                settings_keys = ['amp_settings']
            else:
                settings_keys = [f'{k}_settings']

            for key_variant in settings_keys:
                if override_settings and key_variant in override_settings:
                    settings[k] = override_settings[key_variant]

    # ==========================================================
    # GUITAR SELECTION
    # The guitar is the first element in the signal chain.
    # Artist presets specify guitar/pickup when known (🎨).
    # For DSP-only songs, air and gain measurements determine
    # pickup position and tone knob settings (🔬).
    #
    # Matt's guitars:
    #   American Pro II Strat (SSS) — default for everything
    #   Mexican Strat (SSH) — bridge humbucker only, for humbucker songs
    #
    # The American Pro II has a treble bleed circuit, so volume
    # rollback keeps brightness (unlike standard Strats).
    # ==========================================================
    guitar_info = {
        'guitar': 'American Pro II Strat (SSS)',
        'pickup': 'Position 3 (Middle)',
        'volume': 10,
        'tone': 8
    }
    guitar_source = "🔬"

    # Check if artist preset specifies guitar
    if override_settings and 'guitar' in override_settings:
        guitar_info.update(override_settings['guitar'])
        guitar_source = "🎨"
    elif override_rig and not skip_research:
        # Preset exists but no guitar block — try API guitar research
        try:
            import api_research
            if os.environ.get('ANTHROPIC_API_KEY'):
                # Extract artist hint from preset description
                artist_hint = song_name
                try:
                    with open('artist_presets.json', 'r') as f:
                        all_presets = json.load(f)
                    song_clean = re.sub(r'[^a-zA-Z0-9]', '', song_name.lower())
                    for k, v in all_presets.items():
                        for kw in v.get('keywords', []):
                            if re.sub(r'[^a-z0-9]', '', kw.lower()) in song_clean:
                                artist_hint = v.get('description', song_name).split('(')[0].strip()
                                break
                except:
                    pass
                gi = api_research.research_guitar(song_name, artist_hint, db)
                if gi:
                    guitar_info.update(gi)
                    guitar_source = "🤖"  # API-researched guitar
        except (ImportError, Exception):
            pass  # Fall through to DSP-based selection
    else:
        # DSP-based guitar selection
        #
        # Guitar choice: Default to American SSS. Humbucker detection from DSP
        # alone is unreliable — high gain doesn't guarantee humbucker (SRV, Hendrix).
        # Humbuckers are identified by artist presets or API research layer.
        #
        # Pickup position (SSS) — mapped to air measurement:
        #   Air is the best correlate because pickup position primarily affects
        #   high-frequency content. Neck pickup = dark = low air. Bridge = bright = high air.
        #   Thresholds calibrated against known songs:
        #     Gravity (air=0.328) → Pos 5 Neck ✓ (Mayer warm clean)
        #     Lenny (air=0.400) → Pos 4 Neck+Mid ✓ (SRV glassy)
        #     Comfortably Numb (air=0.446) → Pos 3 Middle (DSP fallback;
        #       artist preset overrides to Neck because Gilmour is known)
        if target_air < 0.35:
            guitar_info['pickup'] = 'Position 5 (Neck)'
        elif target_air < 0.42:
            guitar_info['pickup'] = 'Position 4 (Neck+Mid)'
        elif target_air < 0.50:
            guitar_info['pickup'] = 'Position 3 (Middle)'
        elif target_air < 0.56:
            guitar_info['pickup'] = 'Position 2 (Bridge+Mid)'
        else:
            guitar_info['pickup'] = 'Position 1 (Bridge)'

        # Tone knob — maps to air. Low air = roll back for warmth. High air = open.
        if target_air < 0.35:
            guitar_info['tone'] = round(min(10, max(1, 4.0 + (target_air - 0.25) * 20)), 0)
        elif target_air < 0.45:
            guitar_info['tone'] = round(min(10, max(1, 6.0 + (target_air - 0.35) * 20)), 0)
        elif target_air < 0.55:
            guitar_info['tone'] = round(min(10, max(1, 8.0 + (target_air - 0.45) * 10)), 0)
        else:
            guitar_info['tone'] = 10

        # Volume — default 10. Roll back slightly for very clean tones
        # where dynamics benefit from less input signal.
        if target_gain < 0.25:
            guitar_info['volume'] = 9
        else:
            guitar_info['volume'] = 10

    # HYBRID MODE: Use DSP analysis for amp selection UNLESS forced by preset
    if not rig['amp']:
        # Exclude Bass amps — they're for bass guitars, not guitar stems.
        # Without this filter, clean guitar tones with low gain + moderate treble
        # can match bass amps (RB-800, Sunny 3000, W600, Hammer 500).
        all_amps = [m for c in db['amps'] if c['category'] != 'Bass' for m in c['models']]
        # Normalize air to 0-1 scale for fair distance comparison against base_tone treble.
        # Air (2400-6800Hz, range 0.328-0.618) replaces presence (1200-2400Hz, std=0.021)
        # which had no discrimination across songs.
        air_norm = float(np.clip((target_air - 0.328) / (0.618 - 0.328), 0, 1))

        # Harmonic type affinity: when harmonic analysis confidently identifies
        # tube or solid-state character, penalize amps of the wrong type.
        # Conservative: 0.3 penalty only fires on clear mismatches.
        # Won't override a strong gain/treble/mids match, but breaks ties
        # in favor of the right amp type.
        #
        # Guitar solid-state amps in the pool: Silver 120 (JC-120), Checkmate (Teisco)
        # Everything else is tube. Most guitar tones should select tube, so this
        # primarily prevents the JC-120 from being selected for warm tube recordings.
        harm_ratio = features.get('harmonic_ratio', 1.0)

        def amp_distance(a):
            # L1: Non-linear gain model — sigmoid transform spreads the
            # clean-to-crunch zone (0.20-0.50) where amp character matters most.
            #
            # Without this, a song at gain=0.30 barely distinguishes between
            # AC Boost (0.35) and Checkmate (0.40) — linear distance is 0.03 vs 0.07.
            # The sigmoid centered at 0.45 with steepness 8 maps:
            #   0.10 → 0.06   (clean amps spread apart)
            #   0.25 → 0.15   
            #   0.35 → 0.29   (crunch zone has maximum discrimination)
            #   0.45 → 0.50
            #   0.60 → 0.73
            #   0.80 → 0.93   (high-gain amps compressed — gain dominates anyway)
            #   0.95 → 0.98
            #
            # Both song gain and amp base_tone gain go through the same transform
            # so the distance is measured in sigmoid-space.
            def gain_sigmoid(g):
                return 1.0 / (1.0 + math.exp(-8.0 * (g - 0.45)))

            sg_song = gain_sigmoid(target_gain)
            sg_amp = gain_sigmoid(a['base_tone']['gain'])

            d = (((sg_song - sg_amp)**2) * 3.0 +
                 (air_norm - a['base_tone']['treble'])**2 +
                 (target_mids - a['base_tone']['mids'])**2 * 0.5)
            # Type mismatch penalty (only when harmonic analysis is confident)
            amp_type = a.get('type', 'Tube')
            if harm_ratio > 1.3 and amp_type == 'Solid State':
                d += 0.3  # soft-clip harmonics → penalize solid-state amps
            elif harm_ratio < 0.7 and amp_type == 'Tube':
                d += 0.3  # hard-clip harmonics → penalize tube amps
            return d

        rig['amp'] = min(all_amps, key=amp_distance)
        sources['amp'] = "🔬"  # Analysis-based selection marker

    # ==========================================================
    # DSP-DRIVEN UTILITY EFFECTS
    # These are not detected from the audio — they're informed
    # defaults based on what gain, treble, width, and dynamics
    # tell us about the song's character. A knowledgeable
    # guitarist would dial these in for any patch on the Spark 2.
    # Artist presets always override when present.
    # ==========================================================

    # --- REVERB ---
    # Type selection based on gain character:
    #   Ultra-clean (< 0.30): Hall Natural — open, lush, lets clean tones breathe
    #   Clean/crunch (0.30-0.55): Room Studio A — intimate, doesn't mask note clarity
    #   Mid-gain (0.55-0.75): Plate Short — tight, stays out of the way of drive
    #   High-gain (0.75-0.88): Plate Short, very low — just enough to avoid dry/sterile
    #   Thrash (> 0.88): Plate Short, minimal — tightness is everything
    #
    # Settings scaled by analysis:
    #   LEVEL: inversely proportional to gain (clean = wet, heavy = dry)
    #          also scaled by width (wide stereo = more natural space)
    #   TIME:  longer for sustained playing (low rms_cv), shorter for tight/rhythmic
    #   DAMPING: higher for bright tones (tames harsh reflections)
    #   DWELL: higher for clean (more early reflections = warmth), lower for gain

    if reverb_suggestion:
        reverb_obj = next((r for r in db['reverb'] if r['name'] == reverb_suggestion), None)
        if reverb_obj:
            rig['reverb'] = reverb_obj
            if override_settings and 'reverb_settings' in override_settings:
                # Preset has both type AND knob values — use them as-is.
                settings['reverb'] = override_settings['reverb_settings']
                sources['reverb'] = "🎨"
            # else: preset specifies reverb type only, no knob values.
            # rig['reverb'] is now set, so the DSP block below will be skipped.
            # But we still need DSP to fill in the knob settings.
            # Fall through to DSP reverb knob calculation with type already locked.

    if not rig['reverb']:
        # --- REVERB TYPE SELECTION (B4 Enhanced) ---
        # Expanded from 3 types to 7, using gain + genre + width + rms_cv.
        #
        # Available reverbs (9 total, using 7):
        #   Room Studio A/B: small, intimate, short decay — rhythmic/dynamic songs
        #   Chamber:         warm, mid-range space — classic clean tones
        #   Hall Natural:    open, lush — ambient clean / fingerpicking
        #   Hall Medium:     (Holy Grail) — psychedelic, Hendrix, Gilmour
        #   Hall Ambient:    very wide, atmospheric — post-rock, shoegaze
        #   Plate Short:     tight, focused — drive/high-gain, stays out of the way
        #   Plate Rich:      (EMT 140) — studio plate, warm saturation
        #   Plate Long:      (Wampler Reflection) — long lush plate
        #
        # Selection priority: genre hint > gain + width + dynamics
        def find_reverb(name):
            return next((r for r in db['reverb'] if r['name'] == name), None)

        # Try genre-driven selection first
        detected_genre_rev = None
        try:
            from api_research import detect_genre
            detected_genre_rev = detect_genre(song_name)
        except (ImportError, Exception):
            pass

        reverb_selected = False
        rms_cv_rev = features.get('rms_cv', 0.25)

        if detected_genre_rev == 'hendrix':
            rig['reverb'] = find_reverb('Hall Medium')
            reverb_selected = True
        elif detected_genre_rev == 'progressive' and target_gain < 0.60:
            # Progressive cleans: ambient hall for Gilmour-style space
            rig['reverb'] = find_reverb('Hall Ambient')
            reverb_selected = True
        elif detected_genre_rev == 'progressive':
            # Progressive drive/lead: plate rich for studio polish
            rig['reverb'] = find_reverb('Plate Rich')
            reverb_selected = True
        elif detected_genre_rev in ('blues', 'blues_rock'):
            # Blues: warm plate for studio character
            rig['reverb'] = find_reverb('Plate Rich')
            reverb_selected = True

        if not reverb_selected:
            # Gain + width + dynamics driven selection
            if target_gain < 0.25 and target_width > 0.45:
                # Ultra-clean + wide stereo: big ambient hall
                rig['reverb'] = find_reverb('Hall Ambient')
            elif target_gain < 0.30:
                # Clean: natural hall, open and lush
                rig['reverb'] = find_reverb('Hall Natural')
            elif target_gain < 0.45:
                # Clean-crunch: chamber for warm intimacy
                rig['reverb'] = find_reverb('Chamber')
            elif target_gain < 0.55:
                # Crunch: room for tightness, or plate rich if sustained
                if rms_cv_rev < 0.18:
                    rig['reverb'] = find_reverb('Plate Rich')  # sustained playing
                else:
                    rig['reverb'] = find_reverb('Room Studio A')  # rhythmic
            elif target_gain < 0.75:
                # Drive: plate short, stays out of the way
                rig['reverb'] = find_reverb('Plate Short')
            elif target_gain < 0.88:
                # High gain: plate short, very restrained
                rig['reverb'] = find_reverb('Plate Short')
            else:
                # Thrash/extreme: plate short, minimal
                rig['reverb'] = find_reverb('Plate Short')

        # LEVEL: clean tones get more reverb, heavy tones get less
        # Base: 8.0 at gain=0.0, dropping to 1.5 at gain=1.0
        # Width bonus: wide stereo adds up to +2.0
        base_level = 8.0 - (target_gain * 6.5)
        width_bonus = max(0, (target_width - 0.5)) * 4.0
        reverb_level = round(min(10.0, max(1.0, base_level + width_bonus)), 1)

        # TIME: sustained playing gets longer tails, choppy rhythm gets shorter.
        # Two factors: gain (high gain = tighter reverb) and dynamics (rms_cv).
        # rms_cv < 0.15 = very steady (sustained Gilmour leads) → longer time
        # rms_cv > 0.35 = very dynamic (rhythmic/choppy Malcolm Young) → shorter time
        # rms_cv contribution scaled by 2.0 — gives ~0.6 range across typical songs.
        rms_cv = features.get('rms_cv', 0.25)
        reverb_time = round(min(7.0, max(2.0, 6.0 - (target_gain * 2.5) - (rms_cv * 2.0))), 1)

        # DAMPING: bright tones need more damping to avoid harshness
        # Use air (2400-6800Hz, range 0.33-0.62) for damping — brighter tones need
        # more high-frequency damping in reverb. Gives range 3.0-6.5 across library.
        reverb_damping = round(min(8.0, max(3.0, 3.0 + (target_air * 6.0))), 1)

        # DWELL: more early reflections for clean (warmth), less for gain (clarity)
        reverb_dwell = round(min(8.0, max(2.0, 7.0 - (target_gain * 5.0))), 1)

        settings['reverb'] = {
            "LEVEL": reverb_level,
            "TIME": reverb_time,
            "DAMPING": reverb_damping,
            "DWELL": reverb_dwell,
            # LOW_CUT: roll bass out of reverb tail at higher gain levels.
            # High-gain tones produce bass saturation harmonics that muddy
            # the reverb. Clean tones benefit from full-range reverb warmth.
            # Range: 3.0 (clean, gain=0) to 7.0 (thrash, gain=1.0)
            "LOW_CUT": round(min(8.0, max(3.0, 3.0 + (target_gain * 4.0))), 1),
            # HIGH_CUT: tame treble in reverb tail for bright tones.
            # Bright songs (high air) produce harsh sibilant reflections.
            # Dark songs need the reverb tail to carry some sparkle.
            # Range: 3.4 (very bright, air=0.62) to 6.4 (very dark, air=0.33)
            # Inverted: higher air → lower HIGH_CUT (more treble filtering)
            "HIGH_CUT": round(min(8.0, max(3.0, 8.0 - (target_air * 7.0))), 1),
        }
        sources['reverb'] = "🔬"

    # Reverb type-only fallthrough: preset specified reverb type but no knob values.
    # rig['reverb'] is set (locked to preset type) but settings['reverb'] is empty.
    # Run DSP knob calculation using the locked reverb type.
    elif rig['reverb'] and not settings['reverb']:
        rms_cv_rev = features.get('rms_cv', 0.25)
        base_level = 8.0 - (target_gain * 6.5)
        width_bonus = max(0, (target_width - 0.5)) * 4.0
        reverb_level = round(min(10.0, max(1.0, base_level + width_bonus)), 1)
        rms_cv = features.get('rms_cv', 0.25)
        reverb_time = round(min(7.0, max(2.0, 6.0 - (target_gain * 2.5) - (rms_cv * 2.0))), 1)
        reverb_damping = round(min(8.0, max(3.0, 3.0 + (target_air * 6.0))), 1)
        reverb_dwell = round(min(8.0, max(2.0, 7.0 - (target_gain * 5.0))), 1)
        settings['reverb'] = {
            "LEVEL": reverb_level,
            "TIME": reverb_time,
            "DAMPING": reverb_damping,
            "DWELL": reverb_dwell,
            "LOW_CUT": round(min(8.0, max(3.0, 3.0 + (target_gain * 4.0))), 1),
            "HIGH_CUT": round(min(8.0, max(3.0, 8.0 - (target_air * 7.0))), 1),
        }
        sources['reverb'] = "🎨"  # Artist type, DSP knobs

    # --- NOISE GATE ---
    # High-gain patches on real amps hum and feedback between notes.
    # Any experienced guitarist adds a gate above gain ~0.70.
    # Threshold and decay scale with gain level AND dynamics (rms_cv).
    #
    # L3 Enhancement: rms_cv awareness.
    #   Dynamic songs (high rms_cv > 0.30): gentler gating — lower threshold,
    #     longer decay. Aggressive gating on dynamic material chops off
    #     quiet passages and kills sustain on fading notes.
    #   Compressed songs (low rms_cv < 0.15): tighter gating is safe because
    #     the signal stays loud and consistent — gate only catches true silence.
    #
    # rms_cv adjustment range: ±1.0 on threshold, ±0.8 on decay.
    # Normalized to 0-1 from typical range 0.10-0.40.
    if not rig['gate'] and target_gain >= 0.70:
        rig['gate'] = db['gate'][0]  # Noise Gate
        rms_cv_gate = features.get('rms_cv', 0.25)
        dynamics_gate = max(0.0, min(1.0, (rms_cv_gate - 0.10) / 0.30))

        # Threshold: base 2.0-5.0 from gain, then dynamic songs pull it DOWN (gentler)
        gate_threshold_base = 2.0 + (target_gain - 0.70) * 13.3
        gate_threshold = round(min(6.0, max(1.5, gate_threshold_base - dynamics_gate * 1.0)), 1)

        # Decay: base 3.0-0.5 from gain, then dynamic songs push it UP (longer ring)
        gate_decay_base = 4.0 - (target_gain - 0.70) * 11.7
        gate_decay = round(min(5.0, max(0.5, gate_decay_base + dynamics_gate * 0.8)), 1)

        settings['gate'] = {"THRESHOLD": gate_threshold, "DECAY": gate_decay}
        sources['gate'] = "🔬"

    # --- COMPRESSOR ---
    # Compressors serve different roles at different gain levels:
    #   Clean (< 0.35): sustain + even dynamics, especially for fingerpicking/arpeggios
    #   Clean-crunch (0.35-0.55) with high dynamics: tame peaks, add consistency
    #   Above 0.55: amp distortion already compresses naturally, skip the comp
    #
    # Compressor type selection:
    #   Very clean, warm tone: LA Comp (LA-2A) — smooth optical compression
    #   Clean, bright/snappy: Red Comp (Dyna Comp) — percussive, adds pop
    #   Clean-crunch, general: Sustain Comp (CS-3) — versatile, transparent-ish
    #
    # rms_cv drives compression intensity within each tier:
    #   High rms_cv (> 0.30) = very dynamic playing (SRV dig-in) → more compression
    #   Low rms_cv (< 0.15) = consistent attack (fingerpicking) → lighter touch
    #   Typical range: 0.14-0.37 across the song library
    rms_cv = features.get('rms_cv', 0.25)
    # Normalize rms_cv to 0-1 scale for compression scaling (0.10-0.40 range)
    dynamics_norm = max(0.0, min(1.0, (rms_cv - 0.10) / 0.30))

    if not rig['comp_wah'] and target_gain < 0.55:
        if target_gain < 0.25:
            # Ultra-clean: LA Comp for smooth, warm sustain
            rig['comp_wah'] = db['compressor'][0]  # LA Comp
            comp_intensity = 0.4 + dynamics_norm * 0.4  # range 0.4-0.8
            settings['comp_wah'] = {
                "GAIN": round(5.0 + comp_intensity * 3.0, 1),
                "PEAK_REDUCTION": round(3.0 + comp_intensity * 4.0, 1)
            }
        elif target_gain < 0.35:
            if target_air > 0.45:
                # Clean + bright: Red Comp for snap and pop
                rig['comp_wah'] = db['compressor'][2]  # Red Comp
                settings['comp_wah'] = {
                    "OUTPUT": round(5.0 + dynamics_norm * 2.0, 1),
                    "SENSITIVITY": round(4.0 + dynamics_norm * 3.0, 1)
                }
            else:
                # Clean + warm: Sustain Comp, gentle settings
                rig['comp_wah'] = db['compressor'][1]  # Sustain Comp
                settings['comp_wah'] = {
                    "LEVEL": round(6.0 + dynamics_norm * 2.0, 1),
                    "TONE": 4.5,
                    "ATTACK": round(5.0 - dynamics_norm * 2.0, 1),
                    "SUSTAIN": round(4.0 + dynamics_norm * 4.0, 1)
                }
        else:
            # Clean-crunch (0.35-0.55): only add comp if dynamics are high
            # and gain is still more clean than crunch
            if target_gain < 0.45 and dynamics_norm > 0.4:
                rig['comp_wah'] = db['compressor'][1]  # Sustain Comp
                settings['comp_wah'] = {
                    "LEVEL": round(5.5 + dynamics_norm * 2.0, 1),
                    "TONE": 5.0,
                    "ATTACK": round(5.0 - dynamics_norm * 1.5, 1),
                    "SUSTAIN": round(3.5 + dynamics_norm * 3.0, 1)
                }

        if rig['comp_wah']:
            sources['comp_wah'] = "🔬"

    # --- DSP-BASED DRIVE SELECTION ---
    # When no preset or API research provides a drive pedal, DSP analysis
    # determines whether one is needed and which type fits the tone.
    #
    # Below gain ~0.45, the amp handles distortion alone (clean/light crunch).
    # Above 0.45, a drive pedal improves accuracy — real guitarists almost
    # always use a pedal to push the amp in these gain ranges.
    #
    # Selection logic uses gain level for the broad category and mids
    # measurement to disambiguate character within each tier:
    #   - High mids → mid-hump drives (Tube Screamer family)
    #   - Scooped mids → scooped drives (Muff, RAT)
    #   - Neutral mids → transparent drives (Klon, OD-3)
    #
    # Only standard drives [0-7] are candidates. Bass Muff, J.H. signature
    # fuzzes, and Bassmaster are reserved for presets/API — they're too
    # specific to select from DSP alone.
    #
    # The drive's sonic_profile automatically feeds into:
    #   - Compensatory EQ (bass_comp, treble_comp, future mid_comp)
    #   - Amp GAIN calculation (subtracts drive gain_boost)
    #   - Amp GAIN floor (2.0 minimum when drive present)
    if not rig['drive'] and target_gain >= 0.45:
        # Helper to find drive by name
        def find_drive(name):
            return next((d for d in db['drive'] if d['name'] == name), None)

        if target_gain < 0.60:
            # Light boost zone — push a crunch amp into saturation.
            # The classic "clean boost into cranked Marshall" approach.
            if target_mids >= 0.50:
                # Mid-forward tone: Tube Drive (TS mid hump pushes the amp)
                rig['drive'] = find_drive('Tube Drive')
            else:
                # Neutral/bright: Clone Drive (Klon, transparent with treble lift)
                rig['drive'] = find_drive('Clone Drive')

        elif target_gain < 0.75:
            # Overdrive zone — drive is doing real tonal work, not just boosting.
            if target_mids >= 0.55:
                # Strong mid-forward: Tube Drive (SRV, blues rock)
                rig['drive'] = find_drive('Tube Drive')
            elif target_mids >= 0.45:
                # Neutral mids: Over Drive (OD-3, flat response, versatile)
                rig['drive'] = find_drive('Over Drive')
            else:
                # Scooped mids: SAB Driver (Plexi-Drive, Marshall-voiced distortion)
                rig['drive'] = find_drive('SAB Driver')

        elif target_gain < 0.85:
            # Heavy overdrive/distortion zone.
            if target_mids >= 0.50:
                # Present mids: SAB Driver (amp-in-a-box, thick Marshall push)
                rig['drive'] = find_drive('SAB Driver')
            else:
                # Scooped/tight: Black Op (RAT, hard clipping, cuts through)
                rig['drive'] = find_drive('Black Op')

        else:
            # Fuzz/heavy saturation zone (gain >= 0.85).
            # Harmonic ratio disambiguates two very different sounds at the same
            # gain+mids: Corgan's Muff wall (ratio=2.72, warm tube fuzz) vs
            # Dimebag's tight chug (ratio=0.89, hard-clipping metal).
            # Both read as high gain + scooped mids, but the harmonic character
            # is completely different.
            harm_ratio = features.get('harmonic_ratio', 1.5)

            if target_mids < 0.40 and harm_ratio > 1.3:
                # Deeply scooped + warm harmonics: Guitar Muff (wall of fuzz)
                rig['drive'] = find_drive('Guitar Muff')
            elif target_mids < 0.40:
                # Deeply scooped + hard-clipping harmonics: Black Op (tight metal)
                rig['drive'] = find_drive('Black Op')
            elif target_air < 0.42:
                # Dark/warm: Fuzz Face (wooly, vintage fuzz character)
                rig['drive'] = find_drive('Fuzz Face')
            else:
                # Bright/present high gain: Black Op (tight, cuts, doesn't get muddy)
                rig['drive'] = find_drive('Black Op')

        if rig['drive']:
            sources['drive'] = "🔬"

    # --- DRIVE KNOB SETTINGS (DSP-informed) ---
    # When a drive pedal is in the rig but has no settings (DSP-selected drives,
    # or rare presets missing drive_settings), generate intelligent knob values.
    #
    # Core concept: "drive_intensity" measures how much work the drive needs to do.
    # It's the gap between target_gain and the amp's inherent gain, normalized by
    # the drive's gain_boost capability. 0.0 = amp handles it, 1.0 = drive is maxed.
    #
    # Three knob types, each pedal uses different names:
    #   Amount (overdrive/drive/distortion/fuzz/sustain/gain): How hard the drive clips
    #   Tone (tone/treble/filter): Brightness control, mapped from air measurement
    #   Volume (level/output/volume): Output level, scaled to prevent amp clipping
    #
    # Pedals are classified by character:
    #   Boost/OD (gb <= 0.6): Push the amp harder. Low drive amount, high output level.
    #     The classic Tube Screamer approach — it's a level boost, not a distortion source.
    #   Heavy distortion (gb 0.6-0.8): Drive IS the distortion. Moderate amount and volume.
    #   Fuzz (gb > 0.8): Drive defines the entire tone. Amount scales with song heaviness.
    #     Volume backed off to prevent amp clipping (Cherub Rock: Muff 7.5 → 5.5 lesson).
    #
    # Calibrated against:
    #   Cherub Rock — Muff: Sustain 6.5, Tone 7.8, Volume 5.5
    #   Eruption — Tube Drive: Overdrive 1.7, Tone 6.5, Level 8.0
    if rig['drive'] and not settings['drive']:
        sp = rig['drive'].get('sonic_profile', {})
        gb = sp.get('gain_boost', 0.5)
        drive_name = rig['drive']['name']

        # How hard does the drive need to work?
        amp_base_gain = rig['amp']['base_tone']['gain'] if rig['amp'] else 0.5
        drive_intensity = max(0.0, min(1.0, (target_gain - amp_base_gain) / (gb + 0.01)))

        # Air normalized to 0-1 for tone mapping
        air_norm = max(0.0, min(1.0, (target_air - 0.328) / (0.618 - 0.328)))

        # --- Per-pedal settings ---
        # Each drive has unique parameter names and scaling behavior.

        if drive_name == 'Booster':
            # Only one knob — it's a pure level boost
            settings['drive'] = {
                "GAIN": round(min(10.0, max(1.0, 5.0 + drive_intensity * 4.0)), 1)
            }

        elif drive_name == 'Clone Drive':
            # Klon Centaur: transparent boost. Low gain, treble from air, output pushes amp.
            settings['drive'] = {
                "GAIN": round(min(10.0, max(1.0, 1.0 + drive_intensity * 3.0)), 1),
                "TREBLE": round(min(10.0, max(1.0, 3.0 + air_norm * 5.0)), 1),
                "OUTPUT": round(min(10.0, max(3.0, 7.0 + drive_intensity * 1.5)), 1)
            }

        elif drive_name == 'Tube Drive':
            # Tube Screamer: almost always a boost (low OD, high level).
            # Eruption validation: OD=1.7, Tone=6.5, Level=8.0
            #   drive_intensity=0.50 → OD=1.0+0.50*1.5=1.75 ≈ 1.7 ✓
            #   air_norm=0.69 → Tone=3.0+0.69*5.0=6.47 ≈ 6.5 ✓
            #   Level=7.0+0.50*2.0=8.0 ✓
            settings['drive'] = {
                "OVERDRIVE": round(min(10.0, max(1.0, 1.0 + drive_intensity * 1.5)), 1),
                "TONE": round(min(10.0, max(1.0, 3.0 + air_norm * 5.0)), 1),
                "LEVEL": round(min(10.0, max(3.0, 7.0 + drive_intensity * 2.0)), 1)
            }

        elif drive_name == 'Over Drive':
            # Boss OD-3: flat response, slightly more drive than TS in OD mode
            settings['drive'] = {
                "DRIVE": round(min(10.0, max(1.0, 1.5 + drive_intensity * 3.0)), 1),
                "TONE": round(min(10.0, max(1.0, 3.0 + air_norm * 5.0)), 1),
                "LEVEL": round(min(10.0, max(3.0, 7.0 + drive_intensity * 1.5)), 1)
            }

        elif drive_name == 'SAB Driver':
            # Plexi-Drive: amp-in-a-box, provides significant coloring
            settings['drive'] = {
                "DRIVE": round(min(10.0, max(1.0, 3.0 + drive_intensity * 5.0)), 1),
                "TONE": round(min(10.0, max(1.0, 3.0 + air_norm * 5.0)), 1),
                "VOLUME": round(min(10.0, max(3.0, 6.0 - drive_intensity * 1.5)), 1)
            }

        elif drive_name == 'Black Op':
            # ProCo RAT: hard clipping distortion.
            # FILTER is inverted — clockwise CUTS treble.
            # High air (bright song) → low filter (let treble through).
            # Low air (dark song) → high filter (cut treble).
            settings['drive'] = {
                "DISTORTION": round(min(10.0, max(1.0, 3.0 + drive_intensity * 5.0)), 1),
                "FILTER": round(min(10.0, max(1.0, 8.0 - air_norm * 5.0)), 1),
                "VOLUME": round(min(10.0, max(3.0, 6.0 - drive_intensity * 1.5)), 1)
            }

        elif drive_name == 'Fuzz Face':
            # Dunlop Fuzz Face: no tone control, just fuzz and volume.
            # Amount scales with overall song heaviness.
            # Volume backed off to prevent amp input clipping.
            settings['drive'] = {
                "FUZZ": round(min(10.0, max(3.0, target_gain * 7.0)), 1),
                "VOLUME": round(min(10.0, max(3.0, 7.0 - drive_intensity * 4.0)), 1)
            }

        elif drive_name == 'Guitar Muff':
            # Big Muff: fuzz IS the tone. Amount scales with song heaviness.
            # Cherub Rock validation: Sustain=6.5, Tone=7.8, Volume=5.5
            #   target_gain=0.94 → Sustain=0.94*7.0=6.58 ≈ 6.5 ✓
            #   air_norm=1.0 → Tone=3.0+1.0*5.0=8.0 ≈ 7.8 (within 0.2) ✓
            #   drive_intensity=0.378 → Volume=7.0-0.378*4.0=5.49 ≈ 5.5 ✓
            settings['drive'] = {
                "SUSTAIN": round(min(10.0, max(3.0, target_gain * 7.0)), 1),
                "TONE": round(min(10.0, max(1.0, 3.0 + air_norm * 5.0)), 1),
                "VOLUME": round(min(10.0, max(3.0, 7.0 - drive_intensity * 4.0)), 1)
            }

        if settings['drive'] and sources['drive'] != "🎨":
            sources['drive'] = "🔬"

    # --- DSP-INFORMED EFFECT SETTINGS ---
    # When a preset forces an effect but provides no settings (common with
    # auto-generated API presets), generate intelligent defaults from DSP
    # analysis. Better than generic 5.0/5.0/5.0 on every knob.
    #
    # Scaling principles:
    #   - Effect LEVEL/MIX: inversely proportional to gain (clean = more wet)
    #   - RATE/SPEED: moderate defaults, scaled by rms_cv where appropriate
    #   - TONE: mapped from air measurement (bright song = bright effect)
    #   - DEPTH/INTENSITY: moderate defaults, higher for cleaner tones
    rms_cv_fx = features.get('rms_cv', 0.25)
    air_norm_fx = max(0.0, min(1.0, (target_air - 0.328) / (0.618 - 0.328)))
    # Effect intensity: cleaner tones take more modulation/delay effect
    fx_intensity = max(0.3, min(0.8, 0.8 - target_gain * 0.5))

    # --- Modulation settings (when preset forces effect, no settings provided) ---
    if rig['mod_eq'] and not settings['mod_eq']:
        mod_name = rig['mod_eq'].get('name', '')

        if mod_name == 'Chorus':
            settings['mod_eq'] = {
                "E_LEVEL": round(4.0 + fx_intensity * 3.0, 1),
                "RATE": round(2.0 + rms_cv_fx * 4.0, 1),
                "DEPTH": round(3.0 + fx_intensity * 3.0, 1),
                "TONE": round(3.0 + air_norm_fx * 4.0, 1),
            }
        elif mod_name == 'Cloner Chorus':
            settings['mod_eq'] = {
                "RATE": round(2.5 + rms_cv_fx * 3.0, 1),
            }
        elif mod_name == 'Flanger':
            settings['mod_eq'] = {
                "RATE": round(1.0 + rms_cv_fx * 2.0, 1),
                "MIX": round(3.0 + fx_intensity * 3.0, 1),
                "DEPTH": round(4.0 + fx_intensity * 3.0, 1),
            }
        elif mod_name == 'Phaser':
            settings['mod_eq'] = {
                "SPEED": round(3.0 + rms_cv_fx * 3.0, 1),
                "INTENSITY": round(4.0 + fx_intensity * 3.0, 1),
            }
        elif mod_name in ('Tremolo', 'Tremolator', 'Tremolo Square'):
            settings['mod_eq'] = {
                "SPEED": round(4.0 + rms_cv_fx * 3.0, 1),
                "DEPTH": round(4.0 + fx_intensity * 3.0, 1),
                "LEVEL": 7.0,
            }
        elif mod_name in ('UniVibe', 'Classic Vibe'):
            settings['mod_eq'] = {
                "SPEED": round(3.0 + rms_cv_fx * 3.0, 1),
                "INTENSITY": round(4.0 + fx_intensity * 3.0, 1),
            }
        elif mod_name == 'J.H. Legendary Vibe':
            settings['mod_eq'] = {
                "SPEED": round(3.0 + rms_cv_fx * 3.0, 1),
                "SWEEP": round(5.0, 1),
                "INTENSITY": round(4.0 + fx_intensity * 3.0, 1),
                "MIX": round(4.0 + fx_intensity * 2.0, 1),
            }
        elif mod_name == 'Vibrato':
            settings['mod_eq'] = {
                "SPEED": round(3.0 + rms_cv_fx * 2.0, 1),
                "DEPTH": round(3.0 + fx_intensity * 2.0, 1),
            }

        if settings['mod_eq'] and mod_name not in ('Guitar EQ', 'Bass EQ'):
            sources['mod_eq'] = "🔬"

    # --- DSP-DRIVEN DELAY TYPE SELECTION (B5) ---
    # When no preset provides a delay, DSP analysis + genre hints determine
    # whether to add one and which type fits the song's character.
    #
    # Decision: should a delay be present at all?
    #   - Metal/thrash/djent/punk (gain >= 0.85): usually no delay.
    #     Tight riffing muddies with repeats. Presets override this.
    #   - High rms_cv + high gain: very dynamic/choppy — delay clutters gaps.
    #   - Everything else: a subtle delay improves depth and space.
    #
    # Type mapping (genre → delay character):
    #   Blues/blues_rock:  Echo Tape (Echoplex) — SRV, Clapton, warm tape slapback
    #   Progressive:       Multi Head (Space Echo) — Gilmour, atmospheric repeats
    #   Hendrix:           Multi Head (Space Echo) — psychedelic wash
    #   Classic rock:      Vintage Delay (DM-3) — warm analog-style repeats
    #   Hard rock:         Echo Tape (Echoplex) — classic rock delay tone
    #   Alternative:       Digital Delay (DD-3) — clean, precise repeats
    #   Grunge:            Digital Delay (DD-3) — utilitarian, stays out of the way
    #   No genre / clean:  Vintage Delay (DM-3) — safest musical default
    #
    # The existing per-type settings generator (below) handles knob values
    # once a type is selected here.
    if not rig['delay']:
        # Try to detect genre for informed type selection
        detected_genre = None
        try:
            from api_research import detect_genre
            detected_genre = detect_genre(song_name)
        except (ImportError, Exception):
            pass  # No api_research.py or detect_genre failed — use gain-only

        # Should we add a delay at all?
        rms_cv_delay = features.get('rms_cv', 0.25)
        skip_delay = False

        # Metal/thrash/djent/punk at very high gain: skip delay
        if target_gain >= 0.85 and detected_genre in ('metal', 'thrash', 'djent', 'punk'):
            skip_delay = True
        # Very dynamic + high gain: choppy riffing, delay clutters
        elif target_gain >= 0.75 and rms_cv_delay > 0.35:
            skip_delay = True

        if not skip_delay:
            # Select delay type based on genre, then gain as fallback
            def find_delay(name):
                return next((d for d in db['delay'] if d['name'] == name), None)

            if detected_genre in ('blues', 'blues_rock'):
                rig['delay'] = find_delay('Echo Tape')
            elif detected_genre in ('progressive', 'hendrix'):
                rig['delay'] = find_delay('Multi Head')
            elif detected_genre in ('classic_rock',):
                rig['delay'] = find_delay('Vintage Delay')
            elif detected_genre in ('hard_rock',):
                rig['delay'] = find_delay('Echo Tape')
            elif detected_genre in ('alternative', 'grunge'):
                rig['delay'] = find_delay('Digital Delay')
            else:
                # No genre detected — select by gain character
                if target_gain < 0.35:
                    # Clean: warm analog delay suits ambient cleans
                    rig['delay'] = find_delay('Vintage Delay')
                elif target_gain < 0.55:
                    # Crunch: tape delay for warmth without clutter
                    rig['delay'] = find_delay('Echo Tape')
                elif target_gain < 0.75:
                    # Drive: vintage delay, moderate
                    rig['delay'] = find_delay('Vintage Delay')
                else:
                    # High gain without genre: digital delay, tight and clean
                    rig['delay'] = find_delay('Digital Delay')

            if rig['delay']:
                sources['delay'] = "🔬"

    # --- Delay settings (when delay is in rig but has no settings) ---
    # B3: Delay time now derived from BPM when available.
    # Musical subdivisions:
    #   Quarter note  = 60000 / BPM ms (spacious, ambient)
    #   Dotted eighth = 45000 / BPM ms (rhythmic, The Edge / Knopfler)
    #   Eighth note   = 30000 / BPM ms (fast, busy)
    #
    # Subdivision choice:
    #   Clean/ambient (gain < 0.35): quarter note — lets notes breathe
    #   Most songs (0.35-0.75): dotted eighth — most musical default
    #   High gain (>= 0.75): eighth note — tighter, stays out of the way
    #
    # Time parameter mapping per delay type:
    #   Echo Filt DELAY:       100-1000ms → knob 0.0-10.0
    #   Multi Head REPEAT_RATE: 100-1000ms → knob 0.0-10.0
    #   Reverse Delay TIME:    100-2000ms → knob 0.0-10.0
    #   Vintage Delay REPEAT_RATE: musical subdivision → knob 0.0-10.0
    #   Echo Tape SHORT/LONG:  musical subdivision → knob 0.0-10.0
    #   Digital Delay: MODE sets range, D_TIME is fine adjust (keep simple)
    if rig['delay'] and not settings['delay']:
        delay_name = rig['delay'].get('name', '')
        # Delay level: less for high gain (stay out of the way)
        delay_level = max(1.0, min(6.0, 6.0 - target_gain * 4.0))

        # BPM-derived delay times
        bpm = features.get('bpm', 120.0)
        quarter_ms = 60000.0 / bpm
        dotted_eighth_ms = 45000.0 / bpm
        eighth_ms = 30000.0 / bpm

        # Choose subdivision based on gain
        if target_gain < 0.35:
            delay_ms = quarter_ms
            subdivision = "quarter"
        elif target_gain < 0.75:
            delay_ms = dotted_eighth_ms
            subdivision = "dotted eighth"
        else:
            delay_ms = eighth_ms
            subdivision = "eighth"

        if delay_name == 'Digital Delay':
            # When BPM_MODE is On, D_TIME becomes a subdivision selector:
            #   1/32, 1/16, 1/16t, 1/16d, 1/8, 1/8t, 1/8d,
            #   1/4, 1/4t, 1/4d, 1/2, 1/2t, 1/2d, 1
            # MODE is ignored in BPM mode — timing comes from BPM + D_TIME.
            # Full range up to whole note, so no clamping needed.
            sub_map = {'eighth': '1/8', 'dotted eighth': '1/8d', 'quarter': '1/4'}
            settings['delay'] = {
                "E_LEVEL": round(delay_level, 1),
                "F_BACK": round(3.0 + fx_intensity * 2.0, 1),
                "D_TIME": sub_map.get(subdivision, '1/8d'),
                "BPM_MODE": "On",
            }
        elif delay_name == 'Echo Filt':
            # When BPM_MODE is On, DELAY becomes a subdivision selector:
            #   1/32 through 1 (whole note) — same 14 values as Digital Delay.
            # Full range, no clamping needed.
            sub_map = {'eighth': '1/8', 'dotted eighth': '1/8d', 'quarter': '1/4'}
            settings['delay'] = {
                "DELAY": sub_map.get(subdivision, '1/8d'),
                "FEEDBACK": round(3.0 + fx_intensity * 2.0, 1),
                "LEVEL": round(delay_level, 1),
                "TONE": round(3.0 + air_norm_fx * 4.0, 1),
                "BPM_MODE": "On",
            }
        elif delay_name == 'Vintage Delay':
            # When BPM_MODE is On, REPEAT_RATE becomes a subdivision selector:
            #   1/32 through 1 (whole note) — same 14 values as Digital Delay.
            sub_map = {'eighth': '1/8', 'dotted eighth': '1/8d', 'quarter': '1/4'}
            settings['delay'] = {
                "REPEAT_RATE": sub_map.get(subdivision, '1/8d'),
                "ECHO": round(delay_level, 1),
                "INTENSITY": round(3.0 + fx_intensity * 2.0, 1),
                "BPM_MODE": "On",
            }
        elif delay_name == 'Reverse Delay':
            # When BPM_MODE is On, TIME is locked to 1/4 note — no subdivision choice.
            # The reverse effect captures a quarter-note chunk and plays it backwards.
            settings['delay'] = {
                "MIX": round(delay_level * 0.7, 1),
                "DECAY": round(4.0 + fx_intensity * 2.0, 1),
                "FILTER": round(3.0 + air_norm_fx * 4.0, 1),
                "TIME": "1/4",
                "BPM_MODE": "On",
            }
        elif delay_name == 'Multi Head':
            # When BPM_MODE is On, REPEAT_RATE becomes a subdivision selector:
            #   1/32 through 1 (whole note) — same 14 values as Digital/Vintage.
            # MODE_SELECTOR: 4 head combinations modeled on the RE-201.
            #   1: Head 0+2 — widely spaced rhythmic repeats
            #   2: Head 0+1 — single tight repeat (classic slapback)
            #   3: Head 1+2 — medium-spaced double tap
            #   4: Head 0+1+2 — all heads, full Space Echo wash
            # Default to Head 0+1 (mode 2) for most musical default.
            # Progressive/ambient songs benefit from Head 0+1+2 (all heads).
            sub_map = {'eighth': '1/8', 'dotted eighth': '1/8d', 'quarter': '1/4'}
            settings['delay'] = {
                "REPEAT_RATE": sub_map.get(subdivision, '1/8d'),
                "INTENSITY": round(3.0 + fx_intensity * 2.0, 1),
                "ECHO_VOL": round(delay_level, 1),
                "MODE_SELECTOR": "Head 0+1",
                "BPM_MODE": "On",
            }
        elif delay_name == 'Echo Tape':
            # SHORT/LONG dial: when BPM_MODE is On, becomes a subdivision selector.
            # Echo Tape has a shorter range than the others — maxes out at 1/8d.
            # Quarter note is beyond its range, so clamp to 1/8d.
            sub_map = {'eighth': '1/8', 'dotted eighth': '1/8d', 'quarter': '1/8d'}
            settings['delay'] = {
                "SUSTAIN": round(2.0 + fx_intensity * 3.0, 1),
                "VOLUME": round(delay_level * 0.8, 1),
                "TONE": round(3.0 + air_norm_fx * 4.0, 1),
                "SHORT/LONG": sub_map.get(subdivision, '1/8d'),
                "BPM_MODE": "On",
            }

        if settings['delay']:
            sources['delay'] = "🔬"
            print(f"   🎵 Delay time: {delay_ms:.0f}ms ({subdivision} note at {bpm:.0f} BPM)")

    # --- Compressor settings (when preset forces comp, no settings provided) ---
    # Supplements DSP-selected comp from #14. Handles preset-forced comps
    # that come without settings (e.g., auto-presets from API research).
    if rig['comp_wah'] and not settings['comp_wah']:
        comp_name = rig['comp_wah'].get('name', '')
        dynamics_norm_fx = max(0.0, min(1.0, (rms_cv_fx - 0.10) / 0.30))

        if comp_name == 'LA Comp':
            ci = 0.4 + dynamics_norm_fx * 0.4
            settings['comp_wah'] = {
                "GAIN": round(5.0 + ci * 3.0, 1),
                "PEAK_REDUCTION": round(3.0 + ci * 4.0, 1),
            }
        elif comp_name == 'Sustain Comp':
            settings['comp_wah'] = {
                "LEVEL": round(6.0 + dynamics_norm_fx * 2.0, 1),
                "TONE": round(3.0 + air_norm_fx * 4.0, 1),
                "ATTACK": round(5.0 - dynamics_norm_fx * 2.0, 1),
                "SUSTAIN": round(4.0 + dynamics_norm_fx * 4.0, 1),
            }
        elif comp_name == 'Red Comp':
            settings['comp_wah'] = {
                "OUTPUT": round(5.0 + dynamics_norm_fx * 2.0, 1),
                "SENSITIVITY": round(4.0 + dynamics_norm_fx * 3.0, 1),
            }
        elif comp_name == 'Optical Comp':
            settings['comp_wah'] = {
                "VOLUME": round(5.5 + dynamics_norm_fx * 2.0, 1),
                "COMP": round(3.0 + dynamics_norm_fx * 3.0, 1),
            }
        elif comp_name == 'Bass Comp':
            settings['comp_wah'] = {
                "COMP": round(4.0 + dynamics_norm_fx * 3.0, 1),
                "GAIN": round(5.0 + dynamics_norm_fx * 2.0, 1),
            }

        if settings['comp_wah']:
            sources['comp_wah'] = "🔬"

    # --- COMPENSATORY PARAMETRIC EQ ---
    # The EQ compensates for tone-shaping already done by the drive pedal.
    # Without this, the EQ doubles up: a bass-heavy Muff gets bass EQ boost on top,
    # a dark fuzz gets treble EQ boost while the amp TREBLE is already cranked, etc.
    #
    # Architecture: The base formulas determine what the stem needs (from DSP analysis).
    # Then we subtract the drive pedal's contribution so the EQ only covers the gap.
    #
    # The amp's contribution is NOT subtracted here — the amp TREBLE/MIDDLE knobs
    # already compensate for target vs base_tone differences. The EQ handles the
    # drive's additional coloring that the amp knobs don't know about.
    #
    # Calibrated against Cherub Rock (Muff → Plexi) and Eruption (TS → Plexi).
    if not rig['mod_eq']:
        rig['mod_eq'] = next((eq for eq in db['eq'] if eq['name'] == 'Guitar EQ'), None)
        if rig['mod_eq']:
            # --- Drive compensation ---
            # Compute how much the drive pedal is already shaping bass, mids, and treble.
            # These values reduce the EQ's work in those bands.
            bass_comp = 0
            mid_comp = 0
            treble_comp = 0

            if rig['drive'] and 'sonic_profile' in rig['drive']:
                sp = rig['drive']['sonic_profile']
                gb = sp.get('gain_boost', 0)

                # Bass: high-gain drives add low-end energy from saturation harmonics.
                # Bass-heavy drives (Muff, Fuzz Face) add the most — big reduction.
                # Bass-cutting drives (Tube Screamer) still add some via amp saturation.
                # Scaling: gb * factor, where factor depends on bass character.
                # Validated: Muff (gb=0.9) → bass_comp=3.15, reduces +2.6 to -0.5 (Matt: 0.0)
                #            TS (gb=0.5) → bass_comp=1.0, reduces +1.5 to +0.5 (Matt: 0.0)
                if sp.get('cut_bass'):
                    bass_comp = gb * 2.0
                elif sp.get('scooped') or sp.get('wooly') or sp.get('retains_low_end'):
                    bass_comp = gb * 3.5
                else:
                    bass_comp = gb * 2.5

                # Mids: drives with negative mids values (scoopers) require the EQ to
                # cut mids harder to reinforce the scoop character. Drives with positive
                # mids (TS mid hump) need no compensation — the uncompensated formulas
                # already produce correct values for mid-boosting drives.
                #
                # Why one-directional: the Muff scoops 500-2000Hz, but the Spark 2's
                # Muff model + amp MIDDLE knob don't fully place the scoop where the
                # recording has it. The EQ needs to actively reinforce the scoop.
                # Meanwhile, the TS mid hump at low drive (boost mode) is nearly
                # transparent — Eruption's mid bands match without any compensation.
                #
                # Formula: mid_comp = gb × |negative mids|. Applied per-band with
                # different weights because the scoop affects upper mids (1600Hz)
                # much more than lower mids (400Hz).
                #
                # Per-band factors calibrated against Cherub Rock (Muff → Plexi):
                #   400Hz: ×1.5  (tested -1.7, was -1.0, gap -0.7)
                #   800Hz: ×1.3  (tested -1.0, was -0.4, gap -0.6)
                #   1600Hz: ×4.5 (tested -3.8, was -1.8, gap -2.0)
                # 1600Hz needs the largest factor — it's in the heart of the Muff's
                # scoop range and the most perceptually significant mid frequency.
                #
                # Validated: Muff (mids=-0.5) → mid_comp=0.45, hits all three bands.
                #            TS (mids=+0.4) → mid_comp=0.0, zero effect on Eruption.
                #            Black Op (mids=-0.2) → mid_comp=0.16, moderate correction.
                #            Fuzz Face (mids=-0.1) → mid_comp=0.07, light correction.
                #
                # STUB: Two anchor points (Muff, TS). Needs validation with 2-3 more
                # mid-range drives (Black Op, Fuzz Face, Booster) to confirm linearity
                # and per-band factors. Test Comfortably Numb, Teen Spirit, Stairway.
                sp_mids = sp.get('mids', 0)
                mid_comp = gb * max(0, -sp_mids)

                # Treble: when a drive is present, the amp TREBLE knob is already being
                # pushed high to compensate for the drive's interaction with the signal.
                # The EQ's treble boost needs to back off proportionally.
                # Scaling: gb * 1.5 — higher gain drives cause more treble interaction.
                # Validated: Muff (gb=0.9) → treble_comp=1.35, 5.9→4.6 dB (Matt: 4.6 ✓)
                #            TS (gb=0.5) → treble_comp=0.75, 3.7→3.0 dB (Matt: 3.7, 0.7 off)
                # Note: TS regression is because TS at low drive is nearly transparent.
                # Future: scale by drive amount if settings are available.
                treble_comp = gb * 1.5

            # --- Base EQ formulas (what the stem needs) minus drive compensation ---

            # 100Hz (Low Bass): Scales with gain, minus drive bass contribution
            eq_100 = round(np.clip((target_gain - 0.5) * 6.0 - bass_comp, -10.0, 10.0), 1)

            # 200Hz (Upper Bass): Same bass compensation, slightly attenuated (×0.8)
            eq_200 = round(np.clip((target_gain - 0.4) * 4.0 - bass_comp * 0.8, -10.0, 10.0), 1)

            # 400Hz (Low Mids): mid_comp × 1.5 — lightest correction in the mid range
            eq_400 = round(np.clip((target_mids - 0.5) * 6.0 - mid_comp * 1.5, -10.0, 10.0), 1)

            # 800Hz (Mid Mids): mid_comp × 1.3 — similar to 400Hz
            eq_800 = round(np.clip((target_mids - 0.4) * 5.0 - mid_comp * 1.3, -10.0, 10.0), 1)

            # 1600Hz (Upper Mids / Bite): mid_comp × 4.5 — strongest correction,
            # this is the heart of the Muff scoop and the most perceptually critical band
            eq_1600 = round(np.clip((target_mids - 0.55) * 8.0 - mid_comp * 4.5, -10.0, 10.0), 1)

            # 3200Hz (Air / Rangemaster zone): Minus drive treble compensation
            eq_3200 = round(np.clip((target_air - 0.38) * 25.0 - treble_comp, -10.0, 10.0), 1)

            # LEVEL: output compensation based on total boost/cut
            total_adjustment = eq_100 + eq_200 + eq_400 + eq_800 + eq_1600 + eq_3200
            avg_adjustment = total_adjustment / 6.0
            eq_level = round(np.clip(-avg_adjustment * 0.5, -3.0, 3.0), 1)

            settings['mod_eq'] = {
                "100HZ": eq_100,
                "200HZ": eq_200,
                "400HZ": eq_400,
                "800HZ": eq_800,
                "1600HZ": eq_1600,
                "3200HZ": eq_3200,
                "LEVEL": eq_level
            }
            sources['mod_eq'] = "🔬"

    # ALWAYS calculate amp settings from DSP analysis
    # UNLESS the preset supplied explicit amp_settings (base_tone mismatch would produce wrong knob values)
    if not settings['amp']:
        drive_boost = rig['drive']['sonic_profile'].get('gain_boost', 0) if (rig['drive'] and 'sonic_profile' in rig['drive']) else 0
        amp_gain = 5.0 + (target_gain - (rig['amp']['base_tone']['gain'] + drive_boost)) * 10
        # Floor at 2.0 when a drive pedal is present — even with a high-gain drive
        # providing the dirt, the amp needs some gain for body and response.
        # Without this, amps behind Big Muffs etc. can hit 0.0 which sounds thin/sterile.
        gain_floor = 2.0 if rig['drive'] else 0.0
        settings['amp']['GAIN'] = min(10.0, max(gain_floor, amp_gain))
        # TREBLE: Driven by spark_air (2400-6800Hz energy ratio, range 0.328-0.618).
        # Previous versions used spark_presence (1200-2400Hz) but that band has
        # std=0.021 across 48 stems — no discrimination. Air has real spread and
        # correctly ranks songs by brightness (Gravity=0.328 dark, Cherub Rock=0.618 bright).
        #
        # Formula: 4.0 + air_normalized * 5.0 → range 4.0 (darkest) to 9.0 (brightest).
        # Validated: Cherub Rock air=0.618 (air_normalized=1.0) → TREBLE 9.0 ✓
        #
        # Intentionally decoupled from amp base_tone.treble. The amp's character is
        # already expressed by which amp was selected (base_tone drives the distance
        # formula). Using it again here double-counted and produced absurd values
        # (TREBLE -3.5 → floored at 1.5) when a bright amp was chosen for a dark song.
        # The knob should reflect the song's brightness, not punish the amp mismatch twice.
        air_normalized = np.clip((target_air - 0.328) / (0.618 - 0.328), 0, 1)
        settings['amp']['TREBLE'] = round(min(10.0, max(4.0, 4.0 + air_normalized * 5.0)), 1)
        # MIDDLE: Driven by target_mids (500-2000Hz, clipped 0-1, typical range 0.30-0.65).
        # Formula: 3.0 + target_mids * 6.0 → range ~4.8 (scooped) to ~6.9 (mid-forward).
        # Same decoupling rationale as TREBLE — amp base_tone.mids already influenced
        # amp selection; using it in the knob formula double-counted and hit floor values
        # on high-mids amps paired with scooped songs.
        settings['amp']['MIDDLE'] = round(min(10.0, max(3.0, 3.0 + target_mids * 6.0)), 1)
        settings['amp']['BASS'], settings['amp']['VOLUME'] = 5.0, 8.0

        # --- MOD/EQ SLOT CONFLICT COMPENSATION ---
        # When the mod/eq slot is occupied by a modulation effect (Flanger, Chorus,
        # Phaser, UniVibe, Tremolo, etc.), Guitar EQ is unavailable. The amp's
        # TREBLE, MIDDLE, and BASS knobs must absorb the tonal shaping that the
        # 6-band EQ would have provided.
        #
        # This affects 55 presets — over 40% of the library. Without compensation,
        # the drive pedal's tonal contribution goes uncompensated: a Muff's bass
        # saturation isn't reduced, its mid scoop isn't reinforced, and the treble
        # interaction isn't balanced. The recipe sounds muddier and less defined
        # than it should.
        #
        # Approach: Compute "virtual EQ" — exactly what the EQ section would have
        # calculated — then translate to amp knob adjustments using a scaling factor.
        # The factor (0.25) converts EQ dB to amp knob units. It's conservative
        # because amp knobs are broadband (one knob covers all treble) while EQ
        # bands are narrowband (one slider at 3200Hz). A 1:1 conversion would
        # over-correct.
        #
        # Only fires when ALL three conditions are met:
        #   1. Amp settings are DSP-calculated (not preset amp_settings)
        #   2. Mod/EQ slot is occupied by a non-EQ modulation effect
        #   3. Checked after normal amp settings are computed (adjusts, not replaces)
        #
        # Mapping:
        #   Virtual 100Hz + 200Hz → amp BASS adjustment
        #   Virtual 400Hz + 800Hz + 1600Hz → amp MIDDLE adjustment
        #   Virtual 3200Hz → amp TREBLE adjustment
        mod_eq_is_modulation = (rig['mod_eq'] and
                                rig['mod_eq'].get('name') not in ('Guitar EQ', 'Bass EQ'))

        if mod_eq_is_modulation:
            # Compute drive compensation values (same formulas as EQ section)
            v_bass_comp = 0
            v_mid_comp = 0
            v_treble_comp = 0

            if rig['drive'] and 'sonic_profile' in rig['drive']:
                v_sp = rig['drive']['sonic_profile']
                v_gb = v_sp.get('gain_boost', 0)

                if v_sp.get('cut_bass'):
                    v_bass_comp = v_gb * 2.0
                elif v_sp.get('scooped') or v_sp.get('wooly') or v_sp.get('retains_low_end'):
                    v_bass_comp = v_gb * 3.5
                else:
                    v_bass_comp = v_gb * 2.5

                v_sp_mids = v_sp.get('mids', 0)
                v_mid_comp = v_gb * max(0, -v_sp_mids)
                v_treble_comp = v_gb * 1.5

            # Compute virtual EQ band values
            v_eq_100  = np.clip((target_gain - 0.5) * 6.0 - v_bass_comp, -10.0, 10.0)
            v_eq_200  = np.clip((target_gain - 0.4) * 4.0 - v_bass_comp * 0.8, -10.0, 10.0)
            v_eq_400  = np.clip((target_mids - 0.5) * 6.0 - v_mid_comp * 1.5, -10.0, 10.0)
            v_eq_800  = np.clip((target_mids - 0.4) * 5.0 - v_mid_comp * 1.3, -10.0, 10.0)
            v_eq_1600 = np.clip((target_mids - 0.55) * 8.0 - v_mid_comp * 4.5, -10.0, 10.0)
            v_eq_3200 = np.clip((target_air - 0.38) * 25.0 - v_treble_comp, -10.0, 10.0)

            # Scale virtual EQ into amp knob adjustments
            # 0.25 = conservative conversion (EQ dB → amp knob units)
            EQ_TO_AMP = 0.25
            bass_adj   = ((v_eq_100 + v_eq_200) / 2) * EQ_TO_AMP
            mid_adj    = ((v_eq_400 + v_eq_800 + v_eq_1600) / 3) * EQ_TO_AMP
            treble_adj = v_eq_3200 * EQ_TO_AMP

            settings['amp']['BASS']   = round(min(10.0, max(0.0, settings['amp']['BASS'] + bass_adj)), 1)
            settings['amp']['MIDDLE'] = round(min(10.0, max(2.5, settings['amp']['MIDDLE'] + mid_adj)), 1)
            settings['amp']['TREBLE'] = round(min(10.0, max(3.0, settings['amp']['TREBLE'] + treble_adj)), 1)

            # Log the compensation for recipe diagnostics
            mod_name = rig['mod_eq']['name']
            print(f"   ⚠️ EQ unavailable ({mod_name} in mod/eq slot) — amp knobs compensated: BASS {bass_adj:+.1f}, MIDDLE {mid_adj:+.1f}, TREBLE {treble_adj:+.1f}")

    # ==========================================================
    # CONFIDENCE SCORING (B2)
    # Classifies recipe reliability based on how it was sourced.
    #
    # Tier A — Hand-curated preset: keywords, forced_gear, and
    #   settings were manually verified. Highest trust.
    # Tier B — API-researched: Anthropic API identified the gear
    #   and generated a preset. Plausible but unverified.
    # Tier C — Pure DSP: no preset or API data. Gain, mids, air
    #   drive all the selections. Least reliable for specific
    #   artist tones, but solid for generic recipes.
    #
    # Uses preset_source (from check_artist_override) to distinguish
    # curated from auto-generated presets. The sources dict shows
    # which slots were preset-driven (🎨) vs. DSP (🔬) vs. API (🤖).
    # ==========================================================
    if override_rig and preset_source is None:
        confidence_tier = "A"
        confidence_label = "HIGH — Hand-curated artist preset"
    elif override_rig and preset_source == "auto":
        confidence_tier = "B"
        confidence_label = "MEDIUM — API-researched preset (unverified)"
    elif guitar_source == "🤖":
        confidence_tier = "B"
        confidence_label = "MEDIUM — API-assisted (guitar researched)"
    else:
        confidence_tier = "C"
        confidence_label = "LOW — Pure DSP analysis (no preset data)"

    print(f"   📊 Confidence: Tier {confidence_tier} — {confidence_label}")

    # Build output with source indicators
    # Handle multi-section header formatting
    if section_info and section_info.get('is_multi_section'):
        section_label = section_info['section_label']
        section_idx = section_info['section_index'] + 1
        section_count = section_info['section_count']
        start_ts = format_timestamp(section_info['section_start'])
        end_ts = format_timestamp(section_info['section_end'])
        
        recipe_text = [
            "═" * 60,
            f"SECTION {section_idx}/{section_count}: {section_label.upper()} ({start_ts} — {end_ts})",
            "═" * 60,
            f"🎸 SPARK 2 RECIPE: {song_name}",
            f"📅 Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"🎯 DSP Analysis: Gain {target_gain:.2f} | Presence {target_presence:.2f} | Air {target_air:.2f} | Mids {target_mids:.2f} | Width {target_width:.2f} | BPM {features.get('bpm', 120.0):.0f}",
            f"📊 Confidence: Tier {confidence_tier} — {confidence_label}",
            "",
            "Legend: 🔬 = Analysis-Based | 🎨 = Artist Signature | 🤖 = API-Researched",
            "="*50
        ]
    else:
        recipe_text = [
            f"🎸 SPARK 2 RECIPE (HYBRID MODE): {song_name}",
            f"📅 Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"🎯 DSP Analysis: Gain {target_gain:.2f} | Presence {target_presence:.2f} | Air {target_air:.2f} | Mids {target_mids:.2f} | Width {target_width:.2f} | BPM {features.get('bpm', 120.0):.0f}",
            f"📊 Confidence: Tier {confidence_tier} — {confidence_label}",
            "",
            "Legend: 🔬 = Analysis-Based | 🎨 = Artist Signature | 🤖 = API-Researched",
            "="*50
        ]

    # Guitar block — first in signal chain
    guitar_block = [
        f"[GUITAR] {guitar_info['guitar'].upper()} {guitar_source}",
        f"   • PICKUP: {guitar_info['pickup']}",
        f"   • VOLUME: {int(guitar_info['volume'])}",
        f"   • TONE: {int(guitar_info['tone'])}",
        "-"*50
    ]
    recipe_text.extend(guitar_block)

    slots = ["GATE", "COMP/WAH", "DRIVE", "AMP", "MOD/EQ", "DELAY", "REVERB"]
    gear_map = [rig['gate'], rig['comp_wah'], rig['drive'], rig['amp'], rig['mod_eq'], rig['delay'], rig['reverb']]
    setting_map = [settings['gate'], settings['comp_wah'], settings['drive'], settings['amp'], settings['mod_eq'], settings['delay'], settings['reverb']]
    source_map = [sources['gate'], sources['comp_wah'], sources['drive'], sources['amp'], sources['mod_eq'], sources['delay'], sources['reverb']]

    for i in range(len(slots)):
        block = format_settings(slots[i], gear_map[i], setting_map[i], source_map[i])
        if block: recipe_text.append(block)

    # Print to terminal
    recipe_output = "\n".join(recipe_text)
    print(recipe_output)

    # Save to recipes folder
    recipes_dir = "recipes"
    os.makedirs(recipes_dir, exist_ok=True)

    # Clean filename (remove extension, special chars)
    # Add section suffix for multi-section songs
    clean_name = os.path.splitext(song_name)[0]
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', clean_name)
    
    if section_info and section_info.get('is_multi_section'):
        section_suffix = f"_{section_info['section_label'].lower()}"
        clean_name = f"{clean_name}{section_suffix}"
    
    recipe_filename = os.path.join(recipes_dir, f"{clean_name}.txt")

    with open(recipe_filename, 'w') as f:
        f.write(recipe_output)

    # Save machine-readable JSON recipe alongside text version.
    # This enables the --tested diff tool, automated regression testing,
    # and future Bluetooth integration. Contains all structured data:
    # DSP features, gear selections, settings values, and sources.
    slot_keys = ['gate', 'comp_wah', 'drive', 'amp', 'mod_eq', 'delay', 'reverb']
    recipe_json = {
        "song": song_name,
        "date": datetime.datetime.now().isoformat(),
        "confidence": {
            "tier": confidence_tier,
            "label": confidence_label,
        },
        "dsp": {
            "gain": round(target_gain, 3),
            "mids": round(target_mids, 3),
            "presence": round(target_presence, 3),
            "air": round(target_air, 3),
            "width": round(target_width, 3),
            "bpm": round(features.get('bpm', 120.0), 1),
        },
        "guitar": guitar_info,
        "guitar_source": guitar_source,
        "rig": {},
        "settings": {},
        "sources": {}
    }
    
    # Add section metadata for multi-section songs
    if section_info and section_info.get('is_multi_section'):
        recipe_json["section"] = {
            "label": section_info['section_label'],
            "type": section_info['section_type'],
            "start": round(section_info['section_start'], 2),
            "end": round(section_info['section_end'], 2),
            "index": section_info['section_index'],
            "total_sections": section_info['section_count']
        }
    
    for k in slot_keys:
        recipe_json["rig"][k] = rig[k]['name'] if rig[k] else None
        recipe_json["settings"][k] = settings[k] if settings[k] else None
        recipe_json["sources"][k] = sources[k]

    json_filename = os.path.join(recipes_dir, f"{clean_name}.json")
    with open(json_filename, 'w') as f:
        json.dump(recipe_json, f, indent=2)

    print(f"\n💾 Recipe saved to: {recipe_filename}")
    print(f"💾 JSON saved to:   {json_filename}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Spark AI Tone Engineer")
    parser.add_argument("audio", nargs="?", help="Path to guitar stem audio file")
    parser.add_argument("--research", type=str, help="Force API research for a song/artist (no audio needed)")
    parser.add_argument("--compare", type=str, help="Compare API research against existing preset")
    parser.add_argument("--forget", type=str, help="Delete auto-generated preset matching keyword")
    parser.add_argument("--forget-all-auto", action="store_true", help="Delete ALL auto-generated presets (preserves hand-curated)")
    parser.add_argument("--fill-guitar", action="store_true", help="Batch-fill missing guitar blocks via API")
    parser.add_argument("--no-research", action="store_true", help="Skip automatic API research for unknown songs")
    parser.add_argument("--tested", type=str, help="Record amp test results for a recipe (e.g. --tested cherubrock)")
    parser.add_argument("--test-results", action="store_true", help="Show all test results and systematic biases")
    parser.add_argument("--batch", type=str, help="Generate recipes for all WAV files in a directory (recursive)")
    parser.add_argument("--no-sections", action="store_true", help="Disable multi-section detection (analyze full track)")
    parser.add_argument("--clean", action="store_true", help="Delete all files in recipes/ folder before running")

    args = parser.parse_args()

    # Handle --clean flag (works standalone or with other modes)
    if args.clean:
        import shutil
        recipes_dir = "recipes"
        if os.path.exists(recipes_dir):
            count = len([f for f in os.listdir(recipes_dir) if os.path.isfile(os.path.join(recipes_dir, f))])
            shutil.rmtree(recipes_dir)
            os.makedirs(recipes_dir, exist_ok=True)
            print(f"🗑️  Cleaned recipes/ folder ({count} files deleted)")
        else:
            os.makedirs(recipes_dir, exist_ok=True)
            print(f"🗑️  Created empty recipes/ folder")
        
        # If --clean was the only flag, exit
        if not args.audio and not args.batch and not args.research and not args.compare:
            return

    # Research-only modes (no audio file needed)
    if args.research or args.compare or args.forget or args.forget_all_auto or args.fill_guitar:
        try:
            import api_research
        except ImportError:
            print("❌ api_research.py not found")
            return

        db = load_gear_db()
        if not db: return

        if args.forget_all_auto:
            api_research.forget_all_auto()
        elif args.forget:
            api_research.forget_preset(args.forget)
        elif args.compare:
            api_research.compare_preset(args.compare, db)
        elif args.research:
            api_research.research_song(args.research, db, save=True)
        elif args.fill_guitar:
            api_research.fill_guitar_blocks(db)
        return

    # Test results modes
    if args.tested:
        record_test_results(args.tested)
        return

    if args.test_results:
        show_test_results()
        return

    # Batch mode: generate recipes for all WAV files in a directory
    if args.batch:
        batch_recipes(args.batch, skip_research=args.no_research, skip_sections=args.no_sections)
        return

    # Normal mode: analyze audio and build rig
    if not args.audio:
        parser.print_help()
        return

    song_name = os.path.basename(args.audio)
    
    if args.no_sections:
        # Single-section mode: analyze full track (legacy behavior)
        result = analyze_tone(args.audio)
        if result is None:
            return
        correct_bpm_with_api(result, song_name)
        build_rig(result, song_name=song_name, skip_research=args.no_research)
    else:
        # Multi-section mode: detect and analyze sections separately
        result = analyze_with_sections(args.audio)
        if result is None:
            return
        
        if result['is_multi_section']:
            print(f"\n{'═'*60}")
            print(f"⚡ MULTI-SECTION TRACK DETECTED — Generating {len(result['sections'])} recipes")
            print(f"{'═'*60}")
        
        # BPM API correction: look up once, apply to each section independently.
        # Different sections may have different librosa BPMs (e.g., section 1 = 99,
        # sections 2-6 = 144). Each needs its own octave-error check against the API.
        api_result = lookup_bpm_api(song_name)
        if api_result is not None:
            api_bpm = api_result['bpm']
            title = api_result.get('title', '')
            artist = api_result.get('artist', '')
            for sec in result['sections']:
                librosa_bpm = sec.get('bpm', 120.0)
                ratio = librosa_bpm / api_bpm if api_bpm > 0 else 0
                if 1.7 < ratio < 2.3:
                    corrected = librosa_bpm / 2.0
                    sec['bpm'] = corrected
                    print(f"   🎵 BPM corrected: {librosa_bpm:.0f} → {corrected:.0f} ({title} by {artist} = {api_bpm:.0f} BPM)")
                elif 0.85 < ratio < 1.15:
                    pass  # confirmed, no change needed
            # Print summary
            bpms = set(sec.get('bpm', 120.0) for sec in result['sections'])
            print(f"   🎵 BPM API: {title} by {artist} = {api_bpm:.0f} BPM (section BPMs: {', '.join(f'{b:.0f}' for b in sorted(bpms))})")
        
        for i, section_features in enumerate(result['sections']):
            section_info = None
            
            if result['is_multi_section']:
                section_info = {
                    'is_multi_section': True,
                    'section_index': i,
                    'section_count': len(result['sections']),
                    'section_start': section_features['section_start'],
                    'section_end': section_features['section_end'],
                    'section_label': section_features['section_label'],
                    'section_type': section_features['section_type']
                }
            
            build_rig(section_features,
                      song_name=song_name,
                      skip_research=args.no_research,
                      section_info=section_info)


def batch_recipes(stems_dir, skip_research=False, skip_sections=False):
    """
    Generate recipes for all WAV files in a directory (recursive).

    Usage: python main.py --batch /path/to/stems
           python main.py --batch /path/to/stems --no-research
           python main.py --batch /path/to/stems --no-sections

    Finds all .wav files, runs each through the full pipeline (analyze +
    build_rig), and outputs a summary table at the end. Recipes are saved
    to recipes/ as usual (both .txt and .json).

    Use --no-research to skip API calls for unknown songs (faster, DSP-only).
    Use --no-sections to disable multi-section detection (single recipe per track).
    """
    import glob

    if not os.path.exists(stems_dir):
        print(f"❌ Directory not found: {stems_dir}")
        return

    wav_files = sorted(glob.glob(os.path.join(stems_dir, "**", "*.wav"), recursive=True))

    if not wav_files:
        print(f"❌ No WAV files found in {stems_dir}")
        return

    print(f"\n🎸 BATCH RECIPE GENERATION — {len(wav_files)} stems")
    print(f"   Source: {stems_dir}")
    print(f"   API research: {'disabled' if skip_research else 'enabled'}")
    print(f"   Section detection: {'disabled' if skip_sections else 'enabled'}")
    print(f"{'='*70}")

    results = []
    errors = []

    for i, wav_path in enumerate(wav_files):
        song_name = os.path.basename(wav_path)
        print(f"\n[{i+1}/{len(wav_files)}] {song_name}")
        print(f"   Path: {wav_path}")

        try:
            if skip_sections:
                # Single-section mode: analyze full track
                result_single = analyze_tone(wav_path)
                if result_single is None:
                    errors.append((song_name, "Analysis failed"))
                    continue
                
                correct_bpm_with_api(result_single, song_name)
                build_rig(result_single, song_name=song_name, skip_research=skip_research, save_presets=False)
                
                # Read the JSON recipe to capture summary data
                clean_name = os.path.splitext(song_name)[0]
                clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', clean_name)
                json_path = os.path.join("recipes", f"{clean_name}.json")
                
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        recipe = json.load(f)
                    results.append({
                        'file': song_name,
                        'gain': result_single['gain'],
                        'mids': result_single['mids'],
                        'air': result_single['air'],
                        'harm_ratio': result_single.get('harmonic_ratio', 0),
                        'amp': recipe.get('rig', {}).get('amp', 'None'),
                        'drive': recipe.get('rig', {}).get('drive', 'None'),
                        'mod_eq': recipe.get('rig', {}).get('mod_eq', 'None'),
                        'guitar': recipe.get('guitar', {}).get('guitar', '?'),
                    })
                else:
                    results.append({
                        'file': song_name,
                        'gain': result_single['gain'],
                        'mids': result_single['mids'],
                        'air': result_single['air'],
                        'harm_ratio': result_single.get('harmonic_ratio', 0),
                        'amp': '?', 'drive': '?', 'mod_eq': '?', 'guitar': '?',
                    })
            else:
                # Multi-section mode
                result = analyze_with_sections(wav_path)
                if result is None:
                    errors.append((song_name, "Analysis failed"))
                    continue

                # Handle multi-section songs
                sections = result['sections']
                is_multi = result['is_multi_section']
                
                if is_multi:
                    print(f"   ⚡ Multi-section: {len(sections)} sections detected")
                
                # BPM API correction: look up once, apply to each section independently
                api_result = lookup_bpm_api(song_name)
                if api_result is not None:
                    api_bpm = api_result['bpm']
                    title = api_result.get('title', '')
                    artist = api_result.get('artist', '')
                    for sec in sections:
                        librosa_bpm = sec.get('bpm', 120.0)
                        ratio = librosa_bpm / api_bpm if api_bpm > 0 else 0
                        if 1.7 < ratio < 2.3:
                            corrected = librosa_bpm / 2.0
                            sec['bpm'] = corrected
                            print(f"   🎵 BPM corrected: {librosa_bpm:.0f} → {corrected:.0f} ({title} by {artist} = {api_bpm:.0f} BPM)")
                    bpms = set(sec.get('bpm', 120.0) for sec in sections)
                    print(f"   🎵 BPM API: {title} by {artist} = {api_bpm:.0f} BPM (section BPMs: {', '.join(f'{b:.0f}' for b in sorted(bpms))})")
                
                for sec_idx, section_features in enumerate(sections):
                    section_info = None
                    
                    if is_multi:
                        section_info = {
                            'is_multi_section': True,
                            'section_index': sec_idx,
                            'section_count': len(sections),
                            'section_start': section_features['section_start'],
                            'section_end': section_features['section_end'],
                            'section_label': section_features['section_label'],
                            'section_type': section_features['section_type']
                        }
                    
                    build_rig(section_features, song_name=song_name, skip_research=skip_research, 
                              save_presets=False, section_info=section_info)

                    # Determine the filename suffix for multi-section
                    clean_name = os.path.splitext(song_name)[0]
                    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', clean_name)
                    if is_multi:
                        clean_name = f"{clean_name}_{section_features['section_label'].lower()}"
                    
                    # Read the JSON recipe to capture summary data
                    json_path = os.path.join("recipes", f"{clean_name}.json")

                    display_name = song_name if not is_multi else f"{song_name} [{section_features['section_label']}]"
                    
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            recipe = json.load(f)
                        results.append({
                            'file': display_name,
                            'gain': section_features['gain'],
                            'mids': section_features['mids'],
                            'air': section_features['air'],
                            'harm_ratio': section_features.get('harmonic_ratio', 0),
                            'amp': recipe.get('rig', {}).get('amp', 'None'),
                            'drive': recipe.get('rig', {}).get('drive', 'None'),
                            'mod_eq': recipe.get('rig', {}).get('mod_eq', 'None'),
                            'guitar': recipe.get('guitar', {}).get('guitar', '?'),
                        })
                    else:
                        results.append({
                            'file': display_name,
                            'gain': section_features['gain'],
                            'mids': section_features['mids'],
                            'air': section_features['air'],
                            'harm_ratio': section_features.get('harmonic_ratio', 0),
                            'amp': '?', 'drive': '?', 'mod_eq': '?', 'guitar': '?',
                        })

        except Exception as e:
            errors.append((song_name, str(e)))
            print(f"   ❌ Error: {e}")

    # Summary table
    print(f"\n\n{'='*120}")
    print(f"📊 BATCH SUMMARY — {len(results)} recipes generated, {len(errors)} errors")
    print(f"{'='*120}")
    print(f"  {'File':<35} {'Gain':>5} {'Mids':>5} {'Air':>5} {'H.R.':>5} {'Amp':<20} {'Drive':<18} {'Guitar':<10}")
    print("  " + "-" * 115)

    for r in sorted(results, key=lambda x: x['gain']):
        g_short = 'SSH' if 'SSH' in str(r['guitar']) else 'SSS'
        drive = r['drive'] if r['drive'] else '-'
        print(f"  {r['file']:<35} {r['gain']:>5.2f} {r['mids']:>5.2f} {r['air']:>5.3f} {r['harm_ratio']:>5.2f} {r['amp']:<20} {drive:<18} {g_short}")

    if errors:
        print(f"\n  ❌ ERRORS:")
        for song, err in errors:
            print(f"     {song}: {err}")

    print(f"\n💾 All recipes saved to recipes/")


def record_test_results(song_keyword):
    """
    Interactive diff tool: walk through a recipe's parameters and record
    what was actually dialed in at the amp.

    Usage: python main.py --tested cherubrock

    Reads the JSON recipe, presents each parameter, and asks for the
    tested value. Press Enter to accept the recipe value (it was correct).
    Type a new number to record what you changed it to.
    Results are saved to test_results.json for calibration analysis.
    """
    # Find the JSON recipe
    recipes_dir = "recipes"
    clean = re.sub(r'[^a-zA-Z0-9_-]', '_', song_keyword.lower()).strip('_')

    # Try exact match, then partial
    json_path = os.path.join(recipes_dir, f"{clean}.json")
    if not os.path.exists(json_path):
        # Search for partial match
        import glob
        candidates = glob.glob(os.path.join(recipes_dir, f"*{clean}*.json"))
        if len(candidates) == 1:
            json_path = candidates[0]
        elif len(candidates) > 1:
            print(f"❌ Multiple recipes match '{song_keyword}':")
            for c in candidates:
                print(f"   {os.path.basename(c)}")
            return
        else:
            print(f"❌ No recipe found matching '{song_keyword}'")
            print(f"   Run the recipe first: python main.py stem.wav")
            return

    with open(json_path, 'r') as f:
        recipe = json.load(f)

    song = recipe.get('song', song_keyword)
    print(f"\n🎸 RECORDING TEST RESULTS: {song}")
    print(f"   Recipe: {json_path}")
    print(f"\n   For each parameter: Enter = recipe value was correct, or type your tested value.")
    print(f"   Type 'skip' to skip an entire section, 'done' to finish early.\n")

    tested = {}
    deltas = {}

    # Walk through each slot that has settings
    slot_labels = {
        'gate': 'NOISE GATE',
        'comp_wah': 'COMP/WAH',
        'drive': 'DRIVE',
        'amp': 'AMP',
        'mod_eq': 'MOD/EQ',
        'delay': 'DELAY',
        'reverb': 'REVERB'
    }

    done_early = False
    for slot, label in slot_labels.items():
        if done_early:
            break

        gear_name = recipe['rig'].get(slot)
        slot_settings = recipe['settings'].get(slot)
        if not gear_name or not slot_settings:
            continue

        print(f"   [{label}] {gear_name}")
        tested[slot] = {}
        deltas[slot] = {}

        for param, recipe_val in sorted(slot_settings.items()):
            try:
                recipe_val = float(recipe_val)
            except (ValueError, TypeError):
                # Skip non-numeric params (toggle switches etc.)
                tested[slot][param] = recipe_val
                continue

            response = input(f"      {param}: {recipe_val:.1f} → ").strip()

            if response.lower() == 'done':
                done_early = True
                break
            elif response.lower() == 'skip':
                # Accept all remaining params in this slot as correct
                for p2, v2 in sorted(slot_settings.items()):
                    if p2 not in tested[slot]:
                        try:
                            tested[slot][p2] = float(v2)
                        except (ValueError, TypeError):
                            tested[slot][p2] = v2
                break
            elif response == '':
                # Recipe value was correct
                tested[slot][param] = recipe_val
                deltas[slot][param] = 0.0
            else:
                try:
                    tested_val = float(response)
                    tested[slot][param] = tested_val
                    deltas[slot][param] = round(tested_val - recipe_val, 1)
                except ValueError:
                    print(f"         (invalid, using recipe value)")
                    tested[slot][param] = recipe_val
                    deltas[slot][param] = 0.0

    # Ask for notes
    print()
    notes = input("   Notes (optional, press Enter to skip): ").strip()

    # Build the test result entry
    result_entry = {
        "song": song,
        "date": datetime.datetime.now().isoformat(),
        "recipe_file": json_path,
        "dsp": recipe.get('dsp', {}),
        "guitar": recipe.get('guitar', {}),
        "rig": recipe.get('rig', {}),
        "recipe_settings": recipe.get('settings', {}),
        "tested_settings": tested,
        "deltas": deltas,
        "notes": notes
    }

    # Save to test_results.json
    results_path = "test_results.json"
    try:
        with open(results_path, 'r') as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = []

    all_results.append(result_entry)

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 TEST RESULT SUMMARY: {song}")
    print(f"{'='*60}")

    has_changes = False
    for slot, slot_deltas in deltas.items():
        slot_changes = {p: d for p, d in slot_deltas.items() if d != 0.0}
        if slot_changes:
            has_changes = True
            gear_name = recipe['rig'].get(slot, slot)
            print(f"\n   [{slot_labels.get(slot, slot)}] {gear_name}")
            for param, delta in sorted(slot_changes.items()):
                recipe_val = float(recipe['settings'][slot][param])
                tested_val = tested[slot][param]
                print(f"      {param}: {recipe_val:.1f} → {tested_val:.1f} ({delta:+.1f})")

    if not has_changes:
        print(f"\n   ✅ All parameters matched! Recipe was perfect.")

    if notes:
        print(f"\n   📝 {notes}")

    print(f"\n💾 Saved to {results_path} ({len(all_results)} total test{'s' if len(all_results) != 1 else ''})")


def show_test_results():
    """
    Display all test results with faceted calibration analysis.

    Usage: python main.py --test-results

    Five analysis layers:
      1. Per-song summary (what changed for each tested song)
      2. Global bias (flat aggregation — which params are consistently off)
      3. By drive pedal (e.g., "all Muff songs: TREBLE +1.5")
      4. By amp model (e.g., "Plexiglas: MIDDLE consistently -0.8")
      5. By gain range (e.g., "clean recipes: BASS +1.0")
      6. EQ band correlation (which EQ bands track together — feeds A2)
      7. Per-pedal drive delta (how drive amount/tone/volume differ — feeds A4)

    Also exports a structured calibration summary to test_calibration.json
    for use in coefficient refitting (feeds K4 training pipeline).
    """
    results_path = "test_results.json"
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("❌ No test results found. Use --tested to record results.")
        return

    if not results:
        print("❌ No test results found.")
        return

    print(f"\n{'='*70}")
    print(f"📊 TEST RESULTS DATABASE — {len(results)} tested songs")
    print(f"{'='*70}")

    # ── 1. Per-Song Summary ─────────────────────────────────────
    for r in results:
        song = r.get('song', '?')
        date = r.get('date', '?')[:10]
        notes = r.get('notes', '')
        gain = r.get('dsp', {}).get('gain', '?')
        drive = r.get('rig', {}).get('drive', 'None')
        amp = r.get('rig', {}).get('amp', '?')

        changes = []
        for slot, slot_deltas in r.get('deltas', {}).items():
            for param, delta in slot_deltas.items():
                if delta != 0.0:
                    changes.append(f"{param} {delta:+.1f}")
        change_str = ', '.join(changes) if changes else 'Perfect match'

        gain_str = f"{gain:.2f}" if isinstance(gain, (int, float)) else str(gain)
        print(f"\n   {song} ({date})  [gain={gain_str}]")
        print(f"      Rig: {drive or 'No drive'} → {amp}")
        print(f"      Deltas: {change_str}")
        if notes:
            print(f"      Notes: {notes}")

    # Need 2+ results for statistical analysis
    if len(results) < 2:
        print(f"\n   ⚠️ Need 2+ tested songs for bias analysis. Keep testing!")
        return

    # ── Helper: collect deltas with metadata ────────────────────
    # Each delta entry carries its context so we can slice any way we want.
    delta_entries = []
    for r in results:
        song = r.get('song', '?')
        dsp = r.get('dsp', {})
        rig = r.get('rig', {})
        gain = dsp.get('gain', 0.5)

        # Classify gain range
        if gain < 0.35:
            gain_range = 'clean'
        elif gain < 0.55:
            gain_range = 'crunch'
        elif gain < 0.75:
            gain_range = 'drive'
        else:
            gain_range = 'high_gain'

        for slot, slot_deltas in r.get('deltas', {}).items():
            gear_name = rig.get(slot)
            for param, delta in slot_deltas.items():
                delta_entries.append({
                    'song': song,
                    'slot': slot,
                    'param': param,
                    'delta': delta,
                    'gain': gain,
                    'gain_range': gain_range,
                    'drive': rig.get('drive'),
                    'amp': rig.get('amp'),
                    'gear': gear_name,
                    'key': f"{slot}.{param}",
                })

    if not delta_entries:
        print(f"\n   No delta data to analyze.")
        return

    def report_biases(entries, label, min_count=2, threshold=0.3):
        """
        Report consistent biases from a list of delta entries.
        Groups by slot.param and reports mean, direction, and per-song values.
        Returns list of bias dicts for export.
        """
        from collections import defaultdict
        grouped = defaultdict(list)
        songs_grouped = defaultdict(list)
        for e in entries:
            grouped[e['key']].append(e['delta'])
            songs_grouped[e['key']].append(e['song'])

        biases = []
        for key in sorted(grouped.keys()):
            vals = grouped[key]
            if len(vals) >= min_count:
                mean = np.mean(vals)
                if abs(mean) >= threshold:
                    biases.append({
                        'key': key,
                        'mean': float(round(mean, 2)),
                        'count': len(vals),
                        'values': [round(v, 1) for v in vals],
                        'songs': songs_grouped[key],
                    })

        if biases:
            biases.sort(key=lambda x: -abs(x['mean']))
            for b in biases:
                direction = "too high" if b['mean'] > 0 else "too low"
                vals_str = ', '.join(f"{v:+.1f}" for v in b['values'])
                song_str = ', '.join(b['songs'])
                print(f"      {b['key']:<25} mean={b['mean']:+.1f} ({direction}) n={b['count']} [{vals_str}]")
                print(f"      {'':25} songs: {song_str}")
        else:
            print(f"      No consistent biases detected (threshold ±{threshold}).")

        return biases

    # ── 2. Global Bias (flat, like the old version) ─────────────
    print(f"\n{'='*70}")
    print(f"📊 GLOBAL BIAS ANALYSIS")
    print(f"{'='*70}")
    print(f"\n   All parameters with consistent bias (mean > ±0.3 across 2+ songs):\n")
    global_biases = report_biases(delta_entries, "global")

    # ── 3. By Drive Pedal ───────────────────────────────────────
    # "All Muff songs have treble +1.5" — feeds A4
    from collections import defaultdict
    drive_groups = defaultdict(list)
    for e in delta_entries:
        drive_name = e['drive'] or 'No drive'
        drive_groups[drive_name].append(e)

    if len(drive_groups) >= 1:
        print(f"\n{'='*70}")
        print(f"📊 BIAS BY DRIVE PEDAL — feeds A4 (per-pedal drive curves)")
        print(f"{'='*70}")

        for drive_name in sorted(drive_groups.keys()):
            entries = drive_groups[drive_name]
            n_songs = len(set(e['song'] for e in entries))
            if n_songs < 1:
                continue
            print(f"\n   [{drive_name}] ({n_songs} song{'s' if n_songs != 1 else ''})")

            if n_songs >= 2:
                report_biases(entries, drive_name)
            else:
                # Even with 1 song, show the non-zero deltas — they're still informative
                non_zero = [e for e in entries if e['delta'] != 0.0]
                if non_zero:
                    for e in sorted(non_zero, key=lambda x: -abs(x['delta'])):
                        print(f"      {e['key']:<25} {e['delta']:+.1f}  ({e['song']})")
                else:
                    print(f"      Perfect match.")

    # ── 4. By Amp Model ─────────────────────────────────────────
    # "Plexiglas MIDDLE is consistently -0.8" — feeds A1
    amp_groups = defaultdict(list)
    for e in delta_entries:
        if e['slot'] == 'amp':
            amp_name = e['amp'] or '?'
            amp_groups[amp_name].append(e)

    if amp_groups:
        print(f"\n{'='*70}")
        print(f"📊 AMP KNOB BIAS BY AMP MODEL — feeds A1 (knob scaling)")
        print(f"{'='*70}")

        for amp_name in sorted(amp_groups.keys()):
            entries = amp_groups[amp_name]
            n_songs = len(set(e['song'] for e in entries))
            print(f"\n   [{amp_name}] ({n_songs} song{'s' if n_songs != 1 else ''})")

            if n_songs >= 2:
                report_biases(entries, amp_name)
            else:
                non_zero = [e for e in entries if e['delta'] != 0.0]
                if non_zero:
                    for e in sorted(non_zero, key=lambda x: -abs(x['delta'])):
                        print(f"      {e['key']:<25} {e['delta']:+.1f}  ({e['song']})")
                else:
                    print(f"      Perfect match.")

    # ── 5. By Gain Range ────────────────────────────────────────
    # "Clean recipes have BASS +1.0" — feeds A1 (non-uniform scaling)
    gain_labels = {'clean': 'Clean (< 0.35)', 'crunch': 'Crunch (0.35-0.55)',
                   'drive': 'Drive (0.55-0.75)', 'high_gain': 'High Gain (≥ 0.75)'}
    gain_groups = defaultdict(list)
    for e in delta_entries:
        gain_groups[e['gain_range']].append(e)

    if gain_groups:
        print(f"\n{'='*70}")
        print(f"📊 BIAS BY GAIN RANGE — feeds A1 (non-uniform knob scaling)")
        print(f"{'='*70}")

        for gr in ['clean', 'crunch', 'drive', 'high_gain']:
            if gr not in gain_groups:
                continue
            entries = gain_groups[gr]
            n_songs = len(set(e['song'] for e in entries))
            print(f"\n   [{gain_labels[gr]}] ({n_songs} song{'s' if n_songs != 1 else ''})")

            if n_songs >= 2:
                report_biases(entries, gr)
            else:
                non_zero = [e for e in entries if e['delta'] != 0.0]
                if non_zero:
                    for e in sorted(non_zero, key=lambda x: -abs(x['delta'])):
                        print(f"      {e['key']:<25} {e['delta']:+.1f}  ({e['song']})")
                else:
                    print(f"      Perfect match.")

    # ── 6. EQ Band Correlation ──────────────────────────────────
    # Do EQ bands move together? If 400/800/1600 all shift the same
    # direction, the mid_comp factor is systematically off. Feeds A2.
    eq_bands = ['100HZ', '200HZ', '400HZ', '800HZ', '1600HZ', '3200HZ']
    eq_songs = defaultdict(dict)  # song -> {band: delta}
    for e in delta_entries:
        if e['slot'] == 'mod_eq' and e['param'] in eq_bands:
            eq_songs[e['song']][e['param']] = e['delta']

    songs_with_eq = {s: bands for s, bands in eq_songs.items() if len(bands) >= 3}

    if songs_with_eq:
        print(f"\n{'='*70}")
        print(f"📊 EQ BAND CORRELATION — feeds A2 (mid-band calibration)")
        print(f"{'='*70}")

        # Per-song EQ delta profile
        print(f"\n   Per-song EQ deltas (columns = bands):")
        header = f"      {'Song':<25}"
        for band in eq_bands:
            header += f" {band:>7}"
        print(header)
        print(f"      {'-'*25} {'-------' * len(eq_bands)}")

        for song in sorted(songs_with_eq.keys()):
            bands = songs_with_eq[song]
            row = f"      {song:<25}"
            for band in eq_bands:
                val = bands.get(band)
                if val is not None:
                    row += f" {val:>+7.1f}"
                else:
                    row += f" {'—':>7}"
            print(row)

        # Bass bands (100+200) vs mid bands (400+800+1600) vs treble (3200)
        # correlation — do they move as groups?
        bass_deltas = []
        mid_deltas = []
        treble_deltas = []

        for song, bands in songs_with_eq.items():
            b_vals = [bands.get(b) for b in ['100HZ', '200HZ'] if b in bands]
            m_vals = [bands.get(b) for b in ['400HZ', '800HZ', '1600HZ'] if b in bands]
            t_vals = [bands.get(b) for b in ['3200HZ'] if b in bands]

            if b_vals:
                bass_deltas.append(np.mean(b_vals))
            if m_vals:
                mid_deltas.append(np.mean(m_vals))
            if t_vals:
                treble_deltas.append(np.mean(t_vals))

        print(f"\n   Band group averages across {len(songs_with_eq)} songs:")
        if bass_deltas:
            mean_b = np.mean(bass_deltas)
            print(f"      Bass (100+200Hz):      mean={mean_b:+.2f}  {'← bass_comp may be too aggressive' if mean_b < -0.5 else '← bass_comp may be too weak' if mean_b > 0.5 else '(within range)'}")
        if mid_deltas:
            mean_m = np.mean(mid_deltas)
            print(f"      Mids (400+800+1600Hz): mean={mean_m:+.2f}  {'← mid_comp may be too aggressive' if mean_m < -0.5 else '← mid_comp may be too weak' if mean_m > 0.5 else '(within range)'}")
        if treble_deltas:
            mean_t = np.mean(treble_deltas)
            print(f"      Treble (3200Hz):       mean={mean_t:+.2f}  {'← treble_comp may be too aggressive' if mean_t < -0.5 else '← treble_comp may be too weak' if mean_t > 0.5 else '(within range)'}")

        # Check if mid bands correlate (same sign = systematic mid_comp issue)
        if len(songs_with_eq) >= 2:
            mid_band_names = ['400HZ', '800HZ', '1600HZ']
            all_same_sign = True
            for song, bands in songs_with_eq.items():
                m_vals = [bands.get(b, 0) for b in mid_band_names if b in bands]
                if m_vals and not (all(v >= 0 for v in m_vals) or all(v <= 0 for v in m_vals)):
                    all_same_sign = False

            if all_same_sign and mid_deltas and abs(np.mean(mid_deltas)) > 0.2:
                direction = "all cutting too hard" if np.mean(mid_deltas) < 0 else "all boosting too hard"
                print(f"\n   ⚠️ Mid bands (400/800/1600) are {direction} across all songs.")
                print(f"      This suggests mid_comp factor or base formula needs adjustment.")

    # ── 7. Per-Pedal Drive Delta ────────────────────────────────
    # Show drive parameter deltas grouped by pedal — feeds A4.
    drive_slot_entries = [e for e in delta_entries if e['slot'] == 'drive']
    if drive_slot_entries:
        pedal_drive_deltas = defaultdict(lambda: defaultdict(list))
        pedal_drive_songs = defaultdict(lambda: defaultdict(list))
        for e in drive_slot_entries:
            pedal = e['gear'] or '?'
            pedal_drive_deltas[pedal][e['param']].append(e['delta'])
            pedal_drive_songs[pedal][e['param']].append(e['song'])

        print(f"\n{'='*70}")
        print(f"📊 PER-PEDAL DRIVE KNOB DELTAS — feeds A4 (response curves)")
        print(f"{'='*70}")

        for pedal in sorted(pedal_drive_deltas.keys()):
            params = pedal_drive_deltas[pedal]
            n_songs = len(set(s for songs in pedal_drive_songs[pedal].values() for s in songs))
            print(f"\n   [{pedal}] ({n_songs} song{'s' if n_songs != 1 else ''})")

            for param in sorted(params.keys()):
                vals = params[param]
                songs_list = pedal_drive_songs[pedal][param]
                mean = np.mean(vals)
                if len(vals) == 1:
                    print(f"      {param:<15} delta={vals[0]:+.1f}  ({songs_list[0]})")
                else:
                    vals_str = ', '.join(f"{v:+.1f}" for v in vals)
                    direction = ""
                    if abs(mean) > 0.3:
                        direction = f" ← {'reduce' if mean > 0 else 'increase'} {param.lower()} formula"
                    print(f"      {param:<15} mean={mean:+.1f}  n={len(vals)}  [{vals_str}]{direction}")

    # ── 8. Export Calibration Summary ───────────────────────────
    # Structured JSON for coefficient refitting (feeds K4).
    calibration = {
        "generated": datetime.datetime.now().isoformat(),
        "n_songs": len(results),
        "songs": [r.get('song', '?') for r in results],
        "global_biases": [],
        "by_drive": {},
        "by_amp": {},
        "by_gain_range": {},
        "eq_correlation": {},
        "per_pedal_drive": {},
    }

    # Global biases
    from collections import defaultdict
    all_deltas_grouped = defaultdict(list)
    for e in delta_entries:
        all_deltas_grouped[e['key']].append(e['delta'])
    for key, vals in all_deltas_grouped.items():
        calibration['global_biases'].append({
            'param': key,
            'mean': round(float(np.mean(vals)), 3),
            'std': round(float(np.std(vals)), 3),
            'n': len(vals),
            'values': [round(v, 2) for v in vals],
        })

    # By drive
    for drive_name, entries in drive_groups.items():
        drive_deltas = defaultdict(list)
        for e in entries:
            drive_deltas[e['key']].append(e['delta'])
        calibration['by_drive'][drive_name] = {
            k: {'mean': round(float(np.mean(v)), 3), 'n': len(v)}
            for k, v in drive_deltas.items()
        }

    # By amp
    for amp_name, entries in amp_groups.items():
        amp_deltas_g = defaultdict(list)
        for e in entries:
            amp_deltas_g[e['key']].append(e['delta'])
        calibration['by_amp'][amp_name] = {
            k: {'mean': round(float(np.mean(v)), 3), 'n': len(v)}
            for k, v in amp_deltas_g.items()
        }

    # By gain range
    for gr, entries in gain_groups.items():
        gr_deltas = defaultdict(list)
        for e in entries:
            gr_deltas[e['key']].append(e['delta'])
        calibration['by_gain_range'][gr] = {
            k: {'mean': round(float(np.mean(v)), 3), 'n': len(v)}
            for k, v in gr_deltas.items()
        }

    # EQ correlation
    if songs_with_eq:
        calibration['eq_correlation'] = {
            'bass_mean': round(float(np.mean(bass_deltas)), 3) if bass_deltas else None,
            'mid_mean': round(float(np.mean(mid_deltas)), 3) if mid_deltas else None,
            'treble_mean': round(float(np.mean(treble_deltas)), 3) if treble_deltas else None,
            'per_song': {song: {k: round(v, 2) for k, v in bands.items()}
                         for song, bands in songs_with_eq.items()},
        }

    # Per-pedal drive
    if drive_slot_entries:
        for pedal in pedal_drive_deltas:
            calibration['per_pedal_drive'][pedal] = {}
            for param, vals in pedal_drive_deltas[pedal].items():
                calibration['per_pedal_drive'][pedal][param] = {
                    'mean': round(float(np.mean(vals)), 3),
                    'n': len(vals),
                    'values': [round(v, 2) for v in vals],
                }

    cal_path = "test_calibration.json"
    with open(cal_path, 'w') as f:
        json.dump(calibration, f, indent=2)

    print(f"\n{'='*70}")
    print(f"💾 Calibration data exported to {cal_path}")
    print(f"   Use this file with the training pipeline to refit coefficients.")
    print(f"{'='*70}")

if __name__ == "__main__": main()
