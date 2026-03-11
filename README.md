# 🎸 Spark AI Tone Engineer

An intelligent audio analysis tool that "listens" to isolated guitar tracks and generates hardware-accurate signal chains for the Positive Grid Spark 2 amplifier.

Unlike the Spark 2's built-in AI (which relies on text prompts), this tool performs **actual DSP analysis** of your audio files to mathematically derive amp settings and effect parameters — then layers in artist gear knowledge from a curated preset library or live API research.

---

## 📂 Project Structure

- **`main.py`** — Core engine: DSP analysis, hybrid rig builder, compensatory EQ, guitar selection, CLI
- **`api_research.py`** — Anthropic API-powered gear research, validation, and preset generation
- **`spark_gear.json`** — Complete Spark 2 hardware database (35 amps, 48 effects with sonic profiles)
- **`artist_presets.json`** — 130 artist presets with era-specific gear mappings and guitar blocks
- **`batch_analyze.py`** — Batch DSP feature extraction across a folder of stems, outputs CSV
- **`separated/`** — Directory for storing isolated guitar stems (FLAC/WAV)
- **`recipes/`** — Auto-generated tone recipes (one file per song)

---

## 🛠️ Installation

### 1. Install Python Dependencies

```bash
pip install librosa numpy scipy
```

**Note:** You also need **FFmpeg** installed and added to your system's PATH for audio loading.

### 2. Install Demucs (Audio Source Separation)

```bash
pip install demucs
```

### 3. Install Anthropic SDK (Optional — enables API research)

```bash
pip install anthropic
```

Set your API key:

```bash
export ANTHROPIC_API_KEY='your-key-here'
```

The API research layer is optional. Without it, the system falls back to DSP-only analysis for songs without artist presets. With it, unknown songs are automatically researched and presets are generated on the fly.

---

## 🚀 Complete Workflow

### Step 1: Source Separation

Place your FLAC or WAV file in the project directory, then run:

```bash
demucs -n htdemucs_6s "Song_Name.flac"
```

This creates **six isolated stems** in `separated/htdemucs_6s/Song_Name/`:
- `vocals.wav`, `drums.wav`, `bass.wav`, **`guitar.wav`** ← use this, `piano.wav`, `other.wav`

**Important:** Use the **6-source model** (`htdemucs_6s`) — it separates guitar from keyboards/piano for cleaner analysis.

### Step 2: Generate Spark 2 Recipe

```bash
python main.py "separated/htdemucs_6s/Song_Name/guitar.wav"
```

What happens:
1. DSP analyzes the stem (gain, mids, air, width)
2. Artist preset check — matches filename keywords against 130 presets
3. If no preset found and `ANTHROPIC_API_KEY` is set → automatic API research
4. Recipe is built, printed, and saved to `recipes/`

---

## 🖥️ CLI Reference

### Standard Usage

```bash
python main.py song.wav                    # Analyze + auto-research if no preset
python main.py song.wav --no-research      # DSP-only, skip API even without preset
```

### Research Commands (no audio file needed)

```bash
python main.py --research "cherub rock smashing pumpkins"   # Research and save preset
python main.py --compare "little wing"                       # Diff API vs existing preset
python main.py --forget "cherub rock"                        # Delete auto-generated preset
python main.py --fill-guitar                                 # Batch-fill missing guitar blocks
```

| Command | What It Does |
|---------|-------------|
| `--research "query"` | Forces API gear lookup, saves preset with `source: auto`, `validated: false` |
| `--compare "query"` | Runs API research and shows side-by-side diff against existing preset. Does not save. |
| `--forget "query"` | Deletes auto-generated presets matching keyword. Won't touch hand-curated presets. |
| `--fill-guitar` | Loops through all presets missing guitar blocks, researches each via API. |
| `--no-research` | Disables automatic API research for the current run. |

---

## 📄 Output Format

**HYBRID MODE** — three source types working together:

```
🎸 SPARK 2 RECIPE (HYBRID MODE): cherubrock.wav
📅 Date: 2026-02-28 00:30:00
🎯 DSP Analysis: Gain 0.94 | Presence 0.22 | Air 0.62 | Mids 0.33 | Width 0.82

Legend: 🔬 = Analysis-Based | 🎨 = Artist Signature | 🤖 = API-Researched
==================================================
[GUITAR] MEXICAN STRAT (SSH) 🎨
   • PICKUP: Bridge Humbucker
   • VOLUME: 10
   • TONE: 10
--------------------------------------------------
[GATE] NOISE GATE 🎨
   • THRESHOLD: 5.0          • DECAY: 1.2
--------------------------------------------------
[DRIVE] GUITAR MUFF 🎨
   • SUSTAIN: 6.5            • TONE: 7.8
   • VOLUME: 5.5
--------------------------------------------------
[AMP] PLEXIGLAS 🎨
   • GAIN: 6.5               • BASS: 5.0
   • MIDDLE: 4.0             • TREBLE: 9.0
   • VOLUME: 8.2
--------------------------------------------------
[MOD/EQ] GUITAR EQ 🔬
   • 100HZ: -0.5             • 200HZ: -0.4
   • 400HZ: -1.7             • 800HZ: -1.0
   • 1600HZ: -3.8            • 3200HZ: +4.7
--------------------------------------------------
[REVERB] PLATE SHORT 🔬
   • LEVEL: 3.3              • TIME: 3.0
   ...
--------------------------------------------------
💾 Recipe saved to: recipes/cherubrock.txt
```

**Source indicators:**
- 🔬 **Analysis-Based** — determined by DSP measurements
- 🎨 **Artist Signature** — from curated preset library
- 🤖 **API-Researched** — auto-generated by Anthropic API lookup

---

## 🧠 How It Works

### Signal Chain (in recipe order)

```
[GUITAR] → [GATE] → [COMP/WAH] → [DRIVE] → [AMP] → [MOD/EQ] → [DELAY] → [REVERB]
```

Each slot is filled by one of three sources: DSP analysis, artist preset, or API research.

### Guitar Selection

Every recipe starts with a `[GUITAR]` block telling you which guitar to pick up, which pickup position, and where to set your volume and tone knobs.

**Two guitars in the system:**
- **American Pro II Strat (SSS)** — default for everything. Five pickup positions mapped to the air measurement (darker tones → neck, brighter → bridge). Treble bleed circuit means volume rollback keeps brightness.
- **Mexican Strat (SSH)** — bridge humbucker only, for songs where the original used humbuckers (Les Paul, SG, ES-335, etc.).

**Selection priority:**
1. Artist preset specifies guitar → use that (🎨)
2. Preset exists but no guitar block + API key available → research guitar (🤖)
3. No preset → DSP determines pickup from air measurement (🔬)

**DSP pickup mapping (SSS):**

| Air Range | Pickup | Character |
|-----------|--------|-----------|
| < 0.35 | Position 5 (Neck) | Dark, warm — jazz, Gilmour |
| 0.35–0.42 | Position 4 (Neck+Mid) | Warm quack — SRV, Mayer |
| 0.42–0.50 | Position 3 (Middle) | Balanced, versatile |
| 0.50–0.56 | Position 2 (Bridge+Mid) | Bright quack — Knopfler |
| > 0.56 | Position 1 (Bridge) | Brightest, aggressive |

### Unified Gain Detection (v4)

Converts audio waveform into a 0.0–1.0 gain value using 5 features in a direct linear regression trained on 24 songs:

```
spark_gain = clip(
    2.1855 × zcr
  - 0.0327 × hp_ratio
  - 0.2666 × kurt
  - 0.1359 × sc_high
  + 1.2657 × rms_cv
  + 2.3766
, 0.0, 1.0)
```

| Feature | What It Measures | Distortion Signal |
|---------|-----------------|-------------------|
| ZCR | Zero-crossing rate | Higher = more distortion harmonics |
| HP Ratio | Harmonic-to-percussive energy | Lower = more sustain/distortion |
| Kurtosis | Amplitude distribution shape | Lower = more compressed |
| SC High | Spectral contrast (upper bands) | Lower = valleys filled by distortion |
| RMS CV | Dynamic range variation | Higher = more dynamics |

**P75 diagnostics:** The system also computes an energy-weighted 75th percentile gain estimate alongside the mean. When the two diverge by more than 0.02, both are logged. This data is building toward a future retrain of the gain model on percentile features.

### Energy-Weighted Tonal Analysis

Tonal features (mids, presence, air) use **energy-weighted loud-section analysis** rather than global means. This solves the multi-section problem — a song like Comfortably Numb (clean verse + screaming solo) no longer averages into a mid-range reading that matches neither section.

**How it works:**
1. Compute per-frame energy from the STFT
2. Filter to frames above median energy (loud mask — top 50% of frames)
3. Weight each loud frame's contribution by its energy
4. The loudest sections dominate the measurement

**Diagnostic output shows the shift:**
```
📊 Tonal: Mids=0.566 | Presence=0.225 | Air=0.476  (Δ from global mean: mids -0.035, air +0.030)
📊 Loud frames: 8232/16464 (50%)
```

Dynamically consistent songs (wall-of-sound, steady clean) show near-zero shift. Multi-section songs shift toward their loudest, most tonally defined passages.

### Compensatory EQ

The 6-band parametric EQ accounts for tone-shaping already done by the drive pedal, preventing double-boosting.

**The problem it solves:** A Big Muff adds bass saturation. The old EQ saw high gain and boosted bass on top of that. Result: boomy, muddy mess. Same issue with treble — bright drive + bright amp + bright EQ = ice pick.

**How it works:** Base EQ formulas determine what the stem needs. Then the drive pedal's contribution is subtracted so the EQ only covers the gap.

```
Without drive:  eq_100 = (gain - 0.5) × 6.0
With Muff:      eq_100 = (gain - 0.5) × 6.0 - 3.15    ← bass_comp from Muff's gain_boost
```

**Drive bass compensation:**

| Drive Character | Factor | Example |
|----------------|--------|---------|
| Bass-cutting (Tube Screamer) | gain_boost × 2.0 | TS (0.5) → comp 1.0 |
| Bass-heavy (Muff, Fuzz Face) | gain_boost × 3.5 | Muff (0.9) → comp 3.15 |
| Default high-gain | gain_boost × 2.5 | RAT (0.8) → comp 2.0 |

**Drive treble compensation:** `gain_boost × 1.5` for all drives. Reduces 3200Hz EQ to prevent stacking with cranked amp TREBLE.

**Mid compensation (preliminary):** Drives with scooped mids (negative `mids` value in sonic_profile) get per-band mid correction: `mid_comp = gain_boost × |negative mids|`. Applied with different weights per band — 400Hz ×1.5, 800Hz ×1.3, 1600Hz ×4.5 — because the scoop affects upper mids far more than lower mids. Calibrated against Cherub Rock (Muff → Plexi). Drives with positive mids (e.g., Tube Screamer mid hump) get zero correction — Eruption's mid bands match without compensation. Pending validation with 2–3 more mid-range drives (Black Op, Fuzz Face, Booster) to confirm linearity.

**Frequency bands:**

| Band | Formula | Description |
|------|---------|-------------|
| 100Hz | `(gain - 0.5) × 6.0 - bass_comp` | Bass weight, reduced by drive |
| 200Hz | `(gain - 0.4) × 4.0 - bass_comp × 0.8` | Upper bass, slightly less reduction |
| 400Hz | `(mids - 0.5) × 6.0 - mid_comp × 1.5` | Low mids, lightest mid correction |
| 800Hz | `(mids - 0.4) × 5.0 - mid_comp × 1.3` | Mid mids, similar correction |
| 1600Hz | `(mids - 0.55) × 8.0 - mid_comp × 4.5` | Upper mids / bite, strongest mid correction |
| 3200Hz | `(air - 0.38) × 25.0 - treble_comp` | Air / Rangemaster zone, reduced by drive |

### Amp Selection & Settings

**Amp TREBLE** is driven by `spark_air` (2400–6800Hz energy ratio, range 0.328–0.618). Previous versions used `spark_presence` (1200–2400Hz) but that band has std=0.021 across 48 stems — no discrimination.

**Amp GAIN** has a floor of 2.0 when a drive pedal is present, preventing the amp from being set to near-zero gain when the drive is handling most of the distortion.

**Amp selection** uses air (not presence) in the distance formula, matching the TREBLE knob driver.

### API Research Layer

When no artist preset matches a song, `api_research.py` queries the Anthropic API to research the artist's actual gear.

**Flow:**
1. Build a prompt containing the full Spark 2 gear inventory (real model names + Spark names)
2. Ask Claude Sonnet for a JSON response: amp, drive, modulation, delay, compressor, reverb, guitar type, pickup position
3. Validate every gear name against `spark_gear.json`:
   - Exact Spark name match
   - Real model name mapping (e.g., "Marshall Plexi" → "Plexiglas")
   - Partial match with disambiguation (prefer Guitar Muff over Bass Muff, prefer standard models over J.H. signature variants)
4. Generate a preset with `"source": "auto"` and `"validated": false`
5. Save to `artist_presets.json` — immediately available for current and future runs

**Validation tested against 23 edge cases:** Marshall Super Lead 100 → Plexiglas (not J.H. Super 100), "Big Muff" → Guitar Muff (not Bass Muff), "Tube Screamer" → Tube Drive, explicit J.H. names still resolve correctly.

**Cost:** ~$0.005 per song lookup. Once researched, the preset is saved permanently — no repeat API calls.

**Safety:** Auto-generated presets never overwrite hand-curated ones. The `--forget` flag only deletes `source: auto` presets.

---

## 📊 Calibration & Accuracy

### Gain Detection — v4 vs v3

| Metric | v3 | v4 | Improvement |
|--------|-----|-----|-------------|
| MAE | 0.150 | 0.094 | 37% better |
| Max Error | 0.500 | 0.188 | 62% better |
| Within ±0.10 | 14/24 | 12/24 | -2 (tradeoff) |
| Within ±0.15 | 16/24 | 20/24 | +4 songs |
| Catastrophic (>0.20) | **7** | **0** | Eliminated |

### Compensatory EQ — Validated Against Amp Tests

| Song | Band | Old | Compensatory | Tested | Improvement |
|------|------|-----|-------------|--------|-------------|
| Cherub Rock | 100Hz | +2.6 | -0.5 | 0.0 | 2.1 dB closer |
| Cherub Rock | 3200Hz | +5.9 | +4.6 | +4.6 | Perfect match |
| Eruption | 100Hz | +1.5 | +0.5 | 0.0 | 1.0 dB closer |
| Eruption | 200Hz | +1.4 | +0.6 | 0.0 | 0.8 dB closer |

### EQ Calibration — Air Band (3200Hz)

| Song | Air Ratio | eq_3200 | Character |
|------|-----------|---------|-----------|
| Gravity | 0.328 | -1.3 dB | Dark/warm clean |
| Every Breath You Take | 0.348 | -0.8 dB | Polished clean |
| Lenny | 0.400 | +0.5 dB | Warm blues |
| Back in Black | 0.469 | +2.2 dB | Mid crunch |
| Eruption | 0.529 | +3.0 dB | Bright shred (with TS comp) |
| Teen Spirit | 0.558 | +4.5 dB | Bright distorted |
| Cherub Rock | 0.618 | +4.6 dB | Searing bright (with Muff comp) |

---

## 🎸 Artist Presets

130 presets covering 100+ guitarists with era-specific gear mappings. All presets include guitar blocks with pickup and knob settings. All gear validated against `spark_gear.json`.

Presets trigger automatically based on filename keywords:

| Filename | Preset | Effects | Guitar |
|----------|--------|---------|--------|
| `cherubrock.wav` | Corgan Siamese Dream | Muff → Plexi (tested settings) | Mexi SSH, Bridge HB |
| `eruption.wav` | EVH Brown Sound | Tube Drive → Plexi (tested settings) | Mexi SSH, Bridge HB |
| `comfortablynumb.wav` | Gilmour Wall Lead | Muff + Flanger + Multi Head | SSS, Pos 5 (Neck) |
| `littlewing.wav` | Hendrix Ballad | Sustain Comp + UniVibe | SSS, Pos 4 (Neck+Mid) |
| `stairwaytoheaven.wav` | Page Live | Booster → Plexi + Echo Tape | Mexi SSH, Bridge HB |
| `unknownsong.wav` | No match → auto API research | Validated against gear DB | API-determined |

---

## 💡 Tips for Best Results

1. **Use FLAC files** — highest fidelity for accurate analysis
2. **Studio versions preferred** — live recordings confuse the analyzer
3. **Always use `htdemucs_6s` model** — separates guitar from keyboards
4. **Listen to the stem first** — verify clean guitar isolation before processing
5. **Include artist/song name in filename** — triggers artist presets automatically
6. **Use recipes as starting points** — fine-tune on actual hardware to taste
7. **Set your API key** — enables automatic gear research for unknown songs
8. **Run `--fill-guitar` after adding new presets** — ensures all presets have guitar blocks

---

## 🛠️ Troubleshooting

**All tones reading as high-gain:** Use `htdemucs_6s`, use `guitar.wav` stem, check for keyboard bleed.

**Artist match not triggering:** Script strips all spaces/symbols. `voodoo_child.wav` matches `"voodoo child"`. Check `artist_presets.json`.

**`Audio Load Error`:** FFmpeg not installed or not in PATH.

**API research not working:** Verify `echo $ANTHROPIC_API_KEY` prints a key starting with `sk-ant-`. Install the SDK with `pip install anthropic`.

**Auto preset seems wrong:** Run `python main.py --compare "song name"` to see what the API suggests vs what exists. Use `--forget` to delete and re-research.

---

## 📐 Version History

**v6 (current)** — Compensatory EQ + Guitar Block + Energy-Weighted Analysis + API Research

- **Compensatory EQ:** Bass, mid, and treble EQ bands now account for drive pedal contribution. Bass-heavy drives (Muff) reduce bass EQ by ~3 dB. All drives reduce treble EQ proportional to gain_boost × 1.5. Scooped drives get per-band mid correction (400Hz ×1.5, 800Hz ×1.3, 1600Hz ×4.5). Validated against Cherub Rock (Muff → Plexi) and Eruption (TS → Plexi) at the amp.
- **Guitar block:** Every recipe includes `[GUITAR]` with guitar choice, pickup position, volume and tone knob settings. Artist presets override DSP when known. All 130 presets have guitar blocks.
- **Energy-weighted tonal analysis:** Mids, presence, and air measurements now weight toward loud frames, solving the multi-section averaging problem. Comfortably Numb's solo now dominates over its clean verse.
- **P75 gain diagnostics:** Energy-weighted 75th percentile gain computed alongside mean for future retraining data.
- **API research layer (`api_research.py`):** Automatic gear research via Anthropic API for unknown songs. Full gear name validation with disambiguation (23/23 edge cases pass). CLI: `--research`, `--compare`, `--forget`, `--fill-guitar`.
- **Amp TREBLE and selection:** Switched from presence to air (2400–6800Hz). Presence had std=0.021 — no discrimination.
- **Amp GAIN floor:** Minimum 2.0 when drive pedal present.
- **Drive profile fixes:** Black Op (ProCo RAT) mids corrected to -0.2. J.H. Octave Fuzz gain_boost corrected to 0.65.
- **CLI overhaul:** argparse-based with flags for research, compare, forget, fill-guitar, no-research.
- **Artist presets expanded to 130** with guitar blocks, tested amp settings for Cherub Rock and Eruption.

**v5** — Calibrated dual-band EQ + bug fixes
- Replaced single spectral centroid with `spark_presence` (1200-2400Hz) and `spark_air` (2400-6800Hz)
- 3200Hz driven by air: `(air - 0.38) * 25.0`
- Fixed amp TREBLE, selection formula, reverb DAMPING, stale variable references

**v4** — Unified 5-feature gain detection (no tiers, no gates)
- Eliminated ALL catastrophic errors (7 → 0), MAE improved 37%

**v3** — Three-tier gain detection with flatness regression
- Fixed 49 broken effect parameter names, 16 phantom amp references

**v2** — Two-tier ceiling system with clean-zone detection

**v0** — Original crest factor + spectral flatness

---

## 🔮 Roadmap

**Next (needs testing data):**
- Mid-band compensatory EQ validation — preliminary implementation in place, needs 2–3 more drives tested at amp to confirm per-band factors
- Gain model retrain on P75 features — collecting diagnostic data now
- Harmonic series analysis calibration — even/odd ratio and rolloff implemented, needs amp testing to map ratios to drive/amp type selection

**Future:**
- Non-linear gain model (polynomial or random forest)
- Segment detection — verse vs chorus vs solo, multiple recipes per track
- More training data (48+ songs)

---

## 🎸 Ready to Analyze?

```bash
# 1. Separate your song
demucs -n htdemucs_6s "Your_Song.flac"

# 2. Analyze the guitar (auto-researches unknown songs)
python main.py "separated/htdemucs_6s/Your Song/guitar.wav"

# 3. Or pre-research before you even have stems
python main.py --research "bohemian rhapsody queen"

# 4. Check recipes/ folder for your saved recipe!
```

**Happy tone hunting!** 🎶

---

## Acknowledgments

BPM data provided by [GetSongBPM.com](https://getsongbpm.com)
