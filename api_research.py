#!/usr/bin/env python3
"""
api_research.py — Anthropic API-powered gear research for Spark AI Tone Engineer.

When no artist preset matches a song, this module queries the Anthropic API
to research the artist's actual gear, validates every result against
spark_gear.json, and saves new presets for future use.

Requires: ANTHROPIC_API_KEY environment variable
Install:  pip install anthropic

Usage (standalone):
    python api_research.py --research "cherub rock"
    python api_research.py --compare "little wing"
    python api_research.py --forget "cherub rock"
    python api_research.py --fill-guitar

Usage (from main.py):
    import api_research
    preset = api_research.research_song("cherub rock", db)
"""

import os
import sys
import json
import re

# Load .env file if present (avoids needing `export` every session)
def _load_dotenv():
    """Load key=value pairs from .env file in the script's directory."""
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
# GEAR MAPPING: Real Model Names → Spark Names
# ==========================================

def build_gear_maps(db):
    """Build bidirectional mappings between real model names and Spark names."""
    real_to_spark = {}  # real_model → [list of spark names]
    spark_to_real = {}
    all_spark_names = {}  # category → [names]

    # Amps
    all_spark_names['amp'] = []
    for cat in db['amps']:
        for m in cat['models']:
            real = m.get('real_model', '').lower()
            spark = m['name']
            if real:
                real_to_spark.setdefault(real, []).append(spark)
            spark_to_real[spark.lower()] = real
            all_spark_names['amp'].append(spark)

    # All other categories
    category_map = {
        'drive': 'drive',
        'modulation': 'modulation',
        'delay': 'delay',
        'compressor': 'compressor',
        'reverb': 'reverb',
    }

    for cat_key, db_key in category_map.items():
        all_spark_names[cat_key] = []
        for item in db[db_key]:
            real = item.get('real_model', '').lower()
            spark = item['name']
            if real:
                real_to_spark.setdefault(real, []).append(spark)
            spark_to_real[spark.lower()] = real
            all_spark_names[cat_key].append(spark)

    # Also add EQ to modulation list (it shares the slot)
    if 'eq' in db:
        for item in db['eq']:
            all_spark_names.setdefault('modulation', []).append(item['name'])

    return real_to_spark, spark_to_real, all_spark_names


# ==========================================
# GENRE DETECTION & AMP HINTS
# ==========================================
# The API sometimes picks wildly inappropriate amps when it lacks genre context.
# Metallica's "One" getting matched to Silver 120 (JC-120 jazz chorus) is a
# prime example. This module detects likely genre from song/artist names and
# adds hints to steer the API toward appropriate amp categories.

# Genre → (amp categories to prefer, amp categories to avoid)
GENRE_AMP_HINTS = {
    'metal': (['Metal', 'High Gain'], ['Clean', 'Glassy']),
    'thrash': (['Metal', 'High Gain'], ['Clean', 'Glassy']),
    'death_metal': (['Metal'], ['Clean', 'Glassy', 'Crunch']),
    'djent': (['Metal', 'High Gain'], ['Clean', 'Glassy']),
    'punk': (['Crunch', 'High Gain'], ['Clean']),
    'hardcore': (['High Gain', 'Metal'], ['Clean']),
    'hard_rock': (['High Gain', 'Crunch'], ['Clean']),
    'blues': (['Crunch', 'Glassy'], ['Metal']),
    'blues_rock': (['Crunch', 'High Gain'], ['Metal']),
    'jazz': (['Clean', 'Glassy'], ['Metal', 'High Gain']),
    'fusion': (['Clean', 'High Gain'], ['Metal']),
    'progressive': (['High Gain', 'Clean', 'Crunch'], []),
    'classic_rock': (['Crunch', 'High Gain', 'Glassy'], ['Metal']),
    'hendrix': (['Jimi Hendrix', 'Crunch', 'Glassy'], ['Metal', 'Clean']),
    'grunge': (['High Gain', 'Crunch'], ['Clean']),
    'alternative': (['Crunch', 'High Gain'], []),
    'indie': (['Glassy', 'Clean', 'Crunch'], ['Metal']),
    'country': (['Clean', 'Glassy', 'Crunch'], ['Metal', 'High Gain']),
}

# Artist/band → genre mapping (known heavy artists that might get misclassified)
ARTIST_GENRE_MAP = {
    # Metal/Thrash
    'metallica': 'metal',
    'megadeth': 'thrash',
    'slayer': 'thrash',
    'pantera': 'metal',
    'sepultura': 'thrash',
    'anthrax': 'thrash',
    'iron maiden': 'metal',
    'judas priest': 'metal',
    'black sabbath': 'metal',
    'ozzy': 'metal',
    'dio': 'metal',
    'motorhead': 'metal',
    'lamb of god': 'metal',
    'mastodon': 'metal',
    'gojira': 'metal',
    'meshuggah': 'djent',
    'periphery': 'djent',
    'tool': 'progressive',
    'dream theater': 'progressive',
    'rush': 'progressive',
    'yes': 'progressive',
    
    # Hard Rock
    'van halen': 'hard_rock',
    'van_halen': 'hard_rock',
    'evh': 'hard_rock',
    'eddie van halen': 'hard_rock',
    'ac/dc': 'hard_rock',
    'acdc': 'hard_rock',
    'guns n roses': 'hard_rock',
    'gnr': 'hard_rock',
    'aerosmith': 'hard_rock',
    'def leppard': 'hard_rock',
    'kiss': 'hard_rock',
    'motley crue': 'hard_rock',
    
    # Grunge/Alternative
    'nirvana': 'grunge',
    'pearl jam': 'grunge',
    'soundgarden': 'grunge',
    'alice in chains': 'grunge',
    'stone temple pilots': 'grunge',
    'smashing pumpkins': 'alternative',
    'radiohead': 'alternative',
    'foo fighters': 'alternative',
    'king gizzard': 'alternative',
    'gizzard': 'alternative',
    'kinggizzard': 'alternative',
    'rage against the machine': 'hard_rock',
    'ratm': 'hard_rock',
    'morello': 'hard_rock',
    'killinginthename': 'hard_rock',
    
    # Blues
    'srv': 'blues_rock',
    'stevie ray vaughan': 'blues_rock',
    'bb king': 'blues',
    'buddy guy': 'blues',
    'john mayer': 'blues_rock',
    'john_mayer': 'blues_rock',
    'mayer': 'blues_rock',
    'gary moore': 'blues_rock',
    'joe bonamassa': 'blues_rock',
    
    # Classic Rock / Progressive
    'led zeppelin': 'classic_rock',
    'zeppelin': 'classic_rock',
    'pink floyd': 'progressive',
    'pink_floyd': 'progressive',
    'floyd': 'progressive',
    'gilmour': 'progressive',
    'deep purple': 'classic_rock',
    'the who': 'classic_rock',
    'cream': 'blues_rock',
    'clapton': 'blues_rock',
    'eric clapton': 'blues_rock',
    'derek': 'blues_rock',
    'dominos': 'blues_rock',
    'derek and the dominos': 'blues_rock',
    
    # Hendrix
    'hendrix': 'hendrix',
    'jimi hendrix': 'hendrix',
    'jimi': 'hendrix',
    
    # Punk
    'ramones': 'punk',
    'sex pistols': 'punk',
    'the clash': 'punk',
    'green day': 'punk',
    'blink': 'punk',
}


def detect_genre(song_name):
    """
    Detect likely genre from song/artist name.
    
    Returns genre string or None if no clear genre detected.
    """
    name_lower = song_name.lower()
    
    # Check for artist matches first
    for artist, genre in ARTIST_GENRE_MAP.items():
        if artist in name_lower:
            return genre
    
    # Check for genre keywords in the filename
    genre_keywords = {
        'metal': ['metal', 'thrash', 'death', 'slayer', 'metallica'],
        'punk': ['punk', 'hardcore'],
        'blues': ['blues'],
        'jazz': ['jazz'],
        'grunge': ['grunge'],
        'progressive': ['prog'],
    }
    
    for genre, keywords in genre_keywords.items():
        for kw in keywords:
            if kw in name_lower:
                return genre
    
    return None


def build_genre_hint(genre):
    """
    Build a prompt hint string for a detected genre.
    
    Returns a string to insert into the API prompt, or empty string if no genre.
    """
    if not genre or genre not in GENRE_AMP_HINTS:
        return ""
    
    prefer, avoid = GENRE_AMP_HINTS[genre]
    
    lines = ["\nGENRE GUIDANCE:"]
    lines.append(f"This appears to be a {genre.replace('_', ' ')} track.")
    
    if prefer:
        lines.append(f"Amp categories that typically fit this genre: {', '.join(prefer)}")
    if avoid:
        lines.append(f"Amp categories to generally avoid for this genre: {', '.join(avoid)}")
    
    lines.append("Use this as guidance, but override if you know the actual gear used was different.")
    
    return "\n".join(lines)


def validate_gear_name(name, category, db, real_to_spark, all_spark_names):
    """
    Validate a gear name against spark_gear.json.
    Tries exact match first, then real_model mapping, then fuzzy.
    When multiple Spark models share a real_model (e.g. Plexiglas and J.H. Super 100),
    prefers the non-J.H. variant (J.H. models are Hendrix signature editions).
    Returns (spark_name, matched) or (None, False).
    """
    if not name:
        return None, False

    name_lower = name.lower().strip()

    # 1. Exact Spark name match
    for valid_name in all_spark_names.get(category, []):
        if valid_name.lower() == name_lower:
            return valid_name, True

    # Helper: when multiple Spark names match, prefer non-J.H. variant
    def pick_best(spark_names, category):
        valid = [s for s in spark_names if s in all_spark_names.get(category, [])]
        if not valid:
            return None
        # Prefer non-J.H. variants (they're the "standard" models)
        non_jh = [s for s in valid if not s.startswith('J.H.')]
        return non_jh[0] if non_jh else valid[0]

    # 2. Exact real model name match
    if name_lower in real_to_spark:
        best = pick_best(real_to_spark[name_lower], category)
        if best:
            return best, True

    # 3. Partial match on real model names (API might say "Tube Screamer" not "Ibanez Tube Screamer")
    partial_matches = []
    for real_model, spark_names in real_to_spark.items():
        if name_lower in real_model or real_model in name_lower:
            best = pick_best(spark_names, category)
            if best:
                # Score: prefer shorter real_model (more specific match),
                # penalize Bass variants (Guitar Muff > Bass Muff for "Big Muff")
                bass_penalty = 100 if 'bass' in best.lower() else 0
                partial_matches.append((best, len(real_model) + bass_penalty))

    if partial_matches:
        # Sort by score (lower = better match), then prefer non-J.H.
        partial_matches.sort(key=lambda x: x[1])
        return partial_matches[0][0], True

    # 4. Partial match on Spark names
    for valid_name in all_spark_names.get(category, []):
        if name_lower in valid_name.lower() or valid_name.lower() in name_lower:
            return valid_name, True

    return None, False


# ==========================================
# API PROMPT CONSTRUCTION
# ==========================================

def build_gear_reference(db):
    """Build the gear reference section for the API prompt."""
    lines = []

    lines.append("AVAILABLE AMPS (Spark Name → Real Model):")
    for cat in db['amps']:
        for m in cat['models']:
            bt = m['base_tone']
            lines.append(f"  {m['name']:<25} ({m.get('real_model', '?')}) — gain={bt['gain']:.2f} treble={bt['treble']:.2f} mids={bt['mids']:.2f}")

    def format_params(item):
        """Format parameters as name(range) list, handling all param types."""
        params = item.get('parameters', {})
        parts = []
        for k, v in params.items():
            if isinstance(v, dict):
                if 'min' in v:
                    # Standard continuous knob
                    parts.append(f"{k}({v['min']}-{v['max']})")
                elif 'bpm_on' in v:
                    # Dual-mode param (changes behavior with BPM switch)
                    bpm_off = v.get('bpm_off', {})
                    if 'min' in bpm_off:
                        parts.append(f"{k}({bpm_off['min']}-{bpm_off['max']}{bpm_off.get('unit','')}, or subdivision with BPM)")
                    else:
                        parts.append(f"{k}(subdivision with BPM)")
                elif v.get('type') == 'switch':
                    parts.append(f"{k}({'/'.join(v.get('values', []))})")
                elif v.get('type') == 'selector':
                    vals = v.get('values', [])
                    if len(vals) <= 4:
                        parts.append(f"{k}({'/'.join(vals)})")
                    else:
                        parts.append(f"{k}({vals[0]}...{vals[-1]})")
                elif v.get('type') == 'fixed':
                    parts.append(f"{k}(fixed: {v.get('value', '?')})")
        return ', '.join(parts) if parts else 'no adjustable params'

    lines.append("\nAVAILABLE DRIVE PEDALS (with parameters):")
    for d in db['drive']:
        lines.append(f"  {d['name']:<25} ({d.get('real_model', '?')}) — {format_params(d)}")

    lines.append("\nAVAILABLE MODULATION (with parameters):")
    for m in db['modulation']:
        lines.append(f"  {m['name']:<25} ({m.get('real_model', '?')}) — {format_params(m)}")

    lines.append("\nAVAILABLE DELAYS (with parameters):")
    for d in db['delay']:
        lines.append(f"  {d['name']:<25} ({d.get('real_model', '?')}) — {format_params(d)}")

    lines.append("\nAVAILABLE COMPRESSORS (with parameters):")
    for c in db['compressor']:
        lines.append(f"  {c['name']:<25} ({c.get('real_model', '?')}) — {format_params(c)}")

    lines.append("\nAVAILABLE REVERBS:")
    for r in db['reverb']:
        lines.append(f"  {r['name']:<25} ({r.get('real_model', '?')})")

    return "\n".join(lines)


def build_research_prompt(song_name, db):
    """Build the API prompt for gear research."""
    gear_ref = build_gear_reference(db)
    
    # Detect genre and build hint
    genre = detect_genre(song_name)
    genre_hint = build_genre_hint(genre)

    prompt = f"""You are a guitar tone expert. I need to recreate the guitar tone from a song using a Positive Grid Spark 2 amplifier. The Spark 2 is a digital modelling amp with specific gear models.

SONG/ARTIST: {song_name}

Research what gear was used on this recording and map it to the closest available Spark 2 equivalents. Use your knowledge of the artist's known gear, recording session details, and era-appropriate equipment.

{gear_ref}
{genre_hint}

Respond with ONLY a JSON object (no markdown, no backticks, no explanation) in this exact format:
{{
    "artist": "Artist Name",
    "song": "Song Name",
    "era": "Album/Year",
    "description": "Short description of the tone and signal chain",
    "guitar_type": "humbucker" or "single_coil",
    "pickup_position": "bridge", "neck", "neck_mid", "middle", "bridge_mid",
    "guitar_volume": 1-10,
    "guitar_tone": 1-10,
    "amp": "Spark Name from the list above",
    "amp_notes": "Why this amp was chosen",
    "drive": "Spark Name or null if no drive pedal",
    "drive_settings": {{"param_name": value, ...}} or null,
    "drive_notes": "Why this drive, or null",
    "modulation": "Spark Name or null",
    "mod_settings": {{"param_name": value, ...}} or null,
    "mod_notes": "Why this modulation, or null",
    "delay": "Spark Name or null",
    "delay_settings": {{"param_name": value, ...}} or null,
    "delay_notes": "Why this delay, or null",
    "compressor": "Spark Name or null",
    "comp_settings": {{"param_name": value, ...}} or null,
    "comp_notes": "Why this compressor, or null",
    "reverb": "Spark Name or null",
    "reverb_notes": "Why this reverb, or null",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "confidence": "high", "medium", or "low"
}}

IMPORTANT RULES:
- Use ONLY gear names from the lists above. Do not invent names.
- Use the SPARK NAME (left column), not the real model name.
- If the artist's actual gear isn't available, pick the closest Spark 2 equivalent.
- keywords should be lowercase strings that would appear in a filename — song name words, artist name words, album name words.
- For guitar_type: "humbucker" if Les Paul, SG, ES-335, HH guitars; "single_coil" if Strat, Tele, or SSS/SS guitars.
- Be specific about pickup position based on the actual song's tone.
- For settings objects: use the EXACT parameter names shown in parentheses for each effect. Values must be within the min-max range shown. If unsure about specific settings, use null for the settings object and the system will generate defaults.
- Respond with ONLY the JSON object, nothing else."""

    return prompt


def build_guitar_prompt(song_name, artist, db):
    """Build the API prompt for guitar-only research."""
    prompt = f"""You are a guitar tone expert. I need to know what guitar, pickup position, and knob settings were used on a specific recording.

ARTIST: {artist}
SONG/CONTEXT: {song_name}

I have two guitars available:
1. Fender American Professional II Stratocaster (SSS - three single coils)
   - 5 pickup positions: Position 1 (Bridge), Position 2 (Bridge+Mid), Position 3 (Middle), Position 4 (Neck+Mid), Position 5 (Neck)
   - Has treble bleed circuit (stays bright when volume is rolled back)
2. Fender Mexican Stratocaster (SSH - two single coils + bridge humbucker)
   - Used ONLY for the bridge humbucker position

Respond with ONLY a JSON object (no markdown, no backticks):
{{
    "guitar": "American Pro II Strat (SSS)" or "Mexican Strat (SSH)",
    "pickup": "Position X (Name)" or "Bridge Humbucker",
    "volume": 1-10,
    "tone": 1-10,
    "reasoning": "Brief explanation"
}}

Rules:
- Mexican Strat ONLY for songs where the original used humbuckers (Les Paul, SG, ES-335, etc.)
- American SSS for everything else (Strat, Tele, Jazzmaster, Jaguar, etc.)
- volume: 10 unless the artist is known for riding the volume knob (e.g., SRV at 7-8)
- tone: based on the brightness of the recording. Full open (10) for bright, rolled back for warm."""

    return prompt


# ==========================================
# API COMMUNICATION
# ==========================================

def call_api(prompt):
    """Call the Anthropic API. Returns response text or None."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("   ❌ ANTHROPIC_API_KEY not set. Set it with:")
        print("      export ANTHROPIC_API_KEY='your-key-here'")
        return None

    try:
        import anthropic
    except ImportError:
        print("   ❌ anthropic package not installed. Run:")
        print("      pip install anthropic")
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"   ❌ API Error: {e}")
        return None


def parse_api_response(response_text):
    """Parse JSON from API response, handling common formatting issues."""
    if not response_text:
        return None

    # Strip markdown code fences if present
    text = response_text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"   ❌ Failed to parse API response: {e}")
        print(f"   Raw response: {text[:200]}...")
        return None


# ==========================================
# PRESET GENERATION FROM API RESPONSE
# ==========================================

def validate_settings(settings_dict, gear_name, db_category, db):
    """
    Validate effect settings against spark_gear.json parameter ranges.
    Returns validated settings dict with values clamped to valid ranges,
    or None if settings_dict is None/empty.
    """
    if not settings_dict or not isinstance(settings_dict, dict):
        return None

    # Find the gear item in the database
    item = None
    if db_category == 'drive':
        item = next((d for d in db['drive'] if d['name'] == gear_name), None)
    elif db_category == 'modulation':
        item = next((m for m in db['modulation'] if m['name'] == gear_name), None)
        if not item and 'eq' in db:
            item = next((e for e in db['eq'] if e['name'] == gear_name), None)
    elif db_category == 'delay':
        item = next((d for d in db['delay'] if d['name'] == gear_name), None)
    elif db_category == 'compressor':
        item = next((c for c in db['compressor'] if c['name'] == gear_name), None)

    if not item:
        return None

    params = item.get('parameters', {})
    validated = {}

    for key, value in settings_dict.items():
        # Normalize key to match parameter names (case-insensitive)
        key_lower = key.lower()
        matched_param = None
        for param_name in params:
            if param_name.lower() == key_lower:
                matched_param = param_name
                break

        if not matched_param or not isinstance(params[matched_param], dict):
            continue

        param_def = params[matched_param]

        if 'min' in param_def:
            # Standard continuous knob — clamp to range
            try:
                val = float(value)
                validated[matched_param.upper()] = round(min(param_def['max'], max(param_def['min'], val)), 1)
            except (ValueError, TypeError):
                pass

        elif 'bpm_off' in param_def:
            # Dual-mode param (changes behavior with BPM switch).
            # Accept numeric values (validate against bpm_off range)
            # or string subdivision values (validate against bpm_on values).
            if isinstance(value, str):
                bpm_on = param_def.get('bpm_on', {})
                if bpm_on.get('type') == 'fixed':
                    # Fixed BPM mode — only one valid value
                    if value == str(bpm_on.get('value', '')):
                        validated[matched_param.upper()] = value
                else:
                    valid_values = bpm_on.get('values', [])
                    if value in valid_values:
                        validated[matched_param.upper()] = value
            else:
                try:
                    val = float(value)
                    bpm_off = param_def['bpm_off']
                    if 'min' in bpm_off:
                        validated[matched_param.upper()] = round(min(bpm_off['max'], max(bpm_off['min'], val)), 1)
                except (ValueError, TypeError):
                    pass

        elif param_def.get('type') in ('switch', 'selector'):
            # Switch or selector — validate against allowed values
            str_val = str(value)
            valid_values = param_def.get('values', [])
            if str_val in valid_values:
                validated[matched_param.upper()] = str_val

        elif param_def.get('type') == 'fixed':
            # Fixed value — only accept the defined value
            if str(value) == str(param_def.get('value', '')):
                validated[matched_param.upper()] = str(value)

    return validated if validated else None


def build_preset_from_research(research, db):
    """
    Convert API research response into a validated artist preset.
    Returns (preset_dict, validation_report) or (None, errors).
    """
    real_to_spark, spark_to_real, all_spark_names = build_gear_maps(db)

    preset = {
        "keywords": research.get("keywords", []),
        "description": research.get("description", "API-researched preset"),
        "source": "auto",
        "validated": False,
        "confidence": research.get("confidence", "unknown"),
        "forced_gear": {}
    }

    report = []
    forced = preset["forced_gear"]

    # --- Amp (required) ---
    amp_name = research.get("amp")
    if amp_name:
        validated, matched = validate_gear_name(amp_name, "amp", db, real_to_spark, all_spark_names)
        if matched:
            forced["amp"] = validated
            report.append(f"  ✅ Amp: {validated} (from '{amp_name}')")
        else:
            report.append(f"  ❌ Amp: '{amp_name}' not found in Spark 2")
    else:
        report.append("  ⚠️ Amp: not specified by API (DSP will choose)")

    # --- Drive ---
    drive_name = research.get("drive")
    if drive_name and drive_name != "null":
        validated, matched = validate_gear_name(drive_name, "drive", db, real_to_spark, all_spark_names)
        if matched:
            forced["drive"] = validated
            report.append(f"  ✅ Drive: {validated} (from '{drive_name}')")
            # Extract and validate settings if provided
            drive_settings = validate_settings(research.get("drive_settings"), validated, "drive", db)
            if drive_settings:
                forced["drive_settings"] = drive_settings
                report.append(f"     ⚙️ Drive settings: {drive_settings}")
        else:
            report.append(f"  ❌ Drive: '{drive_name}' not found in Spark 2")

    # --- Modulation ---
    mod_name = research.get("modulation")
    if mod_name and mod_name != "null":
        validated, matched = validate_gear_name(mod_name, "modulation", db, real_to_spark, all_spark_names)
        if matched:
            forced["mod_eq"] = validated
            report.append(f"  ✅ Modulation: {validated} (from '{mod_name}')")
            mod_settings = validate_settings(research.get("mod_settings"), validated, "modulation", db)
            if mod_settings:
                forced["mod_settings"] = mod_settings
                report.append(f"     ⚙️ Mod settings: {mod_settings}")
        else:
            report.append(f"  ❌ Modulation: '{mod_name}' not found in Spark 2")

    # --- Delay ---
    delay_name = research.get("delay")
    if delay_name and delay_name != "null":
        validated, matched = validate_gear_name(delay_name, "delay", db, real_to_spark, all_spark_names)
        if matched:
            forced["delay"] = validated
            report.append(f"  ✅ Delay: {validated} (from '{delay_name}')")
            delay_settings = validate_settings(research.get("delay_settings"), validated, "delay", db)
            if delay_settings:
                forced["delay_settings"] = delay_settings
                report.append(f"     ⚙️ Delay settings: {delay_settings}")
        else:
            report.append(f"  ❌ Delay: '{delay_name}' not found in Spark 2")

    # --- Compressor ---
    comp_name = research.get("compressor")
    if comp_name and comp_name != "null":
        validated, matched = validate_gear_name(comp_name, "compressor", db, real_to_spark, all_spark_names)
        if matched:
            forced["comp_wah"] = validated
            report.append(f"  ✅ Compressor: {validated} (from '{comp_name}')")
            comp_settings = validate_settings(research.get("comp_settings"), validated, "compressor", db)
            if comp_settings:
                forced["comp_settings"] = comp_settings
                report.append(f"     ⚙️ Comp settings: {comp_settings}")
        else:
            report.append(f"  ❌ Compressor: '{comp_name}' not found in Spark 2")

    # --- Reverb ---
    reverb_name = research.get("reverb")
    if reverb_name and reverb_name != "null":
        validated, matched = validate_gear_name(reverb_name, "reverb", db, real_to_spark, all_spark_names)
        if matched:
            forced["reverb"] = validated
            report.append(f"  ✅ Reverb: {validated} (from '{reverb_name}')")
        else:
            report.append(f"  ❌ Reverb: '{reverb_name}' not found in Spark 2")

    # --- Guitar ---
    guitar_type = research.get("guitar_type", "single_coil")
    pickup = research.get("pickup_position", "neck_mid")
    vol = research.get("guitar_volume", 10)
    tone = research.get("guitar_tone", 8)

    pickup_map = {
        "bridge": ("Mexican Strat (SSH)", "Bridge Humbucker") if guitar_type == "humbucker" else ("American Pro II Strat (SSS)", "Position 1 (Bridge)"),
        "bridge_mid": ("American Pro II Strat (SSS)", "Position 2 (Bridge+Mid)"),
        "middle": ("American Pro II Strat (SSS)", "Position 3 (Middle)"),
        "neck_mid": ("American Pro II Strat (SSS)", "Position 4 (Neck+Mid)"),
        "neck": ("American Pro II Strat (SSS)", "Position 5 (Neck)"),
    }

    if guitar_type == "humbucker":
        guitar_name = "Mexican Strat (SSH)"
        pickup_name = "Bridge Humbucker"
    else:
        guitar_name, pickup_name = pickup_map.get(pickup, ("American Pro II Strat (SSS)", "Position 4 (Neck+Mid)"))

    forced["guitar"] = {
        "guitar": guitar_name,
        "pickup": pickup_name,
        "volume": int(min(10, max(1, vol))),
        "tone": int(min(10, max(1, tone)))
    }
    report.append(f"  🎸 Guitar: {guitar_name}, {pickup_name}, V={vol} T={tone}")

    return preset, report


def save_preset(preset, song_name, presets_path="artist_presets.json"):
    """Save a new auto-generated preset to artist_presets.json."""
    try:
        with open(presets_path, 'r') as f:
            presets = json.load(f)
    except:
        presets = {}

    # Generate a key from the song name
    key = re.sub(r'[^a-z0-9]+', '_', song_name.lower()).strip('_')
    key = f"auto_{key}"

    # Don't overwrite hand-curated presets
    if key.replace("auto_", "") in presets:
        existing = presets[key.replace("auto_", "")]
        if existing.get("source") != "auto":
            print(f"   ⚠️ Hand-curated preset exists for this keyword. Use --compare instead.")
            return None

    presets[key] = preset

    with open(presets_path, 'w') as f:
        json.dump(presets, f, indent=2)

    return key


def forget_preset(keyword, presets_path="artist_presets.json"):
    """Delete an auto-generated preset matching a keyword."""
    try:
        with open(presets_path, 'r') as f:
            presets = json.load(f)
    except:
        print("   ❌ Could not load artist_presets.json")
        return False

    keyword_clean = re.sub(r'[^a-z0-9]', '', keyword.lower())
    deleted = []

    for key in list(presets.keys()):
        data = presets[key]
        if data.get("source") != "auto":
            continue
        for kw in data.get("keywords", []):
            kw_clean = re.sub(r'[^a-z0-9]', '', kw.lower())
            if keyword_clean in kw_clean or kw_clean in keyword_clean:
                deleted.append(key)
                del presets[key]
                break

    if deleted:
        with open(presets_path, 'w') as f:
            json.dump(presets, f, indent=2)
        print(f"   🗑️ Deleted {len(deleted)} auto preset(s): {', '.join(deleted)}")
        return True
    else:
        print(f"   ⚠️ No auto-generated presets found matching '{keyword}'")
        return False


def forget_all_auto(presets_path="artist_presets.json"):
    """Delete ALL auto-generated presets. Hand-curated presets are never touched."""
    try:
        with open(presets_path, 'r') as f:
            presets = json.load(f)
    except:
        print("   ❌ Could not load artist_presets.json")
        return False

    auto_keys = [k for k, v in presets.items() if v.get("source") == "auto"]

    if not auto_keys:
        print("   ⚠️ No auto-generated presets found.")
        return False

    for key in auto_keys:
        del presets[key]

    with open(presets_path, 'w') as f:
        json.dump(presets, f, indent=2)

    hand_curated = len(presets)
    print(f"   🗑️ Deleted {len(auto_keys)} auto-generated presets.")
    print(f"   ✅ {hand_curated} hand-curated presets preserved.")
    return True


# ==========================================
# HIGH-LEVEL RESEARCH FUNCTIONS
# ==========================================

def research_song(song_name, db, save=True, presets_path="artist_presets.json"):
    """
    Full gear research for a song. Calls API, validates, optionally saves.
    Returns (preset, report) or (None, None) on failure.
    """
    print(f"   🤖 Researching gear for: {song_name}")

    prompt = build_research_prompt(song_name, db)
    response_text = call_api(prompt)
    if not response_text:
        return None, None

    research = parse_api_response(response_text)
    if not research:
        return None, None

    preset, report = build_preset_from_research(research, db)

    print(f"   🤖 API Research Results ({research.get('confidence', '?')} confidence):")
    print(f"   📝 {research.get('description', 'No description')}")
    for line in report:
        print(f"   {line}")

    if save:
        key = save_preset(preset, song_name, presets_path)
        if key:
            print(f"   💾 Saved as preset '{key}' (validated=false)")

    return preset, report


def research_guitar(song_name, artist, db):
    """
    Guitar-only research for filling missing guitar blocks.
    Returns guitar_info dict or None.
    """
    print(f"   🤖 Researching guitar for: {artist} — {song_name}")

    prompt = build_guitar_prompt(song_name, artist, db)
    response_text = call_api(prompt)
    if not response_text:
        return None

    result = parse_api_response(response_text)
    if not result:
        return None

    guitar_info = {
        "guitar": result.get("guitar", "American Pro II Strat (SSS)"),
        "pickup": result.get("pickup", "Position 3 (Middle)"),
        "volume": int(min(10, max(1, result.get("volume", 10)))),
        "tone": int(min(10, max(1, result.get("tone", 8))))
    }

    print(f"   🎸 {guitar_info['guitar']}, {guitar_info['pickup']}, V={guitar_info['volume']} T={guitar_info['tone']}")
    if result.get("reasoning"):
        print(f"   📝 {result['reasoning']}")

    return guitar_info


def compare_preset(song_name, db, presets_path="artist_presets.json"):
    """
    Research a song and compare results against existing preset.
    Does NOT overwrite.
    """
    # Load existing presets
    try:
        with open(presets_path, 'r') as f:
            presets = json.load(f)
    except:
        presets = {}

    # Find matching preset
    song_clean = re.sub(r'[^a-z0-9]', '', song_name.lower())
    existing_key = None
    existing_preset = None
    for key, data in presets.items():
        for kw in data.get('keywords', []):
            kw_clean = re.sub(r'[^a-z0-9]', '', kw.lower())
            if kw_clean in song_clean or song_clean in kw_clean:
                existing_key = key
                existing_preset = data
                break
        if existing_key:
            break

    # Run research (don't save)
    preset, report = research_song(song_name, db, save=False)
    if not preset:
        return

    # Compare
    print(f"\n   📊 COMPARISON:")
    if existing_preset:
        print(f"   Existing preset: {existing_key}")
        existing_gear = existing_preset.get('forced_gear', {})
        new_gear = preset.get('forced_gear', {})

        for slot in ['amp', 'drive', 'mod_eq', 'comp_wah', 'delay', 'reverb']:
            old_val = existing_gear.get(slot, 'none')
            new_val = new_gear.get(slot, 'none')
            if isinstance(old_val, dict):
                old_val = 'configured'
            if isinstance(new_val, dict):
                new_val = 'configured'
            match = "✓" if old_val == new_val else "≠"
            print(f"   {match} {slot:<12} existing={old_val:<25} api={new_val}")

        # Guitar comparison
        old_guitar = existing_gear.get('guitar', {})
        new_guitar = new_gear.get('guitar', {})
        if old_guitar or new_guitar:
            print(f"\n   Guitar:")
            for field in ['guitar', 'pickup', 'volume', 'tone']:
                ov = old_guitar.get(field, 'none')
                nv = new_guitar.get(field, 'none')
                match = "✓" if str(ov) == str(nv) else "≠"
                print(f"   {match} {field:<12} existing={str(ov):<30} api={str(nv)}")
    else:
        print(f"   No existing preset found. This would create a new one.")


def fill_guitar_blocks(db, presets_path="artist_presets.json"):
    """
    Batch-fill missing guitar blocks in existing presets via API research.
    """
    try:
        with open(presets_path, 'r') as f:
            presets = json.load(f)
    except:
        print("   ❌ Could not load artist_presets.json")
        return

    missing = []
    for key, data in presets.items():
        if 'guitar' not in data.get('forced_gear', {}):
            missing.append(key)

    print(f"   📊 {len(missing)} presets missing guitar blocks")

    if not missing:
        return

    filled = 0
    for key in missing:
        data = presets[key]
        desc = data.get('description', key)
        # Extract artist name from description or key
        artist = key.split('_')[0].title()
        song_hint = ', '.join(data.get('keywords', [])[:3])

        guitar_info = research_guitar(f"{song_hint}", artist, db)
        if guitar_info:
            presets[key]['forced_gear']['guitar'] = guitar_info
            filled += 1
            print(f"   ✅ {key}: {guitar_info['guitar']}, {guitar_info['pickup']}")
        else:
            print(f"   ⚠️ {key}: API failed, skipping")

    with open(presets_path, 'w') as f:
        json.dump(presets, f, indent=2)

    print(f"\n   💾 Filled {filled}/{len(missing)} guitar blocks")


# ==========================================
# CLI
# ==========================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Spark AI Tone Engineer — API Research Layer")
    parser.add_argument("--research", type=str, help="Research gear for a song/artist")
    parser.add_argument("--compare", type=str, help="Compare API research against existing preset")
    parser.add_argument("--forget", type=str, help="Delete auto-generated preset matching keyword")
    parser.add_argument("--forget-all-auto", action="store_true", help="Delete ALL auto-generated presets (preserves hand-curated)")
    parser.add_argument("--fill-guitar", action="store_true", help="Fill missing guitar blocks via API")
    parser.add_argument("--presets", type=str, default="artist_presets.json", help="Path to artist_presets.json")

    args = parser.parse_args()

    # Load gear database
    try:
        with open('spark_gear.json', 'r') as f:
            db = json.load(f)
    except:
        print("❌ Could not load spark_gear.json")
        sys.exit(1)

    if args.forget_all_auto:
        forget_all_auto(args.presets)
    elif args.forget:
        forget_preset(args.forget, args.presets)
    elif args.compare:
        compare_preset(args.compare, db, args.presets)
    elif args.research:
        research_song(args.research, db, save=True, presets_path=args.presets)
    elif args.fill_guitar:
        fill_guitar_blocks(db, args.presets)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
