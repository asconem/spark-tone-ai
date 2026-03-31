#!/usr/bin/env python3
"""
calibrate_amps.py — Automated Spark 2 amp base_tone calibration

Connects to the Spark 2 via BLE, cycles through all guitar amp models
with knobs at noon, sends a test signal via USB audio, records the
processed output, and measures the actual tonal characteristics to
calibrate the base_tone values in spark_gear.json.

Requirements:
    pip install bleak sounddevice

Hardware setup:
    - Spark 2 connected via USB-C to Mac (for bidirectional audio)
    - Spark 2 powered on and discoverable via Bluetooth

Usage:
    python calibrate_amps.py                     # Full calibration (dry run)
    python calibrate_amps.py --write             # Full calibration, update spark_gear.json
    python calibrate_amps.py --amp "Black Duo"   # Calibrate a single amp
    python calibrate_amps.py --list-audio        # Show available audio devices
    python calibrate_amps.py --test              # Test BLE + audio pipeline on one amp

Protocol reference:
    Based on community reverse-engineering by Paul Hamshere, stangreg (Ignitron),
    and the SparkBox/Soundshed projects. BLE service FFC0, chars FFC1/FFC2.
"""

import asyncio
import json
import struct
import os
import sys
import tempfile
import argparse
import shutil
import numpy as np

# Add project root to path for analyze_tone import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
#  Spark BLE Protocol Constants
# =============================================================================

SPARK_BLE_SERVICE_SHORT = "ffc0"
SPARK_BLE_WRITE_UUID  = "0000ffc1-0000-1000-8000-00805f9b34fb"
SPARK_BLE_NOTIFY_UUID = "0000ffc2-0000-1000-8000-00805f9b34fb"

# Amp knob parameter indices (values are 0.0–1.0 floats)
PARAM_GAIN   = 0
PARAM_TREBLE = 1
PARAM_MIDDLE = 2
PARAM_BASS   = 3
PARAM_MASTER = 4

# Feature normalization constants — MUST match main.py exactly
AIR_MIN   = 0.328;  AIR_RANGE   = 0.290   # air  → base_tone['treble']
MIDS_MIN  = 0.30;   MIDS_RANGE  = 0.35    # mids → base_tone['mids']
BASS_MIN  = 0.20;   BASS_RANGE  = 0.30    # low_energy_ratio → base_tone['bass']

# Display name → Spark firmware protocol ID
# Sources: Ignitron EffectReference.csv, paulhamsh/Spark, SparkBox
AMP_IDS = {
    # ── Clean ──
    "Silver 120":          "RolandJC120",
    "Black Duo":           "Twin",
    "AD Clean":            "ADClean",
    "MATCH DC":            "94MatchDCV2",
    "ODS 50":              "ODS50CN",
    # ── Glassy ──
    "Tweed Bass":          "Bassman",
    "AC Boost":            "AC Boost",
    "Two Stone SP50":      "TwoStoneSP50",
    "Checkmate":           "Checkmate",
    # ── Crunch ──
    "Plexiglas":           "Plexi",
    "JM45":                "OverDrivenJM45",
    "American Deluxe":     "Deluxe65",
    "Lux Verb":            "OverDrivenLuxVerb",
    "Blues Boy":            "BluesJrTweed",
    # ── High Gain ──
    "British 30":          "OrangeAD30",
    "American High Gain":  "AmericanHighGain",
    "SLO 100":             "SLO100",
    "RB 101":              "Bogner",
    "YJM100":              "YJM100",
    # ── Metal ──
    "Treadplate":          "Rectifier",
    "Insane":              "EVH",
    "BE 101":              "BE101",
    "SwitchAxe":           "SwitchAxeLead",
    "Rocker V":            "Invader",
    "Insane 6508":         "6505Plus",
    # ── Jimi Hendrix (requires purchased pack) ──
    "J.H. 45/100":         "JH.JTM45",
    "J.H. Super 100":      "JH.SuperLead100",      # confirmed
    "J.H. Bass Master":    "JH.Bassman",
    "J.H. D-Show Master":  "JH.DualShowman",
    "J.H. Sun 100S":       "JH.Sunn100S",
    "J.H. Tone City 100":  "JH.SoundCity100",
    # ── Bass ──
    "RB-800":              "GK800",
    "Sunny 3000":          "Sunny3000",
    "W600":                "W600",
    "Hammer 500":          "Hammer500",
}


# =============================================================================
#  Spark BLE Message Builder
# =============================================================================

class SparkMessage:
    """Builds Spark amp BLE protocol messages.

    The Spark protocol encodes data using a 7-bit scheme (like MIDI SysEx):
    every group of 7 data bytes is stored in 8 buffer bytes — a "high-bits"
    byte followed by the 7 low-7-bit data bytes.  This keeps all buffer
    bytes below 0x80 in the data region.

    Multi-block messages (used for full preset uploads) split data across
    blocks of 128 (0x80) data bytes each, with a 3-byte chunk header at
    the start of each block's data region.

    Block layout (single or per-chunk):
        [0-1]   01 FE           direction marker (app → amp)
        [2-3]   00 00           reserved
        [4-5]   53 FE           direction marker continued
        [6]     length          total block byte count
        [7-15]  00 … 00         padding
        [16-17] F0 01           chunk signature
        [18]    seq             sequence number
        [19]    15              constant
        [20]    cmd             command byte
        [21]    sub_cmd         sub-command byte
        [22+]   7-bit data      payload (with high-bits packing)
        [last]  F7              end marker
    """

    BLK_SIZE = 173          # 0xAD max bytes per BLE block
    DATA_PER_BLOCK = 128    # 0x80 logical data bytes per block

    def __init__(self):
        self._seq = 0x40

    # ── low-level encoding ──────────────────────────────────────────────

    @staticmethod
    def _new_ctx(cmd, sub_cmd, multi=False):
        return {
            'blocks': [bytearray(SparkMessage.BLK_SIZE) for _ in range(10)],
            'data_pos': 0,
            'last_block': 0,
            'last_pos': 0,
            'multi': multi,
            'cmd': cmd,
            'sub_cmd': sub_cmd,
        }

    @staticmethod
    def _add_byte(ctx, b):
        blk = ctx['data_pos'] // SparkMessage.DATA_PER_BLOCK
        multi_off = 3 if ctx['multi'] else 0
        temp = (ctx['data_pos'] % SparkMessage.DATA_PER_BLOCK) + multi_off
        sl   = 8 * (temp // 7)
        bp   = temp % 7
        bit_pos = 22 + sl
        pos     = 23 + bp + sl

        ctx['blocks'][blk][pos] = b & 0x7F
        if b & 0x80:
            ctx['blocks'][blk][bit_pos] |= (1 << bp)

        ctx['data_pos'] += 1
        ctx['last_block'] = blk
        ctx['last_pos'] = pos

    @staticmethod
    def _add_prefixed_string(ctx, s):
        """Length + (0xA0+length) + chars  — used for effect names in commands."""
        SparkMessage._add_byte(ctx, len(s))
        SparkMessage._add_byte(ctx, len(s) + 0xA0)
        for c in s:
            SparkMessage._add_byte(ctx, ord(c))

    @staticmethod
    def _add_string(ctx, s):
        """Short or long string encoding for preset fields."""
        if len(s) > 31:
            SparkMessage._add_byte(ctx, 0xD9)
            SparkMessage._add_byte(ctx, len(s))
        else:
            SparkMessage._add_byte(ctx, len(s) + 0xA0)
        for c in s:
            SparkMessage._add_byte(ctx, ord(c))

    @staticmethod
    def _add_long_string(ctx, s):
        SparkMessage._add_byte(ctx, 0xD9)
        SparkMessage._add_byte(ctx, len(s))
        for c in s:
            SparkMessage._add_byte(ctx, ord(c))

    @staticmethod
    def _add_float(ctx, f):
        SparkMessage._add_byte(ctx, 0xCA)
        for b in struct.pack('>f', f):
            SparkMessage._add_byte(ctx, b)

    @staticmethod
    def _add_onoff(ctx, on):
        SparkMessage._add_byte(ctx, 0xC3 if on else 0xC2)

    def _finalize(self, ctx):
        """Fill block headers, chunk metadata, and return list of byte blocks."""
        self._seq = (self._seq + 1) & 0xFF

        num_chunks = ctx['last_block'] + 1
        data_len   = ctx['data_pos']
        last_chunk_len = data_len % self.DATA_PER_BLOCK
        if last_chunk_len == 0 and data_len > 0:
            last_chunk_len = self.DATA_PER_BLOCK

        # multi-chunk: write chunk header directly into buffer positions 23-25
        if ctx['multi']:
            for i in range(num_chunks):
                ctx['blocks'][i][23] = num_chunks
                ctx['blocks'][i][24] = i
                if i < num_chunks - 1:
                    ctx['blocks'][i][25] = 0x00        # 0x80 with bit 7 stored separately
                    ctx['blocks'][i][22] |= 0x04       # bit 7 of byte at position 25
                else:
                    ctx['blocks'][i][25] = last_chunk_len

        result = []
        for i in range(num_chunks):
            # end marker position
            if i == num_chunks - 1:
                pos = ctx['last_pos'] + 1
            else:
                pos = 172                              # full block: 0xAD - 1
            ctx['blocks'][i][pos] = 0xF7

            # block header
            blk = ctx['blocks'][i]
            blk[0]  = 0x01;  blk[1]  = 0xFE
            blk[4]  = 0x53;  blk[5]  = 0xFE
            blk[6]  = pos + 1                          # block size
            blk[16] = 0xF0;  blk[17] = 0x01
            blk[18] = self._seq
            blk[19] = 0x15
            blk[20] = ctx['cmd']
            blk[21] = ctx['sub_cmd']

            result.append(bytes(blk[:pos + 1]))

        return result

    # ── public message builders ─────────────────────────────────────────

    def create_preset(self, amp_id, master_vol=0.15):
        """Build a full preset message: target amp ON at noon, everything else OFF.

        This is a multi-block message.  It sets the amp in a known state
        regardless of what was loaded before — no need to know the current preset.
        """
        ctx = self._new_ctx(0x01, 0x01, multi=True)

        # preset header
        self._add_byte(ctx, 0x00)
        self._add_byte(ctx, 0x7F)                     # temp preset slot
        self._add_long_string(ctx, "CALIB000-0000-0000-0000-000000000000")
        self._add_string(ctx, "Calibrate")
        self._add_string(ctx, "0.7")
        self._add_string(ctx, "Cal")
        self._add_string(ctx, "icon.png")
        self._add_float(ctx, 120.0)

        self._add_byte(ctx, 0x90 + 7)                 # always 7 pedal slots

        # slot 0 — noise gate OFF
        self._add_string(ctx, "bias.noisegate")
        self._add_onoff(ctx, False)
        self._add_byte(ctx, 0x90 + 3)
        for i, v in enumerate([0.5, 0.5, 0.0]):
            self._add_byte(ctx, i);  self._add_byte(ctx, 0x91);  self._add_float(ctx, v)

        # slot 1 — compressor OFF
        self._add_string(ctx, "LA2AComp")
        self._add_onoff(ctx, False)
        self._add_byte(ctx, 0x90 + 3)
        for i, v in enumerate([0.0, 0.5, 0.5]):
            self._add_byte(ctx, i);  self._add_byte(ctx, 0x91);  self._add_float(ctx, v)

        # slot 2 — drive OFF
        self._add_string(ctx, "Booster")
        self._add_onoff(ctx, False)
        self._add_byte(ctx, 0x90 + 1)
        self._add_byte(ctx, 0);  self._add_byte(ctx, 0x91);  self._add_float(ctx, 0.5)

        # slot 3 — AMP ON (the one we're calibrating)
        self._add_string(ctx, amp_id)
        self._add_onoff(ctx, True)
        self._add_byte(ctx, 0x90 + 5)
        for i, v in enumerate([0.5, 0.5, 0.5, 0.5, master_vol]):
            self._add_byte(ctx, i);  self._add_byte(ctx, 0x91);  self._add_float(ctx, v)

        # slot 4 — modulation OFF
        self._add_string(ctx, "Cloner")
        self._add_onoff(ctx, False)
        self._add_byte(ctx, 0x90 + 2)
        for i, v in enumerate([0.5, 0.0]):
            self._add_byte(ctx, i);  self._add_byte(ctx, 0x91);  self._add_float(ctx, v)

        # slot 5 — delay OFF
        self._add_string(ctx, "VintageDelay")
        self._add_onoff(ctx, False)
        self._add_byte(ctx, 0x90 + 4)
        for i, v in enumerate([0.5, 0.5, 0.5, 1.0]):
            self._add_byte(ctx, i);  self._add_byte(ctx, 0x91);  self._add_float(ctx, v)

        # slot 6 — reverb OFF
        self._add_string(ctx, "bias.reverb")
        self._add_onoff(ctx, False)
        self._add_byte(ctx, 0x90 + 7)
        for i, v in enumerate([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]):
            self._add_byte(ctx, i);  self._add_byte(ctx, 0x91);  self._add_float(ctx, v)

        self._add_byte(ctx, 0xB4)                     # end filler

        return self._finalize(ctx)

    def change_effect(self, old_id, new_id):
        """Swap one effect for another in its slot (single-block)."""
        ctx = self._new_ctx(0x01, 0x06)
        self._add_prefixed_string(ctx, old_id)
        self._add_prefixed_string(ctx, new_id)
        return self._finalize(ctx)

    def change_parameter(self, effect_id, param_idx, value):
        """Set a single parameter on an effect (single-block)."""
        ctx = self._new_ctx(0x01, 0x04)
        self._add_prefixed_string(ctx, effect_id)
        self._add_byte(ctx, param_idx)
        self._add_float(ctx, value)
        return self._finalize(ctx)

    def change_hardware_preset(self, preset_num):
        """Switch to a hardware preset 0-3 (single-block)."""
        ctx = self._new_ctx(0x01, 0x38)
        self._add_byte(ctx, 0)
        self._add_byte(ctx, preset_num)
        return self._finalize(ctx)


# =============================================================================
#  Spark BLE Connection (bleak)
# =============================================================================

class SparkBLE:
    """Async BLE connection to the Spark 2."""

    def __init__(self):
        self.client = None
        self.msg = SparkMessage()

    async def connect(self, timeout=15.0):
        from bleak import BleakScanner, BleakClient

        print("   Scanning for Spark amp...")
        spark_device = None
        devices = await BleakScanner.discover(timeout=timeout)

        for d in devices:
            uuids = [str(u).lower() for u in d.metadata.get("uuids", [])]
            if any(SPARK_BLE_SERVICE_SHORT in u for u in uuids):
                spark_device = d
                break
            if d.name and "spark" in d.name.lower():
                spark_device = d
                break

        if not spark_device:
            print("   ERROR: No Spark amp found.")
            print("   Make sure it's powered on and Bluetooth is enabled.")
            return False

        print(f"   Found: {spark_device.name} ({spark_device.address})")
        self.client = BleakClient(spark_device.address)
        await self.client.connect()

        if not self.client.is_connected:
            print("   ERROR: BLE connection failed.")
            return False

        print(f"   Connected to {spark_device.name}")
        return True

    async def disconnect(self):
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            print("   BLE disconnected.")

    async def send(self, blocks):
        """Send one or more BLE blocks to the Spark."""
        for i, block in enumerate(blocks):
            await self.client.write_gatt_char(SPARK_BLE_WRITE_UUID, block, response=True)
            if len(blocks) > 1:
                await asyncio.sleep(0.1)

    async def send_preset(self, amp_id, master_vol=0.15):
        blocks = self.msg.create_preset(amp_id, master_vol)
        await self.send(blocks)


# =============================================================================
#  Test Signal Generation
# =============================================================================

def generate_test_signal(sr=44100, duration=5.0):
    """Synthetic DI-like guitar signal for amp calibration.

    Four plucks at different pitches covering the guitar range, each with
    11 harmonics and a natural attack/decay envelope.  The signal has known,
    flat-ish spectral content so any shaping in the output is attributable
    to the amp model.

    Returns mono float32 array at -18 dBFS peak.
    """
    n_samples = int(sr * duration)
    signal = np.zeros(n_samples, dtype=np.float32)

    plucks = [
        (0.3, 196.0, 0.9),    # G3 — low-mid
        (1.4, 329.6, 0.9),    # E4 — mid
        (2.5, 440.0, 0.9),    # A4 — upper-mid
        (3.6, 196.0, 0.9),    # G3 repeat — reinforce low end
    ]

    for start_s, freq, dur in plucks:
        i0 = int(start_s * sr)
        i1 = min(i0 + int(dur * sr), n_samples)
        t  = np.arange(i1 - i0, dtype=np.float32) / sr

        pluck = np.zeros_like(t)
        for h in range(1, 12):
            if freq * h > sr / 2:
                break
            amp   = 1.0 / (h ** 0.7)
            h_freq = freq * h * (1.0 + 0.0002 * h * h)   # slight inharmonicity
            pluck += amp * np.sin(2 * np.pi * h_freq * t)

        attack  = np.minimum(t / 0.005, 1.0)              # 5 ms
        decay   = np.exp(-t / 0.40)                        # 400 ms τ
        signal[i0:i1] += (pluck * attack * decay).astype(np.float32)

    # normalize to -18 dBFS
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal *= (10 ** (-18 / 20)) / peak

    return signal


# =============================================================================
#  USB Audio I/O
# =============================================================================

def find_spark_audio_device():
    """Auto-detect the Spark USB audio device index and name."""
    import sounddevice as sd
    devices = sd.query_devices()

    # Pass 1: exact match on "spark" with both in + out channels
    for i, d in enumerate(devices):
        nm = d['name'].lower()
        if 'spark' in nm and d['max_input_channels'] > 0 and d['max_output_channels'] > 0:
            return i, d['name']

    # Pass 2: separate input/output devices that both match "spark"
    in_idx = out_idx = None
    for i, d in enumerate(devices):
        nm = d['name'].lower()
        if 'spark' in nm or 'positive grid' in nm:
            if d['max_input_channels'] > 0 and in_idx is None:
                in_idx = i
            if d['max_output_channels'] > 0 and out_idx is None:
                out_idx = i

    if in_idx is not None and out_idx is not None:
        return (in_idx, out_idx), devices[in_idx]['name']

    return None, None


def record_through_spark(test_signal, device, sr=44100):
    """Play test_signal to Spark via USB and record the processed output.

    device: int (combined device) or (int, int) for separate in/out devices.
    Returns mono float32 numpy array.
    """
    import sounddevice as sd

    # pad with 0.5s silence before and 1.0s after (capture amp tail / latency)
    pre  = np.zeros(int(0.5 * sr), dtype=np.float32)
    post = np.zeros(int(1.0 * sr), dtype=np.float32)
    padded = np.concatenate([pre, test_signal, post])

    if isinstance(device, tuple):
        in_dev, out_dev = device
        sd.default.device = (in_dev, out_dev)
        rec = sd.playrec(padded.reshape(-1, 1), samplerate=sr, channels=1, dtype='float32')
    else:
        rec = sd.playrec(padded.reshape(-1, 1), samplerate=sr, channels=1,
                         device=device, dtype='float32')
    sd.wait()
    return rec.flatten()


def save_temp_wav(audio, sr=44100):
    """Write audio to a temp .wav file; caller must os.unlink() it."""
    from scipy.io import wavfile
    path = tempfile.mktemp(suffix='.wav')
    wavfile.write(path, sr, np.int16(np.clip(audio, -1, 1) * 32767))
    return path


# =============================================================================
#  Feature → base_tone Mapping
# =============================================================================

def features_to_base_tone(features):
    """Convert analyze_tone() features to base_tone dict.

    Uses the SAME normalization constants as build_rig() in main.py so that
    calibrated values are directly comparable to normalized song features.
    """
    gain = features.get('gain', 0.5)
    air  = features.get('air', 0.5)
    mids = features.get('mids', 0.5)
    bass = features.get('low_energy_ratio', 0.35)

    return {
        'gain':   round(float(np.clip(gain, 0, 1)), 3),
        'treble': round(float(np.clip((air  - AIR_MIN)  / AIR_RANGE,  0, 1)), 3),
        'mids':   round(float(np.clip((mids - MIDS_MIN) / MIDS_RANGE, 0, 1)), 3),
        'bass':   round(float(np.clip((bass - BASS_MIN) / BASS_RANGE, 0, 1)), 3),
    }


# =============================================================================
#  Main Calibration Loop
# =============================================================================

async def calibrate(args):
    import sounddevice as sd
    from main import analyze_tone

    gear_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spark_gear.json')
    with open(gear_path) as f:
        db = json.load(f)

    print("\n=== Spark 2 Amp Calibration ===\n")

    # ── audio device ────────────────────────────────────────────────────
    if args.list_audio:
        print(sd.query_devices())
        return

    if args.audio_device is not None:
        device = args.audio_device
        dev_name = sd.query_devices(device)['name']
    else:
        device, dev_name = find_spark_audio_device()

    if device is None:
        print("ERROR: Spark USB audio device not found.")
        print("Make sure the Spark 2 is connected via USB-C.")
        print("Use --list-audio to see available devices, --audio-device N to select one.")
        return

    print(f"   USB audio: {dev_name}")

    # ── test signal ─────────────────────────────────────────────────────
    print("   Generating test signal...")
    test_signal = generate_test_signal()

    # ── BLE connect ─────────────────────────────────────────────────────
    spark = SparkBLE()
    if not await spark.connect():
        return

    # ── build amp list ──────────────────────────────────────────────────
    amps = []
    for cat in db['amps']:
        if cat['category'] == 'Bass' and not args.include_bass:
            continue
        for amp in cat['models']:
            pid = AMP_IDS.get(amp['name'])
            if pid is None:
                print(f"   WARNING: no protocol ID for '{amp['name']}' — skipping")
                continue
            amps.append({
                'name':        amp['name'],
                'pid':         pid,
                'category':    cat['category'],
                'old_bt':      dict(amp['base_tone']),
            })

    if args.amp:
        amps = [a for a in amps if a['name'].lower() == args.amp.lower()]
        if not amps:
            print(f"   ERROR: amp '{args.amp}' not found")
            await spark.disconnect()
            return

    if not args.include_jh:
        before = len(amps)
        amps = [a for a in amps if not a['pid'].startswith('JH.')]
        skipped = before - len(amps)
        if skipped:
            print(f"   Skipping {skipped} Jimi Hendrix model(s) (use --include-jh)")

    if args.test:
        amps = amps[:1]
        print(f"   TEST MODE: calibrating only '{amps[0]['name']}'")

    total = len(amps)
    print(f"\n   Calibrating {total} amp model{'s' if total != 1 else ''}  "
          f"(master vol {args.volume:.0%})\n")

    results = {}

    for idx, amp in enumerate(amps):
        name = amp['name']
        pid  = amp['pid']
        tag  = f"[{idx+1}/{total}]"

        print(f"   {tag} {name} ({amp['category']})")

        # send calibration preset
        try:
            await spark.send_preset(pid, args.volume)
        except Exception as e:
            print(f"         BLE send error: {e} — skipping")
            continue

        # let the amp model load
        await asyncio.sleep(args.settle)

        # play / record via USB
        try:
            recording = record_through_spark(test_signal, device)
        except Exception as e:
            print(f"         Audio error: {e} — skipping")
            continue

        # analyze
        tmp = save_temp_wav(recording)
        try:
            features = analyze_tone(tmp)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

        if features is None:
            print(f"         Analysis failed — skipping")
            continue

        cal = features_to_base_tone(features)
        old = amp['old_bt']
        results[name] = cal

        # delta display
        for key in ('gain', 'treble', 'mids', 'bass'):
            o, n = old[key], cal[key]
            delta = n - o
            sign = '+' if delta >= 0 else ''
            print(f"         {key:7s}  {o:.2f} → {n:.3f}  ({sign}{delta:.3f})")

    await spark.disconnect()

    if not results:
        print("\n   No amps calibrated successfully.")
        return

    # ── save results ────────────────────────────────────────────────────
    print(f"\n   Calibrated {len(results)}/{total} amp models.")

    # always save raw results
    raw_path = os.path.join(os.path.dirname(gear_path), 'calibration_results.json')
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Raw results → {raw_path}")

    if not args.write:
        print("\n   DRY RUN — spark_gear.json not modified.")
        print("   Re-run with --write to apply calibrated values.")
        return

    # backup original
    backup = gear_path + '.pre_calibration'
    if not os.path.exists(backup):
        shutil.copy2(gear_path, backup)
        print(f"   Backup → {backup}")

    # update
    updated = 0
    for cat in db['amps']:
        for amp in cat['models']:
            if amp['name'] in results:
                amp['base_tone'] = results[amp['name']]
                updated += 1

    with open(gear_path, 'w') as f:
        json.dump(db, f, indent=2)

    print(f"   Updated {updated} base_tone values in spark_gear.json")


# =============================================================================
#  CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Automated Spark 2 amp base_tone calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --list-audio            Show audio devices
  %(prog)s --test                  Pipeline test on one amp (dry run)
  %(prog)s                         Full calibration dry run
  %(prog)s --write                 Full calibration, update spark_gear.json
  %(prog)s --amp "SLO 100"         Calibrate one amp
  %(prog)s --write --include-jh    Include Jimi Hendrix pack models
""")

    p.add_argument('--write', action='store_true',
                   help='Write calibrated values to spark_gear.json (default: dry run)')
    p.add_argument('--volume', type=float, default=0.15,
                   help='Master volume 0.0–1.0 (default: 0.15 ≈ quiet)')
    p.add_argument('--settle', type=float, default=1.5,
                   help='Seconds to wait after loading each amp model (default: 1.5)')
    p.add_argument('--amp', type=str, default=None,
                   help='Calibrate a single amp by display name')
    p.add_argument('--test', action='store_true',
                   help='Test mode: one amp only, dry run')
    p.add_argument('--include-bass', action='store_true',
                   help='Include bass amp models')
    p.add_argument('--include-jh', action='store_true',
                   help='Include Jimi Hendrix models (requires purchased pack)')
    p.add_argument('--audio-device', type=int, default=None,
                   help='Audio device index (see --list-audio)')
    p.add_argument('--list-audio', action='store_true',
                   help='List available audio devices and exit')

    args = p.parse_args()
    asyncio.run(calibrate(args))


if __name__ == '__main__':
    main()
