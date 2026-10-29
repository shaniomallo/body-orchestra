import os
import re
import glob
import numpy as np
import librosa
import soundfile as sf

# --- config ---------------------------------------------------------
SRC_DIR   = "SalamanderGrandPianoV3_44.1khz16bit/44.1khz16bit"  #the .wav files
OUT_DIR   = "audio/piano"
OUT_LO, OUT_HI = 33, 96        
TARGET_VEL = 9                 
SR        = 44100
OCTAVE_OFFSET = 0              
# --------------------------------------------------------------------

LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NAME_RE = re.compile(r"^([A-G])(#?)(\d+)v(\d+)\.wav$", re.IGNORECASE)


def name_to_midi(letter, sharp, octave):
    pc = LETTER[letter.upper()] + (1 if sharp else 0)
    return pc + 12 * (int(octave) + 1 + OCTAVE_OFFSET)


def scan_sources():
    """Return {midi: filepath} choosing the velocity layer nearest TARGET_VEL."""
    best = {}          # midi -> (filepath, vel_distance)
    for path in glob.glob(os.path.join(SRC_DIR, "*.wav")):
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        letter, sharp, octave, vel = m.group(1), m.group(2), m.group(3), int(m.group(4))
        midi = name_to_midi(letter, sharp, octave)
        dist = abs(vel - TARGET_VEL)
        if midi not in best or dist < best[midi][1]:
            best[midi] = (path, dist)
    return {midi: fp for midi, (fp, _) in best.items()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sources = scan_sources()
    if not sources:
        print("No source samples parsed. Check SRC_DIR and the filename pattern.")
        print("Expected names like 'A1v9.wav', 'C4v9.wav', 'F#3v9.wav'.")
        return

    src_midis = sorted(sources)
    print(f"Found {len(src_midis)} recorded notes "
          f"(MIDI {src_midis[0]}..{src_midis[-1]}). Generating {OUT_LO}..{OUT_HI}.")

    for target in range(OUT_LO, OUT_HI + 1):
        src_midi = min(src_midis, key=lambda m: abs(m - target))
        y, _ = librosa.load(sources[src_midi], sr=SR, mono=True)
        n_steps = target - src_midi
        if n_steps != 0:
            y = librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)
        # normalise gently so every note sits at a consistent level
        peak = np.max(np.abs(y)) or 1.0
        y = (y / peak) * 0.9
        out = os.path.join(OUT_DIR, f"piano_{target}.wav")
        sf.write(out, y, SR, subtype="PCM_16")

    print(f"Done. Wrote {OUT_HI - OUT_LO + 1} files to {OUT_DIR}/")


if __name__ == "__main__":
    main()
