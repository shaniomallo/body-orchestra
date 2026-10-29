"""
Jazz Drum Looper — Organic Shapes + Markov Ensemble + Overdub

Body pose → IML classifier → drum sound → drum loop.
Keyboard keys bring in Markov-generated piano, bass, and sax melodies.
Organic visuals react to drum hits and ensemble notes.

Keys:
R = record first loop / close loop / re-record
O = overdub drums on/off
U = undo last overdub layer
P = toggle piano Markov melody
B = toggle bass Markov melody
S = toggle sax Markov melody
V = show / hide skeleton
C = clear loop + visuals
UP = faster melodies
DOWN = slower melodies
"""

import cv2
import time
import math
import random
import os
import pygame
import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dorothy import Dorothy
from learner import Learner, LearnerGUI, CLASSIFICATION


dot = Dorothy()


# ── AUDIO: PYGAME MIXER ───────────────────────────────────────────
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.init()
pygame.mixer.set_num_channels(32)


def load_sound(path):
    if not os.path.exists(path):
        print(f"Missing audio file: {path}")
        return None
    try:
        return pygame.mixer.Sound(path)
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return None


def play_sound(sound, volume=1.0):
    if sound is None:
        return
    try:
        sound.set_volume(volume)
        sound.play()
    except Exception as e:
        print(f"Audio play error: {e}")


# ── IML CLASSIFIER ────────────────────────────────────────────────
NUM_LANDMARKS = 33
NUM_CLASSES = 6  # 0-4 drums, class 5 can be neutral/rest if you want

learner = Learner(
    num_inputs=NUM_LANDMARKS * 2,
    num_classes=NUM_CLASSES,
    model_type=CLASSIFICATION,
    model_name="skeleton"
)
gui = LearnerGUI(learner)


# ── DRUM SAMPLES ──────────────────────────────────────────────────
DRUM_SOUNDS = {
    0: load_sound("audio/kick.wav"),
    1: load_sound("audio/snare.wav"),
    2: load_sound("audio/hihat_closed.wav"),
    3: load_sound("audio/hihat_open.wav"),
    4: load_sound("audio/crash.wav"),
}

DRUM_NAMES = {
    0: "KICK",
    1: "SNARE",
    2: "HH CLOSED",
    3: "HH OPEN",
    4: "CRASH",
}

DRUM_COLOURS = {
    0: (220, 50, 30),
    1: (240, 240, 240),
    2: (220, 180, 40),
    3: (100, 220, 80),
    4: (140, 80, 220),
}


# ── PIANO/BASS/SAX NOTE SAMPLES ───────────────────────────────
PIANO_FILES = [
    "audio/piano_Bb4.wav",
    "audio/piano_C4.wav",
    "audio/piano_C5.wav",
    "audio/piano_Eb4.wav",
    "audio/piano_F4.wav",
    "audio/piano_G4.wav",
    "audio/piano_Gb4.wav",
]

BASS_FILES = [
    "audio/bass_A2.wav",
    "audio/bass_B2.wav",
    "audio/bass_Bb2.wav",
    "audio/bass_C2.wav",
    "audio/bass_D2.wav",
    "audio/bass_C3.wav",
    "audio/bass_E2.wav",
    "audio/bass_Eb2.wav",
    "audio/bass_F2.wav",
    "audio/bass_G2.wav",
    "audio/bass_Gb2.wav",
]

SAX_FILES = [
    "audio/sax_A4.wav",
    "audio/sax_B4.wav",
    "audio/sax_Bb4.wav",
    "audio/sax_C4.wav",
    "audio/sax_C5.wav",
    "audio/sax_D4.wav",
    "audio/sax_E4.wav",
    "audio/sax_Eb4.wav",
    "audio/sax_F4.wav",
    "audio/sax_Gb4.wav",
    "audio/sax_G4.wav",
    "audio/sax_G5.wav",
]

PIANO_SOUNDS = [load_sound(p) for p in PIANO_FILES]
BASS_SOUNDS = [load_sound(p) for p in BASS_FILES]
SAX_SOUNDS = [load_sound(p) for p in SAX_FILES]


# ── MARKOV MELODY MODELS ──────────────────────────────────────────

PIANO_SEED_PHRASES = [
    [1, 3, 4, 5, 4, 3, 1, 0],
    [1, 4, 6, 5, 3, 4, 1, 3],
    [5, 4, 3, 1, 3, 4, 5, 3],
    [1, 3, 5, 6, 5, 4, 3, 1],
]

BASS_SEED_PHRASES = [
    [3, 7, 2, 8, 7, 5, 3, 7],
    [3, 5, 7, 8, 7, 5, 3, 2],
    [3, 3, 7, 2, 8, 7, 5, 3],
    [5, 7, 8, 10, 8, 7, 5, 3],
]

SAX_SEED_PHRASES = [
    [3, 5, 8, 10, 8, 5, 2, 3],
    [0, 3, 5, 10, 11, 10, 8, 5],
    [5, 8, 10, 8, 5, 3, 2, 0],
    [3, 5, 8, 11, 10, 8, 5, 3],
]


class MarkovMelody:
    """
    Very small first-order Markov melody generator.
    It learns transitions from seed phrases:
        current note index -> possible next note indexes
    """

    def __init__(self, seed_phrases, num_notes, name="INSTRUMENT", mutation_chance=0.12):
        self.seed_phrases = seed_phrases
        self.num_notes = num_notes
        self.name = name
        self.mutation_chance = mutation_chance
        self.transitions = {}
        self.starts = []
        self.current = None
        self.train(seed_phrases)

    def train(self, seed_phrases):
        self.transitions = {}
        self.starts = []

        for phrase in seed_phrases:
            clean = [n for n in phrase if isinstance(n, int) and 0 <= n < self.num_notes]
            if not clean:
                continue

            self.starts.append(clean[0])

            for a, b in zip(clean, clean[1:]):
                self.transitions.setdefault(a, []).append(b)

            #phrases wrap around so loops can continue smoothly
            if len(clean) > 1:
                self.transitions.setdefault(clean[-1], []).append(clean[0])

        if not self.starts:
            self.starts = list(range(max(1, self.num_notes)))

    def reset(self):
        self.current = random.choice(self.starts)
        return self.current

    def next(self):
        if self.current is None:
            return self.reset()

        # Small mutation chance stops it sounding too fixed.
        if random.random() < self.mutation_chance:
            self.current = random.randrange(self.num_notes)
            return self.current

        options = self.transitions.get(self.current)

        if not options:
            self.current = random.choice(self.starts)
        else:
            self.current = random.choice(options)

        return self.current


class EnsembleGenerator:
    """
    Keyboard-controlled Markov ensemble
    P starts piano, B starts bass, S starts sax
    Active instruments generate note-by-note as soon as they are toggled on
    """

    def __init__(self):
        self.piano_on = False
        self.bass_on = False
        self.sax_on = False

        self.interval = 0.45
        self.last_time = 0.0

        self.piano_model = MarkovMelody(PIANO_SEED_PHRASES, len(PIANO_SOUNDS), "PIANO", mutation_chance=0.10)
        self.bass_model = MarkovMelody(BASS_SEED_PHRASES, len(BASS_SOUNDS), "BASS", mutation_chance=0.06)
        self.sax_model = MarkovMelody(SAX_SEED_PHRASES, len(SAX_SOUNDS), "SAX", mutation_chance=0.18)

        self.last_phrase_change = time.time()
        self.phrase_length = 8.0

    def reset_steps(self):
        self.piano_model.reset()
        self.bass_model.reset()
        self.sax_model.reset()
        self.last_time = time.time()

    def choose_new_phrase(self):
        #Resetting the Markov state creates a new phrase direction.
        self.reset_steps()
        self.last_phrase_change = time.time()
        print("Markov phrase states reset")

    def faster(self):
        self.interval = max(0.18, self.interval - 0.05)
        print(f"Melody interval: {self.interval:.2f}s")

    def slower(self):
        self.interval = min(1.2, self.interval + 0.05)
        print(f"Melody interval: {self.interval:.2f}s")

    def toggle_piano(self):
        self.piano_on = not self.piano_on
        self.piano_model.reset()
        #Make the first note play immediately after toggling on
        self.last_time = 0.0
        print(f"PIANO MARKOV {'ON' if self.piano_on else 'OFF'}")

    def toggle_bass(self):
        self.bass_on = not self.bass_on
        self.bass_model.reset()
        #Make the first note play immediately after toggling on
        self.last_time = 0.0
        print(f"BASS MARKOV {'ON' if self.bass_on else 'OFF'}")

    def toggle_sax(self):
        self.sax_on = not self.sax_on
        self.sax_model.reset()
        # Make the first note play immediately after toggling on
        self.last_time = 0.0
        print(f"SAX MARKOV {'ON' if self.sax_on else 'OFF'}")

    def update(self, loop_is_playing=True):
       
        if not (self.piano_on or self.bass_on or self.sax_on):
            return []

        now = time.time()

        if now - self.last_phrase_change > self.phrase_length:
            self.choose_new_phrase()

        if now - self.last_time < self.interval:
            return []

        events = []

        if self.piano_on:
            idx = self.piano_model.next()
            if idx < len(PIANO_SOUNDS):
                play_sound(PIANO_SOUNDS[idx], volume=0.55)
                print(f"PIANO Markov note {idx}")
                events.append(("PIANO", idx))

        if self.bass_on:
            idx = self.bass_model.next()
            if idx < len(BASS_SOUNDS):
                play_sound(BASS_SOUNDS[idx], volume=0.75)
                print(f"BASS Markov note {idx}")
                events.append(("BASS", idx))

        if self.sax_on:
            # Sax plays slightly less often so it feels like a solo 
            if random.random() < 0.55:
                idx = self.sax_model.next()
                if idx < len(SAX_SOUNDS):
                    play_sound(SAX_SOUNDS[idx], volume=0.8)
                    print(f"SAX Markov note {idx}")
                    events.append(("SAX", idx))

        self.last_time = now
        return events


ensemble = EnsembleGenerator()


# ── SKELETON CONNECTIONS ──────────────────────────────────────────
SKELETON = [
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]


# ── LOOPER WITH OVERDUB ───────────────────────────────────────────
class Looper:
    def __init__(self):
        self.reset()

    def reset(self):
        self.recording = False
        self.overdubbing = False
        self.has_loop = False

        self.record_start = None
        self.loop_start = None
        self.loop_length = 0.0

        self.hits = []
        self.overdub_layers = []
        self.current_overdub_layer = []
        self._fired = set()

    def start_recording(self):
        self.recording = True
        self.overdubbing = False
        self.has_loop = False
        self.hits = []
        self.overdub_layers = []
        self.current_overdub_layer = []
        self.record_start = time.time()
        self.loop_start = None
        self.loop_length = 0.0
        self._fired = set()
        ensemble.reset_steps()
        print("● REC — play first loop, press R to close")

    def stop_recording(self):
        if self.record_start:
            self.loop_length = time.time() - self.record_start

        if self.loop_length < 0.5:
            print("Loop too short — try again")
            self.reset()
            return

        self.recording = False
        self.has_loop = True
        self.overdubbing = False
        self.loop_start = time.time()
        self._fired = set()
        ensemble.reset_steps()
        print(f"■ LOOP {self.loop_length:.2f}s | {len(self.hits)} hits")

    def toggle(self):
        if not self.recording and not self.has_loop:
            self.start_recording()
        elif self.recording:
            self.stop_recording()
        else:
            # R while loop exists means re-record from scratch
            self.reset()
            self.start_recording()

    def toggle_overdub(self):
        if not self.has_loop:
            print("No loop yet — press R to record your first loop")
            return

        if not self.overdubbing:
            self.overdubbing = True
            self.current_overdub_layer = []
            print("● OVERDUB ON")
        else:
            self.overdubbing = False
            if self.current_overdub_layer:
                self.overdub_layers.append(self.current_overdub_layer[:])
                print(f"○ OVERDUB OFF — saved layer with {len(self.current_overdub_layer)} hits")
            else:
                print("○ OVERDUB OFF — no hits added")
            self.current_overdub_layer = []

    def undo_last_overdub(self):
        if self.overdubbing:
            self.toggle_overdub()

        if not self.overdub_layers:
            print("No overdub layers to undo")
            return

        layer = self.overdub_layers.pop()

        for hit in layer:
            try:
                self.hits.remove(hit)
            except ValueError:
                pass

        self._fired = set()
        print(f"UNDO — removed last overdub layer ({len(layer)} hits)")

    def log_hit(self, drum_class):
        now = time.time()

        if self.recording and self.record_start:
            offset = now - self.record_start
            self.hits.append((drum_class, offset))
            return

        if self.overdubbing and self.has_loop and self.loop_start and self.loop_length > 0:
            offset = (now - self.loop_start) % self.loop_length
            hit = (drum_class, offset)
            self.hits.append(hit)
            self.current_overdub_layer.append(hit)
            self.hits.sort(key=lambda x: x[1])
            print(f"Overdub hit at {offset:.2f}s")

    def progress(self):
        if not self.has_loop or self.loop_length <= 0:
            return 0.0
        return ((time.time() - self.loop_start) % self.loop_length) / self.loop_length

    def update(self):
        out = []

        if not self.has_loop or not self.hits or self.loop_length < 0.1:
            return out

        elapsed = time.time() - self.loop_start
        loop_pos = elapsed % self.loop_length
        cycle = int(elapsed / self.loop_length)

        WINDOW = 0.025

        for i, (drum_class, offset) in enumerate(self.hits):
            hit_key = (cycle, i)

            if hit_key in self._fired:
                continue

            normal_hit = abs(loop_pos - offset) < WINDOW
            wrap_hit = offset > self.loop_length - WINDOW and loop_pos < WINDOW

            if normal_hit or wrap_hit:
                out.append(drum_class)
                self._fired.add(hit_key)

        # Clean up old fired hits so the set does not grow forever
        self._fired = {key for key in self._fired if key[0] >= cycle - 1}

        return out


looper = Looper()


# ── ORGANIC SHAPE SYSTEM ──────────────────────────────────────────
def blob_points(cx, cy, radius, n_points=18, roughness=0.35, seed=None):
    rng = random.Random(seed if seed else random.random())
    pts = []

    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        r = radius * (
            1.0
            + roughness * math.sin(3 * angle + rng.uniform(0, 2 * math.pi))
            + roughness * 0.5 * math.sin(7 * angle + rng.uniform(0, 2 * math.pi))
            + roughness * 0.3 * rng.uniform(-1, 1)
        )
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

    return pts


def spiral_points(cx, cy, turns=2.5, n=120, r_start=4, r_end=180, angle_offset=0):
    pts = []

    for i in range(n):
        t = i / n
        angle = angle_offset + t * turns * 2 * math.pi
        radius = r_start + (r_end - r_start) * t
        pts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

    return pts


class Shape:
    def __init__(self, life, colour):
        self.born = time.time()
        self.life = life
        self.colour = colour

    @property
    def alive(self):
        return (time.time() - self.born) < self.life

    @property
    def alpha(self):
        return max(0.0, 1.0 - (time.time() - self.born) / self.life)

    def faded(self, base_alpha=1.0):
        a = self.alpha * base_alpha
        r, g, b = self.colour
        return (int(r * a), int(g * a), int(b * a))

    def draw(self):
        pass


class BlobRing(Shape):
    def __init__(self, cx, cy, colour):
        super().__init__(life=1.4, colour=colour)
        self.cx = cx
        self.cy = cy
        self.n_blobs = 10
        self.seeds = [random.random() for _ in range(self.n_blobs)]
        self.max_r = min(dot.width, dot.height) * 0.45

    def draw(self):
        age = time.time() - self.born
        frac = age / self.life
        ring_r = self.max_r * frac
        blob_r = 28 * math.sin(frac * math.pi)

        if blob_r < 1:
            return

        dot.fill(self.faded())
        dot.no_stroke()

        for i in range(self.n_blobs):
            angle = 2 * math.pi * i / self.n_blobs
            bx = self.cx + ring_r * math.cos(angle)
            by = self.cy + ring_r * math.sin(angle)
            pts = blob_points(bx, by, blob_r, n_points=14, roughness=0.4, seed=self.seeds[i])
            dot.polygon(pts)


class SpiralBurst(Shape):
    def __init__(self, cx, cy, colour):
        super().__init__(life=0.9, colour=colour)
        self.cx = cx
        self.cy = cy
        self.n_arms = 6
        self.offsets = [random.uniform(0, 2 * math.pi) for _ in range(self.n_arms)]

    def draw(self):
        age = time.time() - self.born
        frac = age / self.life
        r_end = min(dot.width, dot.height) * 0.4 * frac

        dot.no_fill()
        dot.set_stroke_weight(2)
        dot.stroke(self.faded())

        for off in self.offsets:
            pts = spiral_points(self.cx, self.cy, turns=1.5, n=60, r_start=5, r_end=max(6, r_end), angle_offset=off)
            for i in range(len(pts) - 1):
                dot.line(pts[i], pts[i + 1])


class MiniSpirals(Shape):
    def __init__(self, cx, cy, colour):
        super().__init__(life=0.7, colour=colour)
        spread = min(dot.width, dot.height) * 0.25
        self.spirals = []

        for _ in range(8):
            sx = cx + random.uniform(-spread, spread)
            sy = cy + random.uniform(-spread, spread)
            off = random.uniform(0, 2 * math.pi)
            self.spirals.append((sx, sy, off))

    def draw(self):
        age = time.time() - self.born
        frac = age / self.life
        r = 5 + 35 * frac

        dot.no_fill()
        dot.set_stroke_weight(1)
        dot.stroke(self.faded())

        for sx, sy, off in self.spirals:
            pts = spiral_points(sx, sy, turns=1.8, n=40, r_start=2, r_end=r, angle_offset=off)
            for i in range(len(pts) - 1):
                dot.line(pts[i], pts[i + 1])


class DriftingBlob(Shape):
    def __init__(self, cx, cy, colour):
        super().__init__(life=2.5, colour=colour)
        self.x = cx + random.uniform(-dot.width * 0.2, dot.width * 0.2)
        self.y = cy + random.uniform(-dot.height * 0.2, dot.height * 0.2)
        self.vx = random.uniform(-25, 25)
        self.vy = random.uniform(-25, 25)
        self.seed = random.random()
        self.base_r = random.uniform(40, 90)

    def draw(self):
        age = time.time() - self.born

        self.x += self.vx / 60
        self.y += self.vy / 60
        self.vx *= 0.99
        self.vy *= 0.99

        r = self.base_r * (1.0 + 0.15 * math.sin(age * 2.1))

        dot.fill(self.faded())
        dot.no_stroke()

        pts = blob_points(self.x, self.y, r, n_points=20, roughness=0.45, seed=self.seed + age * 0.1)
        dot.polygon(pts)


class SpiralExplosion(Shape):
    def __init__(self, cx, cy, colour):
        super().__init__(life=1.6, colour=colour)
        self.cx = cx
        self.cy = cy
        self.n_arms = 14
        self.offsets = [
            2 * math.pi * i / self.n_arms + random.uniform(-0.2, 0.2)
            for i in range(self.n_arms)
        ]

    def draw(self):
        age = time.time() - self.born
        frac = age / self.life
        r_end = min(dot.width, dot.height) * 0.65 * math.sin(frac * math.pi)

        if r_end < 2:
            return

        dot.no_fill()
        dot.set_stroke_weight(2)
        dot.stroke(self.faded())

        for off in self.offsets:
            pts = spiral_points(self.cx, self.cy, turns=2.8, n=80, r_start=3, r_end=r_end, angle_offset=off)
            for i in range(len(pts) - 1):
                dot.line(pts[i], pts[i + 1])


class InstrumentPulse(Shape):
    def __init__(self, instrument):
        colours = {
            "PIANO": (80, 180, 255),
            "BASS": (255, 80, 120),
            "SAX": (255, 190, 60),
        }
        super().__init__(life=0.9, colour=colours.get(instrument, (255, 255, 255)))
        self.instrument = instrument
        self.x = random.uniform(dot.width * 0.2, dot.width * 0.8)
        self.y = random.uniform(dot.height * 0.2, dot.height * 0.8)
        self.radius = random.uniform(20, 55)
        self.seed = random.random()

    def draw(self):
        age = time.time() - self.born
        frac = age / self.life
        r = self.radius + frac * 80

        dot.no_fill()
        dot.set_stroke_weight(2)
        dot.stroke(self.faded())

        if self.instrument == "PIANO":
            dot.circle((self.x, self.y), r)
        elif self.instrument == "BASS":
            pts = blob_points(self.x, self.y, r * 0.6, n_points=18, roughness=0.5, seed=self.seed + age)
            dot.polygon(pts)
        else:
            pts = spiral_points(self.x, self.y, turns=1.2, n=50, r_start=2, r_end=r, angle_offset=self.seed)
            for i in range(len(pts) - 1):
                dot.line(pts[i], pts[i + 1])


class ShapeEngine:
    def __init__(self):
        self.shapes = []
        self.show_skel = True

    def clear(self):
        self.shapes.clear()

    def hit(self, drum_class, cx, cy):
        c = DRUM_COLOURS[drum_class]

        if drum_class == 0:
            self.shapes.append(BlobRing(cx, cy, c))
        elif drum_class == 1:
            self.shapes.append(SpiralBurst(cx, cy, c))
        elif drum_class == 2:
            for _ in range(3):
                self.shapes.append(MiniSpirals(cx, cy, c))
        elif drum_class == 3:
            for _ in range(5):
                self.shapes.append(DriftingBlob(cx, cy, c))
        elif drum_class == 4:
            self.shapes.append(SpiralExplosion(cx, cy, c))

    def instrument_event(self, instrument):
        self.shapes.append(InstrumentPulse(instrument))

    def draw(self):
        self.shapes = [s for s in self.shapes if s.alive]
        for s in self.shapes:
            s.draw()


engine = ShapeEngine()


# ── MEDIAPIPE SETUP ───────────────────────────────────────────────
camera = cv2.VideoCapture(0)
pose_results = None


def on_pose_result(result, output_image, timestamp_ms):
    global pose_results
    pose_results = result


base_options = mp_python.BaseOptions(
    model_asset_path="models/pose_landmarker_full.task",
    delegate=mp_python.BaseOptions.Delegate.CPU
)

options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.LIVE_STREAM,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=on_pose_result
)

pose_detector = mp_vision.PoseLandmarker.create_from_options(options)


# ── STATE ─────────────────────────────────────────────────────────
prev_gesture = None
last_hit_name = ""
last_hit_time = 0
last_drum_cls = 0
last_ensemble_name = ""
last_ensemble_time = 0


def get_gesture(output):
    if output is None:
        return None

    if isinstance(output, (int, float)):
        return int(output)

    if hasattr(output, "__len__"):
        lst = list(output)
        return int(lst[0]) if len(lst) == 1 else lst.index(max(lst))

    return None


def setup():
    gui.setup(dot)

    print("=" * 65)
    print("  JAZZ DRUM LOOPER + MARKOV MACHINE ENSEMBLE")
    print("  Train classes 0-4 for drums. Class 5 can be neutral/rest.")
    print("  R record/re-record | O overdub | U undo overdub")
    print("  P piano Markov | B bass Markov | S sax Markov — playable anytime")
    print("  V skeleton | C clear loop+visuals | UP/DOWN melody speed")
    print("=" * 65)


def draw():
    global prev_gesture
    global pose_results
    global last_hit_name
    global last_hit_time
    global last_drum_cls
    global last_ensemble_name
    global last_ensemble_time

    W = dot.width
    H = dot.height
    cx = W // 2
    cy = H // 2

    dot.background((0, 0, 0))

    engine.draw()

    # Webcam → MediaPipe skeleton
    success, frame = camera.read()
    input_data = [0.0] * (NUM_LANDMARKS * 2)

    if success:
        frame = cv2.resize(frame, (W, H))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        pose_detector.detect_async(mp_image, int(time.time() * 1000))

        if pose_results and pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks[0]

            for i, pt in enumerate(lm):
                input_data[i * 2] = pt.x
                input_data[i * 2 + 1] = pt.y

            if engine.show_skel:
                dot.set_stroke_weight(2)
                dot.stroke((80, 80, 80))
                dot.no_fill()

                for a, b in SKELETON:
                    dot.line(
                        (lm[a].x * W, lm[a].y * H),
                        (lm[b].x * W, lm[b].y * H)
                    )

                joint_colour = DRUM_COLOURS.get(last_drum_cls, (255, 255, 255))
                dot.fill(joint_colour)
                dot.no_stroke()

                for pt in lm:
                    dot.circle((pt.x * W, pt.y * H), 4)

    # IML training / prediction
    learner.new_example(input_data, gui.label)

    # Gesture → drums only
    if learner.output is not None:
        current_gesture = get_gesture(learner.output)

        if current_gesture is not None and current_gesture != prev_gesture:
            if current_gesture in DRUM_SOUNDS:
                play_sound(DRUM_SOUNDS[current_gesture], volume=1.0)

                last_hit_name = DRUM_NAMES[current_gesture]
                last_hit_time = time.time()
                last_drum_cls = current_gesture

                engine.hit(current_gesture, cx, cy)

                if looper.recording or looper.overdubbing:
                    looper.log_hit(current_gesture)

            # Class 5 - use as neutral pose

        prev_gesture = current_gesture

    #Drum loop playback
    for drum_class in looper.update():
        if drum_class in DRUM_SOUNDS:
            play_sound(DRUM_SOUNDS[drum_class], volume=1.0)

            last_hit_name = DRUM_NAMES[drum_class]
            last_hit_time = time.time()
            last_drum_cls = drum_class

            engine.hit(drum_class, cx, cy)

    # Markov ensemble playback: plays whenever P/B/S are on
    events = ensemble.update(loop_is_playing=True)

    for instrument, note_index in events:
        last_ensemble_name = f"{instrument} {note_index}"
        last_ensemble_time = time.time()
        engine.instrument_event(instrument)

    # HUD: drum hit
    if time.time() - last_hit_time < 0.6:
        dot.fill(DRUM_COLOURS.get(last_drum_cls, (255, 255, 255)))
        dot.text(last_hit_name, 20, 45, 28)

    # HUD: ensemble note
    if time.time() - last_ensemble_time < 0.6:
        dot.fill((170, 200, 255))
        dot.text(last_ensemble_name, 20, 78, 18)

    # HUD: looper
    if looper.recording:
        elapsed = time.time() - looper.record_start
        dot.fill((220, 50, 30))
        dot.text(f"REC  {elapsed:.1f}s  |  R to close loop", 20, H - 65, 14)

    elif looper.has_loop:
        progress = looper.progress()
        bar = "|" * int(progress * 20) + "." * (20 - int(progress * 20))

        if looper.overdubbing:
            dot.fill((255, 180, 40))
            status = f"OVERDUB [{bar}]  {len(looper.hits)} hits  |  O stop | U undo"
        else:
            dot.fill((80, 220, 120))
            status = f"LOOP [{bar}]  {len(looper.hits)} hits  |  O overdub | R re-record"

        dot.text(status, 20, H - 65, 14)

    else:
        dot.fill((80, 80, 80))
        dot.text("TRAIN → RUN → R to record drum loop", 20, H - 65, 13)

    # HUD: ensemble status
    dot.fill((150, 150, 150))
    dot.text(
        f"Piano:{'ON' if ensemble.piano_on else 'OFF'}  "
        f"Bass:{'ON' if ensemble.bass_on else 'OFF'}  "
        f"Sax:{'ON' if ensemble.sax_on else 'OFF'}  "
        f"speed={ensemble.interval:.2f}s",
        20,
        H - 35,
        13
    )

    dot.text("Keys: R rec | O overdub | U undo | P/B/S Markov ensemble | C clear", 20, H - 15, 12)

    gui.draw(dot)


# ── KEY HANDLING ──────────────────────────────────────────────────
try:
    def handle_key(key, action, mods):
        action_text = str(action)
        if "PRESS" not in action_text:
            return

        try:
            key_code = int(key)
        except Exception:
            key_code = key

        # Lowercase raw key codes are included because Dorothy sometimes
        # returns numbers like 112 for P instead of dot.keys.P.
        if key == dot.keys.R or key_code in (114, 82):
            looper.toggle()
        elif key == dot.keys.O or key_code in (111, 79):
            looper.toggle_overdub()
        elif key == dot.keys.U or key_code in (117, 85):
            looper.undo_last_overdub()
        elif key == dot.keys.P or key_code in (112, 80):
            ensemble.toggle_piano()
        elif key == dot.keys.B or key_code in (98, 66):
            ensemble.toggle_bass()
        elif key == dot.keys.S or key_code in (115, 83):
            ensemble.toggle_sax()
        elif key == dot.keys.V or key_code in (118, 86):
            engine.show_skel = not engine.show_skel
        elif key == dot.keys.C or key_code in (99, 67):
            looper.reset()
            engine.clear()
            print("Cleared loop and visuals")
        elif key == dot.keys.UP:
            ensemble.faster()
        elif key == dot.keys.DOWN:
            ensemble.slower()

    dot.on_key_press = handle_key

except Exception as e:
    print(f"Key callback not available: {e}")


# ── START ─────────────────────────────────────────────────────────
dot.start_loop(setup, draw)
