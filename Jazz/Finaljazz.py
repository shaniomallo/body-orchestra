

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


#pygame mixer for audio mix
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


#IML classifier 
NUM_LANDMARKS = 33
NUM_CLASSES   = 9  # 1-5 drums, class 0 is neutral/rest

learner = Learner(
    num_inputs=NUM_LANDMARKS * 2,
    num_classes=NUM_CLASSES,
    model_type=CLASSIFICATION,
    model_name="skeleton"
)
gui = LearnerGUI(learner)


#drum samples 
DRUM_SOUNDS = {
    1: load_sound("audio/kick.wav"),
    2: load_sound("audio/snare.wav"),
    3: load_sound("audio/hihat_closed.wav"),
    4: load_sound("audio/hihat_open.wav"),
    5: load_sound("audio/crash.wav"),
}

DRUM_NAMES = {
    1: "KICK",
    2: "SNARE",
    3: "HH CLOSED",
    4: "HH OPEN",
    5: "CRASH",
}

DRUM_COLOURS = {
    1: (220, 50, 30),
    2: (240, 240, 240),
    3: (220, 180, 40),
    4: (100, 220, 80),
    5: (140, 80, 220),
}

# Number row / numpad keys mapped to drum classes 1-5
DRUM_KEY_MAP = {
    49: 1,  # '1' -> kick
    50: 2,  # '2' -> snare
    51: 3,  # '3' -> hihat closed
    52: 4,  # '4' -> hihat open
    53: 5,  # '5' -> crash
}

#sound banks- notes for Piano, Bass and Saxophone 
#map notes to te audios
PIANO_BANK = {
    "Bb4": load_sound("audio/piano_Bb4.wav"),
    "C4":  load_sound("audio/piano_C4.wav"),
    "C5":  load_sound("audio/piano_C5.wav"),
    "Eb4": load_sound("audio/piano_Eb4.wav"),
    "F4":  load_sound("audio/piano_F4.wav"),
    "G4":  load_sound("audio/piano_G4.wav"),
    "Gb4": load_sound("audio/piano_Gb4.wav"),
}

BASS_BANK = {
    "A2":  load_sound("audio/bass_A2.wav"),
    "B2":  load_sound("audio/bass_B2.wav"),
    "Bb2": load_sound("audio/bass_Bb2.wav"),
    "C3":  load_sound("audio/bass_C3.wav"),
    "E2":  load_sound("audio/bass_E2.wav"),
    "Eb2": load_sound("audio/bass_Eb2.wav"),
    "F2":  load_sound("audio/bass_F2.wav"),
    "G2":  load_sound("audio/bass_G2.wav"),
    "Gb2": load_sound("audio/bass_Gb2.wav"),
}

SAX_BANK = {
    "A4":  load_sound("audio/sax_A4.wav"),
    "B4":  load_sound("audio/sax_B4.wav"),
    "Bb4": load_sound("audio/sax_Bb4.wav"),
    "C4":  load_sound("audio/sax_C4.wav"),
    "D4":  load_sound("audio/sax_D4.wav"),
    "E4":  load_sound("audio/sax_E4.wav"),
    "Eb4": load_sound("audio/sax_Eb4.wav"),
    "F4":  load_sound("audio/sax_F4.wav"),
    "Gb4": load_sound("audio/sax_Gb4.wav"),
}


#Markov training model 
#sequence obtained by combining jazz blues notes with the help of LLM
PIANO_TRAINING = [
    "C4", "Eb4", "F4", "Gb4", "G4", "Bb4", "C5",
    "Bb4", "G4", "Gb4", "F4", "Eb4", "C4",
    "Eb4", "G4", "Bb4", "G4", "F4", "Eb4",
    "C4", "F4", "G4", "Bb4", "G4", "Eb4", "C4",
]

BASS_TRAINING = [
    "C3", "E2", "G2", "Bb2",
    "C3", "Bb2", "G2", "E2",
    "F2", "G2", "Bb2", "C3",
    "Bb2", "G2", "Gb2", "F2",
    "E2", "Eb2", "E2", "G2", "Bb2", "C3",
]

SAX_TRAINING = [
    "C4", "D4", "Eb4", "F4", "Gb4", "G4",
    "A4", "Bb4", "B4", "Bb4", "A4",
    "Gb4", "F4", "Eb4", "D4", "C4",
    "Eb4", "F4", "G4", "Bb4", "A4", "G4", "F4",
]


#Markov model- to create melody 
class MarkovMelody:
    """
    First-order Markov chain built from a training sequence of note names.
    Each note maps to a list of notes that followed it in the training data.
    next_note() picks randomly from that list, producing jazz-idiomatic movement.
    """

    def __init__(self, training_notes, name="Markov"):
        self.name           = name
        self.training_notes = [n for n in training_notes if n]

        if len(self.training_notes) < 2:
            raise ValueError(f"{name} needs at least 2 training notes")

        # Build transition table from consecutive note pairs
        self.model = {}
        for a, b in zip(self.training_notes[:-1], self.training_notes[1:]):
            if a not in self.model:
                self.model[a] = []
            self.model[a].append(b)

        self.current = random.choice(self.training_notes)

    def reset(self):
        self.current = random.choice(self.training_notes)

    def next_note(self):
        choices = self.model.get(self.current)
        if not choices:
            # Dead end - jump to a random training note
            self.current = random.choice(self.training_notes)
        else:
            self.current = random.choice(choices)
        return self.current


piano_model = MarkovMelody(PIANO_TRAINING, "Piano Markov")
bass_model  = MarkovMelody(BASS_TRAINING,  "Bass Markov")
sax_model   = MarkovMelody(SAX_TRAINING,   "Sax Markov")


#Ensemblegenerators to control Piano, Bass and saxophone sounds 
class EnsembleGenerator:
    """P/B/S toggle three independent trained Markov melody generators."""

    def __init__(self):
        self.piano_on        = False
        self.bass_on         = False
        self.sax_on          = False
        self.interval        = 0.45
        self.last_time       = 0.0
        self.sax_probability = 0.55

    def reset_steps(self):
        piano_model.reset()
        bass_model.reset()
        sax_model.reset()
        self.last_time = time.time()

    def faster(self):
        self.interval = max(0.18, self.interval - 0.05)
        print(f"Markov interval: {self.interval:.2f}s")

    def slower(self):
        self.interval = min(1.2, self.interval + 0.05)
        print(f"Markov interval: {self.interval:.2f}s")

    def toggle_piano(self):
        self.piano_on  = not self.piano_on
        piano_model.reset()
        self.last_time = time.time()
        print(f"PIANO MARKOV {'ON' if self.piano_on else 'OFF'}")

    def toggle_bass(self):
        self.bass_on   = not self.bass_on
        bass_model.reset()
        self.last_time = time.time()
        print(f"BASS MARKOV {'ON' if self.bass_on else 'OFF'}")

    def toggle_sax(self):
        self.sax_on    = not self.sax_on
        sax_model.reset()
        self.last_time = time.time()
        print(f"SAX MARKOV {'ON' if self.sax_on else 'OFF'}")

    def update(self, loop_is_playing):
        if not loop_is_playing:
            return []
        if not (self.piano_on or self.bass_on or self.sax_on):
            return []

        now = time.time()
        if now - self.last_time < self.interval:
            return []

        events = []

        if self.piano_on:
            note = piano_model.next_note()
            play_sound(PIANO_BANK.get(note), volume=0.55)
            events.append(("PIANO", note))
            print(f"PIANO: {note}")

        if self.bass_on:
            note = bass_model.next_note()
            play_sound(BASS_BANK.get(note), volume=0.75)
            events.append(("BASS", note))
            print(f"BASS: {note}")

        if self.sax_on and random.random() < self.sax_probability:
            note = sax_model.next_note()
            play_sound(SAX_BANK.get(note), volume=0.8)
            events.append(("SAX", note))
            print(f"SAX: {note}")

        self.last_time = now
        return events


ensemble = EnsembleGenerator()


#skeleton pose 
SKELETON = [
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]


#creating loop with overdub and undo 
class Looper:
    def __init__(self):
        self.reset()

    def reset(self):
        self.recording      = False
        self.overdubbing    = False
        self.has_loop       = False
        self.record_start   = None
        self.loop_start     = None
        self.loop_length    = 0.0
        self.hits           = []
        self._fired         = set()
        self.current_layer  = []
        self.overdub_layers = []

    def start_recording(self):
        self.recording      = True
        self.overdubbing    = False
        self.has_loop       = False
        self.hits           = []
        self.current_layer  = []
        self.overdub_layers = []
        self.record_start   = time.time()
        self.loop_start     = None
        self.loop_length    = 0.0
        self._fired         = set()
        ensemble.reset_steps()
        print("● REC — play drums, press R to close loop")

    def stop_recording(self):
        if self.record_start:
            self.loop_length = time.time() - self.record_start

        if self.loop_length < 0.3:
            print("Loop too short — try again")
            self.reset()
            return

        self.recording   = False
        self.has_loop    = True
        self.overdubbing = False
        self.loop_start  = time.time()
        self._fired      = set()
        print(f"■ LOOP {self.loop_length:.2f}s | {len(self.hits)} hits")

    def toggle(self):
        if not self.recording and not self.has_loop:
            self.start_recording()
        elif self.recording:
            self.stop_recording()
        else:
            self.reset()
            self.start_recording()

    def toggle_overdub(self):
        if not self.has_loop:
            print("No loop yet — press R to record first")
            return

        self.overdubbing = not self.overdubbing

        if self.overdubbing:
            self.current_layer = []
            print("● OVERDUB ON")
        else:
            if self.current_layer:
                self.overdub_layers.append(self.current_layer[:])
                print(f"○ OVERDUB SAVED | {len(self.current_layer)} hits")
            else:
                print("○ OVERDUB OFF | no hits added")
            self.current_layer = []

    def undo_last_overdub(self):
        if self.overdubbing:
            for hit in self.current_layer:
                if hit in self.hits:
                    self.hits.remove(hit)
            removed = len(self.current_layer)
            self.current_layer = []
            self.overdubbing   = False
            print(f"UNDO current overdub | removed {removed} hits")
            return

        if not self.overdub_layers:
            print("No overdub layer to undo")
            return

        layer = self.overdub_layers.pop()
        for hit in layer:
            if hit in self.hits:
                self.hits.remove(hit)
        self._fired = set()
        print(f"UNDO last overdub | removed {len(layer)} hits")

    def log_hit(self, drum_class):
        now = time.time()

        if self.recording and self.record_start:
            self.hits.append((drum_class, now - self.record_start))
            return

        if self.overdubbing and self.has_loop and self.loop_start:
            loop_pos = (now - self.loop_start) % self.loop_length
            hit = (drum_class, loop_pos)
            self.hits.append(hit)
            self.current_layer.append(hit)
            self.hits.sort(key=lambda h: h[1])
            print(f"Overdub hit added at {loop_pos:.2f}s")

    def progress(self):
        if not self.has_loop or self.loop_length <= 0:
            return 0.0
        return ((time.time() - self.loop_start) % self.loop_length) / self.loop_length

    def update(self):
        out = []

        if not self.has_loop or not self.hits or self.loop_length < 0.1:
            return out

        elapsed  = time.time() - self.loop_start
        loop_pos = elapsed % self.loop_length
        cycle    = int(elapsed / self.loop_length)

        WINDOW = 0.025

        for i, (drum_class, offset) in enumerate(self.hits):
            hit_key = (cycle, i)

            if hit_key in self._fired:
                continue

            normal_hit = abs(loop_pos - offset) < WINDOW
            wrap_hit   = offset > self.loop_length - WINDOW and loop_pos < WINDOW

            if normal_hit or wrap_hit:
                out.append(drum_class)
                self._fired.add(hit_key)

        self._fired = {key for key in self._fired if key[0] >= cycle - 1}
        return out


looper = Looper()


#creating shapes for animated visuals
#BlobRing, SpiralBurst, MiniSpirals, and SpiralExplosion
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
        t      = i / n
        angle  = angle_offset + t * turns * 2 * math.pi
        radius = r_start + (r_end - r_start) * t
        pts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return pts


class Shape:
    def __init__(self, life, colour):
        self.born   = time.time()
        self.life   = life
        self.colour = colour

    @property
    def alive(self):
        return (time.time() - self.born) < self.life

    @property
    def alpha(self):
        return max(0.0, 1.0 - (time.time() - self.born) / self.life)

    def faded(self, base_alpha=1.0):
        a       = self.alpha * base_alpha
        r, g, b = self.colour
        return (int(r * a), int(g * a), int(b * a))

    def draw(self):
        pass


class BlobRing(Shape):
    def __init__(self, cx, cy, colour):
        super().__init__(life=1.4, colour=colour)
        self.cx      = cx
        self.cy      = cy
        self.n_blobs = 10
        self.seeds   = [random.random() for _ in range(self.n_blobs)]
        self.max_r   = min(dot.width, dot.height) * 0.45

    def draw(self):
        age    = time.time() - self.born
        frac   = age / self.life
        ring_r = self.max_r * frac
        blob_r = 28 * math.sin(frac * math.pi)
        if blob_r < 1:
            return
        dot.fill(self.faded())
        dot.no_stroke()
        for i in range(self.n_blobs):
            angle = 2 * math.pi * i / self.n_blobs
            bx    = self.cx + ring_r * math.cos(angle)
            by    = self.cy + ring_r * math.sin(angle)
            pts   = blob_points(bx, by, blob_r, n_points=14, roughness=0.4, seed=self.seeds[i])
            dot.polygon(pts)


class SpiralBurst(Shape):
    def __init__(self, cx, cy, colour):
        super().__init__(life=0.9, colour=colour)
        self.cx      = cx
        self.cy      = cy
        self.n_arms  = 6
        self.offsets = [random.uniform(0, 2 * math.pi) for _ in range(self.n_arms)]

    def draw(self):
        age   = time.time() - self.born
        frac  = age / self.life
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
        spread       = min(dot.width, dot.height) * 0.25
        self.spirals = []
        for _ in range(8):
            sx  = cx + random.uniform(-spread, spread)
            sy  = cy + random.uniform(-spread, spread)
            off = random.uniform(0, 2 * math.pi)
            self.spirals.append((sx, sy, off))

    def draw(self):
        age  = time.time() - self.born
        frac = age / self.life
        r    = 5 + 35 * frac
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
        self.x      = cx + random.uniform(-dot.width * 0.2, dot.width * 0.2)
        self.y      = cy + random.uniform(-dot.height * 0.2, dot.height * 0.2)
        self.vx     = random.uniform(-25, 25)
        self.vy     = random.uniform(-25, 25)
        self.seed   = random.random()
        self.base_r = random.uniform(40, 90)

    def draw(self):
        age      = time.time() - self.born
        self.x  += self.vx / 60
        self.y  += self.vy / 60
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
        self.cx      = cx
        self.cy      = cy
        self.n_arms  = 14
        self.offsets = [
            2 * math.pi * i / self.n_arms + random.uniform(-0.2, 0.2)
            for i in range(self.n_arms)
        ]

    def draw(self):
        age   = time.time() - self.born
        frac  = age / self.life
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
            "BASS":  (255, 80, 120),
            "SAX":   (255, 190, 60),
        }
        super().__init__(life=0.9, colour=colours.get(instrument, (255, 255, 255)))
        self.instrument = instrument
        self.x          = random.uniform(dot.width * 0.2, dot.width * 0.8)
        self.y          = random.uniform(dot.height * 0.2, dot.height * 0.8)
        self.radius     = random.uniform(20, 55)
        self.seed       = random.random()

    def draw(self):
        age  = time.time() - self.born
        frac = age / self.life
        r    = self.radius + frac * 80
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
        self.shapes    = []
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
# Intro screen
show_intro = True
INTRO_IMAGE_PATH = "images/intro.jpg"


#Mediapipe setup
camera       = cv2.VideoCapture(0)
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



prev_gesture       = None
last_hit_name      = ""
last_hit_time      = 0
last_drum_cls      = 0
last_ensemble_name = ""
last_ensemble_time = 0

#Shared centre coords so handle_key can place visuals at screen centre
cx = 0
cy = 0


def get_gesture(output):
    if output is None:
        return None
    if isinstance(output, (int, float)):
        return int(output)
    if hasattr(output, "__len__"):
        lst = list(output)
        return int(lst[0]) if len(lst) == 1 else lst.index(max(lst))
    return None


def fire_drum(drum_class):
    """Shared helper - plays a drum hit, updates HUD, visuals, and looper.
    Called from both pose detection and keyboard handler."""
    global last_hit_name, last_hit_time, last_drum_cls

    play_sound(DRUM_SOUNDS[drum_class], volume=1.0)
    last_hit_name = DRUM_NAMES[drum_class]
    last_hit_time = time.time()
    last_drum_cls = drum_class
    engine.hit(drum_class, cx, cy)

    if looper.recording or looper.overdubbing:
        looper.log_hit(drum_class)


def setup():
    gui.setup(dot)
    print("=" * 70)
    print("  JAZZ DRUM LOOPER + TRAINED MARKOV ENSEMBLE + OVERDUB")
    print("  Pose classes 0-4 trigger drums. Class 5 = neutral/rest.")
    print("  Keys 1-5 also trigger drums directly.")
    print("  R record/close/re-record | O overdub | U undo | C clear")
    print("  P piano Markov | B bass Markov | S sax Markov")
    print("  V skeleton | UP/DOWN melody speed")
    print("=" * 70)


def draw():
    
    global prev_gesture, pose_results
    global last_hit_name, last_hit_time, last_drum_cls
    global last_ensemble_name, last_ensemble_time
    global cx, cy

    W  = dot.width
    H  = dot.height
    cx = W // 2
    cy = H // 2

    dot.background((0, 0, 0))
    engine.draw()
    

    # Webcam → MediaPipe skeleton
    success, frame = camera.read()
    input_data = [0.0] * (NUM_LANDMARKS * 2)

    if success:
        frame     = cv2.resize(frame, (W, H))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        pose_detector.detect_async(mp_image, int(time.time() * 1000))

        if pose_results and pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks[0]

            for i, pt in enumerate(lm):
                input_data[i * 2]     = pt.x
                input_data[i * 2 + 1] = pt.y

            if engine.show_skel:
                dot.set_stroke_weight(2)
                dot.stroke((80, 80, 80))
                dot.no_fill()
                for a, b in SKELETON:
                    dot.line((lm[a].x * W, lm[a].y * H), (lm[b].x * W, lm[b].y * H))
                joint_colour = DRUM_COLOURS.get(last_drum_cls, (255, 255, 255))
                dot.fill(joint_colour)
                dot.no_stroke()
                for pt in lm:
                    dot.circle((pt.x * W, pt.y * H), 4)

    # IML training / prediction
    learner.new_example(input_data, gui.label)

    
    # Pose gesture → drums + instrument triggers
    if learner.output is not None:
        current_gesture = get_gesture(learner.output)
        if current_gesture is not None and current_gesture != prev_gesture:
            if current_gesture in DRUM_SOUNDS:
                fire_drum(current_gesture)
            elif current_gesture == 6:
                ensemble.toggle_piano()
            elif current_gesture == 7:
                ensemble.toggle_bass()
            elif current_gesture == 8:
                ensemble.toggle_sax()
        prev_gesture = current_gesture

    # Drum loop playback
    for drum_class in looper.update():
        if drum_class in DRUM_SOUNDS:
            fire_drum(drum_class)

    # Markov ensemble - only plays while loop is active
    events = ensemble.update(loop_is_playing=looper.has_loop and not looper.recording)
    for instrument, note in events:
        last_ensemble_name = f"{instrument} {note}"
        last_ensemble_time = time.time()
        engine.instrument_event(instrument)

    # HUD: drum hit
    if time.time() - last_hit_time < 0.6:
        dot.fill(DRUM_COLOURS.get(last_drum_cls, (255, 255, 255)))
        dot.text(last_hit_name, 20, 45, 28)

    # HUD: Markov note
    if time.time() - last_ensemble_time < 0.6:
        dot.fill((170, 200, 255))
        dot.text(last_ensemble_name, 20, 78, 18)

    # HUD: looper
    if looper.recording:
        elapsed = time.time() - looper.record_start
        dot.fill((220, 50, 30))
        dot.text(f"REC  {elapsed:.1f}s  |  R to close loop", 20, H - 65, 14)

    elif looper.overdubbing:
        progress = looper.progress()
        bar      = "|" * int(progress * 20) + "." * (20 - int(progress * 20))
        dot.fill((255, 180, 40))
        dot.text(
            f"OVERDUB [{bar}]  {len(looper.current_layer)} new hits  |  O save | U undo",
            20, H - 65, 14
        )

    elif looper.has_loop:
        progress = looper.progress()
        bar      = "|" * int(progress * 20) + "." * (20 - int(progress * 20))
        dot.fill((80, 220, 120))
        dot.text(
            f"LOOP [{bar}]  {len(looper.hits)} hits  |  O overdub | U undo | R re-record",
            20, H - 65, 14
        )

    else:
        dot.fill((80, 80, 80))
        dot.text("TRAIN → RUN → R to record drum loop", 20, H - 65, 13)

    # HUD: ensemble status
    dot.fill((150, 150, 150))
    dot.text(
        f"Markov  Piano:{'ON' if ensemble.piano_on else 'OFF'}  "
        f"Bass:{'ON' if ensemble.bass_on else 'OFF'}  "
        f"Sax:{'ON' if ensemble.sax_on else 'OFF'}  "
        f"speed={ensemble.interval:.2f}s",
        20, H - 35, 13
    )
    dot.text(
        "Keys: 1-5 drums | R record | O overdub | U undo | P piano | B bass | S sax",
        20, H - 15, 12
    )

    gui.draw(dot)


#key handles 
try:
    def handle_key(key, action, mods):
        print(key, action, mods)

        if "PRESS" not in str(action):
            return

        try:
            key_code = int(key)
        except Exception:
            key_code = key

        # Drum keys 1-5
        if key_code in DRUM_KEY_MAP:
            fire_drum(DRUM_KEY_MAP[key_code])
            return

        # Looper / ensemble controls
        if key_code == 114 or key == dot.keys.R:
            looper.toggle()

        elif key_code == 111 or key == dot.keys.O:
            looper.toggle_overdub()

        elif key_code == 117 or key == dot.keys.U:
            looper.undo_last_overdub()

        elif key_code == 112 or key == dot.keys.P:
            ensemble.toggle_piano()

        elif key_code == 98 or key == dot.keys.B:
            ensemble.toggle_bass()

        elif key_code == 115 or key == dot.keys.S:
            ensemble.toggle_sax()

        elif key_code == 118 or key == dot.keys.V:
            engine.show_skel = not engine.show_skel

        elif key_code == 99 or key == dot.keys.C:
            engine.clear()
            looper.reset()
            print("CLEARED LOOP + VISUALS")

        elif key == dot.keys.UP:
            ensemble.faster()

        elif key == dot.keys.DOWN:
            ensemble.slower()

    dot.on_key_press = handle_key

except Exception as e:
    print(f"Key callback not available: {e}")


#start
dot.start_loop(setup, draw)
