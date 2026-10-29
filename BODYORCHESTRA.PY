"""

Main idea:
The body does not trigger single notes. Instead, the regression model outputs
two continuous values: energy and colour.

energy = density + volume
colour = register + melodic brightness

Manual mode with arrow keys is included for rehearsal and live control.
"""


# Libraries used
# cv2: webcam capture
# MediaPipe: full-body pose tracking
# Dorothy: live visual canvas
# Learner: IML classification/regression framework
# pygame.mixer: real-time audio sample playback
# math/random/time: animation, timing and generative decisions
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

# NOTE: we now want REGRESSION, not CLASSIFICATION.
from learner import Learner, LearnerGUI, REGRESSION

dot = Dorothy()

# Audio setup
# Adapted from Week 4 audio/generative sound ideas.
# pygame.mixer lets several piano samples ring at the same time.
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.init()
pygame.mixer.set_num_channels(64)

# Let each note ring and decay naturally on a free channel (pygame picks
# one).  An earlier version hard-cut each note on a fixed channel, which
# chopped the previous note mid-ring and produced a click on every note.
# Now nothing is chopped, so the piano decays like a piano.


def load_sound(path):
    if not os.path.exists(path):
        return None
    try:
        return pygame.mixer.Sound(path)
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return None


# Piano sample bank
# Uses MIDI-numbered piano samples prepared before running the performance.
# This supports the later MIDI/generative-music direction of the project.
PIANO_DIR = "audio/piano"
PIANO = {}          # midi -> Sound
for midi in range(33, 97):          # A1 .. C7, plenty of range for 3 voices
    s = load_sound(os.path.join(PIANO_DIR, f"piano_{midi}.wav"))
    if s is not None:
        PIANO[midi] = s

if not PIANO:
    print("!! No piano samples found in", PIANO_DIR,
          "- run prep_piano_samples.py first.")


def nearest_sample(midi):
    """Return a Sound for the requested midi, or the closest one we have."""
    if midi in PIANO:
        return PIANO[midi]
    if not PIANO:
        return None
    closest = min(PIANO.keys(), key=lambda m: abs(m - midi))
    return PIANO[closest]


def play_voice(midi, volume):
    snd = nearest_sample(midi)
    if snd is None:
        return
    snd.set_volume(max(0.0, min(1.0, volume)))
    snd.play(fade_ms=6)        # tiny fade-in removes the onset click


# Shared beat clock
# Added after V1 because independent timers made the generated music feel messy.
# Every voice now plays on the same 16th-note grid.
BPM           = 84
BEATS_PER_BAR = 4
NUM_BARS      = 4
STEPS_PER_BAR = 16          # 16th-note grid

def beat_dur():  return 60.0 / BPM
def step_dur():  return beat_dur() / 4.0     # one 16th


# Harmonic structure
# Developed from the Markov/generative music experiments.
# A fixed chord spine keeps the generated notes musically coherent.
SCALE_PCS = [0, 2, 4, 5, 7, 9, 11]           # C major scale, pitch classes

# 4-bar progression, one chord per bar:  Am  F  C  G
# Each entry: (name, root pitch-class, set of chord-tone pitch classes)
SPINE = [
    ("Am", 9, {9, 0, 4}),
    ("F",  5, {5, 9, 0}),
    ("C",  0, {0, 4, 7}),
    ("G",  7, {7, 11, 2}),
]

# register windows (MIDI) for each voice
BASS_LO,   BASS_HI,   BASS_PREF   = 36, 52, 45
COMP_LO,   COMP_HI,   COMP_PREF   = 52, 67, 60
MEL_LO,    MEL_HI,    MEL_BASE    = 60, 88, 67


def place_pc(pc, lo, hi, prefer):
    """Lowest-cost MIDI note of pitch-class pc inside [lo,hi], near prefer."""
    cands = [m for m in range(lo, hi + 1) if m % 12 == pc]
    if not cands:
        return prefer
    return min(cands, key=lambda m: abs(m - prefer))


def snap_to_pcs(midi, allowed_pcs, lo, hi):
    """Nearest MIDI in [lo,hi] whose pitch-class is allowed."""
    best, best_d = None, 999
    for m in range(lo, hi + 1):
        if m % 12 in allowed_pcs:
            d = abs(m - midi)
            if d < best_d:
                best, best_d = m, d
    return best if best is not None else midi


# Markov melody
# Adapted from the module's generative music / Markov work.
# The model learns simple note movement patterns and generates a melodic contour.
class MarkovMelody:
    def __init__(self, tokens, name="markov"):
        self.name = name
        self.tokens = list(tokens)
        self.model = {}
        for a, b in zip(self.tokens[:-1], self.tokens[1:]):
            self.model.setdefault(a, []).append(b)
        self.current = random.choice(self.tokens)

    def next(self):
        choices = self.model.get(self.current)
        self.current = random.choice(choices) if choices else random.choice(self.tokens)
        return self.current


# a gentle diatonic contour as scale degrees (0 = tonic, 7 = octave up)
MELODY_CONTOUR = [0, 1, 2, 4, 2, 1, 0, 2, 4, 5, 4, 2,
                  4, 5, 7, 5, 4, 2, 0, 1, 2, 4, 2, 0]
melody_markov = MarkovMelody(MELODY_CONTOUR, "melody")


def degree_to_midi(degree, base=MEL_BASE):
    """Map a scale degree to a MIDI note in C major above `base`."""
    octave, idx = divmod(degree, 7)
    return base + 12 * octave + (SCALE_PCS[idx] - SCALE_PCS[0])


# Rhythmic patterns
# These probability patterns decide when bass, chords and melody may play.
# This gives structure instead of fully random note triggering.
def _pattern(strong, light, weak):
    p = [weak] * 16
    for i in strong: p[i] = 0.95
    for i in light:  p[i] = 0.55
    return p

BASS_PATTERN = _pattern(strong=[0, 4, 8, 12], light=[2, 6, 10, 14], weak=0.06)
COMP_PATTERN = _pattern(strong=[2, 6, 10, 14], light=[0, 8],        weak=0.10)
MEL_PATTERN  = _pattern(strong=[0, 6, 8, 14],  light=[3, 11],       weak=0.18)


def clamp01(x):
    return max(0.0, min(1.0, x))


# Week 3 / Regression idea
# The body is mapped to two smooth outputs instead of fixed classes.
# energy controls density and loudness.
# colour controls register and brightness.
class Body:
    def __init__(self):
        self.energy = 0.0       # arms low -> high : sparse/soft -> busy/loud
        self.colour = 0.5       # lean left -> right : warm/low -> bright/high
        self._s_energy = 0.0    # smoothed values actually used by the engine
        self._s_colour = 0.5

    def update_smoothed(self, smooth=0.15):
        self._s_energy += smooth * (self.energy - self._s_energy)
        self._s_colour += smooth * (self.colour - self._s_colour)

    @property
    def e(self): return clamp01(self._s_energy)
    @property
    def c(self): return clamp01(self._s_colour)

body = Body()


# Generative music engine
# Final V2 design: the AI trio plays autonomously on the beat grid.
# The performer conducts the behaviour using energy and colour.
class Engine:
    def __init__(self):
        self.running   = False
        self.start_t   = 0.0
        self.last_step = -1
        self.events    = []     # (voice, midi) produced this frame, for visuals

    def start(self):
        self.running   = True
        self.start_t   = time.time()
        self.last_step = -1
        print("BAND ON")

    def stop(self):
        self.running = False
        pygame.mixer.fadeout(200)
        print("BAND OFF")

    def toggle(self):
        self.stop() if self.running else self.start()

    def update(self):
        self.events = []
        if not self.running:
            return

        step = int((time.time() - self.start_t) / step_dur())
        if step == self.last_step:
            return
        self.last_step = step
        self._play_step(step)

    def _play_step(self, step):
        step_in_bar = step % STEPS_PER_BAR
        bar         = (step // STEPS_PER_BAR) % NUM_BARS
        name, root, chord_pcs = SPINE[bar]

        e = body.e
        c = body.c
        if e < 0.04:
            return                               # arms down = silence

        density = 0.35 + 0.85 * e                # overall busyness
        master  = 0.18 + 0.82 * e                # overall loudness
        fifth   = (root + 7) % 12

        # BASS: root, sometimes the fifth, on the strong beats 
        if random.random() < BASS_PATTERN[step_in_bar] * density:
            pc   = root if random.random() < 0.75 else fifth
            midi = place_pc(pc, BASS_LO, BASS_HI, BASS_PREF)
            play_voice(midi, master * 0.55)
            self.events.append(("BASS", midi))

        # COMP: chord tones on the offbeats, fades in with energy 
        comp_gain = clamp01((e - 0.18) / 0.45)
        if comp_gain > 0 and random.random() < COMP_PATTERN[step_in_bar] * density:
            pc   = random.choice(tuple(chord_pcs))
            midi = place_pc(pc, COMP_LO, COMP_HI, COMP_PREF)
            play_voice(midi, master * 0.42 * (0.4 + 0.6 * comp_gain))
            self.events.append(("COMP", midi))

        # MELODY: Markov line, register lifted by colour 
        # colour brings the melody in and raises it; warm side drops it out
        mel_gain = clamp01((e - 0.30) / 0.5) * clamp01(0.15 + 1.1 * c)
        if mel_gain > 0.05 and random.random() < MEL_PATTERN[step_in_bar] * density:
            degree = melody_markov.next()
            midi   = degree_to_midi(degree)
            midi  += 12 if c > 0.6 else (0 if c > 0.3 else -12)   # colour -> octave
            # land on a chord tone on the bar's strong points, else stay in scale
            if step_in_bar in (0, 8):
                midi = snap_to_pcs(midi, chord_pcs, MEL_LO, MEL_HI)
            else:
                midi = snap_to_pcs(midi, set(SCALE_PCS), MEL_LO, MEL_HI)
            play_voice(midi, master * 0.5 * mel_gain)
            self.events.append(("MELODY", midi))

engine = Engine()


# Week 1 + Week 3 / Learner regression model
# Adapted from Dorothy + Learner IML workflow.
# MediaPipe landmarks are the inputs; energy and colour are the outputs.
NUM_LANDMARKS = 33

# ASSUMPTION about learner.py's regression API: it takes `num_outputs`.
# If yours differs, this is the ONE line to change.
learner = Learner(
    num_inputs=NUM_LANDMARKS * 2,
    num_outputs=2,                 # [energy, colour]
    model_type=REGRESSION,
    model_name="conductor",
)
gui = LearnerGUI(learner, x=20, y=105, btn_w=150, btn_h=32)

# Performance training poses
# These five anchor poses define the regression control space.
# 1 = arms down, 2 = arms up, 3 = lean left, 4 = lean right, 5 = neutral.
# The model interpolates between these examples during performance.
ANCHORS = {
    49: [0.00, 0.50],   # '1' low / compact   (arms down)
    50: [1.00, 0.50],   # '2' high / peak     (arms up)
    51: [0.50, 0.00],   # '3' warm / left     (lean left)
    52: [0.50, 1.00],   # '4' bright / right  (lean right)
    53: [0.35, 0.50],   # '5' neutral rest
}
current_target = [0.5, 0.5]
manual_mode    = False

# Keyboard training fallback:
# Z = record on/off, X = train, N = run/stop.
# This works even if the LearnerGUI buttons are not visible.


# Week 5 / Audio-reactive visuals
# Dorothy draws simple visual pulses from musical events.
# Higher notes appear higher on screen, making the music easier to read visually.
VOICE_COL = {"BASS": (90, 130, 240), "COMP": (120, 200, 170), "MELODY": (230, 170, 90)}

class Pulse:
    def __init__(self, voice, midi):
        self.born = time.time()
        self.life = 1.2
        self.col  = VOICE_COL.get(voice, (255, 255, 255))
        self.x    = random.uniform(0.25, 0.75)
        self.y    = 1.0 - (midi - 33) / (96 - 33)     # low notes low, high high
        self.r0   = 18 + (24 if voice == "MELODY" else 0)

    @property
    def alive(self): return (time.time() - self.born) < self.life

    def draw(self):
        age  = time.time() - self.born
        frac = age / self.life
        a    = max(0.0, 1.0 - frac)
        r, g, b = self.col
        dot.no_fill()
        dot.set_stroke_weight(2)
        dot.stroke((int(r * a), int(g * a), int(b * a)))
        dot.circle((self.x * dot.width, self.y * dot.height), self.r0 + frac * 70)

pulses = []

SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
            (24, 26), (26, 28)]
show_skel = True


# Week 1 / MediaPipe full-body tracking
# OpenCV captures the webcam.
# MediaPipe extracts 33 landmarks, giving 66 input values for the model.
camera       = cv2.VideoCapture(0)
pose_results = None

def on_pose_result(result, output_image, timestamp_ms):
    global pose_results
    pose_results = result

base_options = mp_python.BaseOptions(
    model_asset_path="models/pose_landmarker_full.task",
    delegate=mp_python.BaseOptions.Delegate.CPU,
)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.LIVE_STREAM,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=on_pose_result,
)
pose_detector = mp_vision.PoseLandmarker.create_from_options(options)


# Dorothy performance loop
# setup() creates the Learner GUI.
# draw() updates camera input, ML prediction, music and visuals every frame.
def setup():
    gui.setup(dot)
    print("GUI CREATED")
    print("=" * 64)
    print("  CONDUCTOR : generative piano trio")
    print("  SPACE start/stop | M manual mode | arrows energy/colour")
    print("  Train: strike pose -> press 1-5 -> press Z to record -> hold -> Z stop")
    print("  Then press X to TRAIN and N to RUN. GUI buttons also work if visible.")
    print("=" * 64)


def read_dials_from_model(input_data):
    """Pull [energy, colour] out of the regression model, smoothed."""
    learner.new_example(input_data, current_target)   # GUI gates recording
    out = learner.output
    if out is not None and hasattr(out, "__len__") and len(out) >= 2:
        body.energy = clamp01(float(out[0]))
        body.colour = clamp01(float(out[1]))



def draw_training_panel(W, H):
    """Always-visible text fallback for training, even if Dorothy buttons are hard to see."""
    panel_x = 20
    panel_y = 92

    dot.fill((8, 12, 18))
    dot.no_stroke()
    try:
        dot.rect((panel_x - 8, panel_y - 8, 330, 92))
    except Exception:
        pass

    dot.fill((210, 230, 240))
    dot.text("TRAINING CONTROLS", panel_x, panel_y + 12, 13)

    dot.fill((170, 190, 200))
    dot.text(
        f"target: energy {current_target[0]:.2f}  colour {current_target[1]:.2f}",
        panel_x,
        panel_y + 32,
        12
    )

    state = []
    if learner.recording:
        state.append("RECORDING")
    if learner.trained:
        state.append("TRAINED")
    if learner.running:
        state.append("RUNNING")
    state_text = " | ".join(state) if state else "idle"

    dot.text(
        f"examples: {learner.num_examples()}  status: {state_text}",
        panel_x,
        panel_y + 50,
        12
    )

    dot.fill((140, 170, 190))
    dot.text(
        "1-5 choose pose target  |  Z record  X train  N run",
        panel_x,
        panel_y + 70,
        11
    )


def draw():
    global pose_results
    W, H = dot.width, dot.height
    c = body.c
    # background tint: warm (low colour) -> cool (high colour)
    dot.background((int(14 + 10 * (1 - c)), 12, int(14 + 14 * c)))

    pulses[:] = [p for p in pulses if p.alive]
    for p in pulses:
        p.draw()

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
            if show_skel:
                dot.set_stroke_weight(2)
                dot.stroke((70, 70, 80))
                dot.no_fill()
                for a, b in SKELETON:
                    dot.line((lm[a].x * W, lm[a].y * H), (lm[b].x * W, lm[b].y * H))

    # drive the dials: model in run mode, OR the keyboard in manual mode
    if not manual_mode:
        read_dials_from_model(input_data)
    else:
        learner.new_example(input_data, current_target)  # keep GUI usable
    body.update_smoothed()

    # the band
    engine.update()
    for voice, midi in engine.events:
        pulses.append(Pulse(voice, midi))

    # HUD
    def meter(label, val, y, col):
        dot.fill((150, 150, 150)); dot.text(label, 20, y, 14)
        x0, w = 90, 180
        top, bot = y - 12, y + 2
        dot.no_fill(); dot.set_stroke_weight(1); dot.stroke((90, 90, 90))
        dot.polygon([(x0, top), (x0 + w, top), (x0 + w, bot), (x0, bot)])
        fw = int(w * val)
        if fw > 0:
            dot.no_stroke(); dot.fill(col)
            dot.polygon([(x0, top), (x0 + fw, top), (x0 + fw, bot), (x0, bot)])

    meter("energy", body.e, 40, (230, 120, 90))
    meter("colour", body.c, 64, (110, 170, 230))
    dot.fill((150, 150, 150))
    dot.text(f"{BPM} BPM   band {'ON' if engine.running else 'off'}"
             f"   {'MANUAL' if manual_mode else 'MODEL'} mode", 20, H - 20, 13)

    draw_training_panel(W, H)
    gui.draw(dot)


# Keyboard controls
# Manual mode is used for testing and live scoring.
# Z/X/N were added as fallback controls when the GUI buttons were not visible.
def handle_key(key, action, mods):
    global BPM, manual_mode, show_skel, current_target
    if "PRESS" not in str(action):
        return
    try:
        kc = int(key)
    except Exception:
        kc = key

    if kc in ANCHORS:                       # 1-5 select a training target
        current_target = ANCHORS[kc][:]
        print("target ->", current_target)
    elif key == dot.keys.SPACE or kc == 32:
        engine.toggle()
    elif kc == 109 or key == dot.keys.M:    # M
        manual_mode = not manual_mode
        print("MANUAL" if manual_mode else "MODEL", "mode")
    elif kc == 118 or key == dot.keys.V:    # V
        show_skel = not show_skel

    elif kc == 122:                         # Z = record on/off
        learner.toggle_record()

    elif kc == 120:                         # X = train
        learner.train()

    elif kc == 110:                         # N = run on/off
        learner.toggle_run()

    elif key == dot.keys.UP:
        if manual_mode: body.energy = clamp01(body.energy + 0.1)
    elif key == dot.keys.DOWN:
        if manual_mode: body.energy = clamp01(body.energy - 0.1)
    elif key == dot.keys.LEFT:
        if manual_mode: body.colour = clamp01(body.colour - 0.1)
    elif key == dot.keys.RIGHT:
        if manual_mode: body.colour = clamp01(body.colour + 0.1)
    elif kc == 91:                          # [
        BPM = max(50, BPM - 4); print("BPM", BPM)
    elif kc == 93:                          # ]
        BPM = min(160, BPM + 4); print("BPM", BPM)

try:
    dot.on_key_press = handle_key
except Exception as e:
    print("key callback not available:", e)


dot.start_loop(setup, draw)