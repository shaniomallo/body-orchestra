# Body Orchestra: Conducting Music for Video and Live Performance

### Performing Data

Contributors: 

Shani O'Mallo-Mitchell

Harshada Pathare

## Overview
This project explored connecting body movement to sound. What began as a experimental performance where body movement triggered sounds and ml generated melodies, turned into a controlled performance where the body is used to conduct an instrument. In the final performance, the body is tracked through a webcam and a regression model maps each pose to two continuous controls, energy and colour. Those controls drive a rule based engine that generates bass, comp and melody in time and in key, so the performer shapes the music.

## Setup

**Requirements**

- Python 3.10 to 3.12 (the range MediaPipe supports)
- A webcam
- Speakers or headphones

**1. Install the Python packages**

```bash
pip install opencv-python mediapipe pygame numpy scikit-learn librosa soundfile
```

- `opencv-python`: webcam capture
- `mediapipe`: body pose tracking
- `pygame`: audio playback
- `numpy`, `scikit-learn`: power the regression model in `learner.py`
- `librosa`, `soundfile`: used by `prep_piano_samples.py` to build the sample bank

**2. Dorothy framework**

**Dorothy** (v0.67) and **`learner.py`** 

**3. Pose model**

`pose_landmarker_full.task`

**4. Piano samples**

Salamander Grand Piano WAV pack from <https://freepats.zenvoid.org/Piano/>, then run:

```bash
python prep_piano_samples.py
```

**5. Run**

```bash
python BODYORCHESTRA.PY
```

---

**Folder structure**

```
project/
  BODYORCHESTRA.PY
  learner.py
  prep_piano_samples.py
  models/pose_landmarker_full.task
  audio/piano/piano_*.wav
  conductor.csv          # created automatically when you train
```

**Controls (If in manual mode)**

- **SPACE**: start / stop the band
- **M**: manual mode on/off (drive the dials with the arrow keys to test the sound)
- **Arrow keys**: energy (up / down) and colour (left / right) in manual mode
- **1 to 5**: select a training-anchor target before recording
- **X**: wipe the training data and start the poses over
- **[ / ]**: tempo down / up
- **V**: show / hide the skeleton
- **Record / Train / Run**: on-screen Learner buttons, top-left

**Credits**

- Salamander Grand Piano: public domain
- MediaPipe Pose: Google, Apache 2.0


### Disclaimer
This performance includes short excerpts from two feature films, Notting Hill (1999, Universal Pictures / Working Title Films, dir. Roger Michell) and The Notebook (2004, New Line Cinema, dir. Nick Cassavetes), shown on screen as visual material for a live performance. The clips were used solely for non-commercial, educational purposes as part of academic coursework, under fair dealing for the purposes of illustration for instruction and criticism. The original film soundtracks were not used; all audio in the performance was generated live by our own system as an original score created through body movement. We claim no ownership of the film footage, and all rights remain with the respective copyright holders. This work is not intended for commercial distribution, and its use does not imply any affiliation with or endorsement by the rights holders.