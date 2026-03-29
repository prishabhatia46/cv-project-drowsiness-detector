# Driver Drowsiness Detection System

A real-time computer vision application that monitors a driver's eyes through a webcam and sounds an alarm when drowsiness is detected — before it causes an accident.

---

## What This Project Does

The system uses your computer's webcam to continuously analyse your face. It tracks the position of your eyelids using 68 facial landmarks, computes a value called the **Eye Aspect Ratio (EAR)**, and triggers an audio alarm when your eyes stay closed long enough to indicate drowsiness (rather than a normal blink).

No wearable sensors. No internet connection. No GPU required. It runs on any standard laptop.

---

## How It Works

The Eye Aspect Ratio measures the geometry of the eye using six landmark points:

```
        ||p2 - p6|| + ||p3 - p5||
EAR  =  ──────────────────────────
              2 · ||p1 - p4||
```

- When eyes are **open**: EAR ≈ 0.30 (stays roughly constant)
- When eyes are **closing**: EAR drops toward 0
- When EAR stays **below 0.25 for 20+ consecutive frames** (~0.67 seconds): alarm fires

This distinguishes a normal blink (EAR dips briefly then recovers) from genuine drowsiness (EAR stays low).

---

## Requirements

- Python 3.7 or higher
- A working webcam
- Windows, macOS, or Linux

---

## Installation

### Step 1 — Install Python dependencies

```bash
pip install opencv-python dlib-bin scipy pygame imutils
```

> **Windows users:** Use `dlib-bin` (not `dlib`). Building dlib from source on Windows requires a complex CMake setup. `dlib-bin` provides a pre-compiled version that installs cleanly.

### Step 2 — Download the facial landmark model

The face landmark predictor is a pre-trained model (~95 MB) that is not included in this repository. Download it here:

```
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

1. Download the `.bz2` file
2. Extract it (using 7-Zip on Windows, or `bzip2 -d` on Mac/Linux)
3. Place the resulting `shape_predictor_68_face_landmarks.dat` file in the project folder

### Step 3 — Add an alarm sound

Place any `.wav` audio file in the project folder and name it `alarm.wav`.

You can download a free alarm sound from: https://mixkit.co/free-sound-effects/alarm/

---

## Project Structure

After completing setup, your folder should look like this:

```
drowsiness_detector/
├── detect.py                               ← main script (this is what you run)
├── shape_predictor_68_face_landmarks.dat   ← downloaded in Step 2
├── alarm.wav                               ← added in Step 3
└── README.md
```

---

## Running the Application

Navigate to the project folder in your terminal and run:

```bash
python detect.py
```

A window will open showing your webcam feed. You will see:

- **Green outlines** drawn around your eyes
- **EAR value** displayed in the top-right corner
- A **red "DROWSINESS ALERT!" message** and audio alarm if your eyes stay closed too long

Press **`Q`** to quit.

---

## Configuration

Two values at the top of `detect.py` control the sensitivity:

```python
EAR_THRESHOLD = 0.25   # EAR below this value counts as "eyes closed"
FRAME_LIMIT   = 20     # How many consecutive frames before the alarm fires
```

| If you want... | Change... |
|----------------|-----------|
| Fewer false alarms | Decrease `EAR_THRESHOLD` (e.g. 0.22) |
| Faster alarm response | Decrease `FRAME_LIMIT` (e.g. 15) |
| More tolerance before alarm | Increase `FRAME_LIMIT` (e.g. 30) |

---

## Troubleshooting

**Webcam not opening**
Make sure no other application (Zoom, Teams, etc.) is using the camera. Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` if you have multiple cameras.

**Face not being detected**
Ensure your face is well-lit from the front. Backlighting (bright window behind you) makes detection unreliable.

**Alarm not playing**
Confirm that `alarm.wav` is in the same folder as `detect.py` and that your system volume is not muted.

**dlib install fails**
Make sure you are using `pip install dlib-bin` and not `pip install dlib`.

---

## Known Limitations

- Works best with frontal face orientation; large head rotations reduce accuracy
- Thick-framed glasses can partially occlude eye landmarks and affect EAR readings
- Performance may degrade in very low light conditions
- Detects one face per frame (the first face found)

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Webcam capture, frame processing, drawing overlays |
| `dlib-bin` | Face detection and 68-point facial landmark prediction |
| `imutils` | Frame resizing and facial landmark index utilities |
| `scipy` | Euclidean distance calculations for EAR formula |
| `pygame` | Audio alarm playback |

---

## References

- Soukupová, T. & Cech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks.* 21st Computer Vision Winter Workshop.
- Kazemi, V. & Sullivan, J. (2014). *One Millisecond Face Alignment with an Ensemble of Regression Trees.* IEEE CVPR.
- dlib library: http://dlib.net
- OpenCV: https://docs.opencv.org

---

*B.Tech Computer Science & Engineering — Computer Vision Assignment, 2024–25*
