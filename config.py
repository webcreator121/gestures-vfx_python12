"""Central configuration.

Values marked (calibratable) can be tuned live in calibration mode (press C)
and saved to calibration.json with S. Always access them as ``config.NAME`` so
runtime changes are seen everywhere.
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- display -------------------------------------------------------------
RENDER_SIZE = (1280, 720)      # internal resolution, scaled to the monitor
FULLSCREEN = True
TARGET_FPS = 60
CAMERA_DIM = 0.70              # darken the webcam image so effects pop

# ---- camera / MediaPipe ---------------------------------------------------
CAMERA_INDEX = 0
CAM_SIZE = (1280, 720)
CAM_FPS = 30
DETECT_WIDTH = 640             # frame width fed to MediaPipe (speed)
MODEL_COMPLEXITY = 0           # 0 = fastest
DETECT_CONF = 0.6
TRACK_CONF = 0.5
LM_SMOOTH = 0.55               # (calibratable) EMA weight of the new sample
HAND_GHOST_TIME = 0.5          # keep a hand id alive this long when lost
STALE_TIME = 0.6               # ignore tracker data older than this

# ---- gesture recognition --------------------------------------------------
PINCH_ON = 0.32                # (calibratable) thumb-index gap / palm size
PINCH_OFF = 0.45               # (calibratable) release threshold (hysteresis)
EXT_MARGIN = 1.05              # (calibratable) finger-extended sensitivity
POSE_STABLE_TIME = 0.08        # seconds a pose must persist to be accepted
ZONE_LEFT = 0.40               # palm.x / width below this -> left zone
ZONE_RIGHT = 0.60              # above this -> right zone

# ---- air drawing ----------------------------------------------------------
STROKE_HOLD = 3.0              # seconds a stroke stays fully visible
STROKE_FADE = 0.6
STROKE_MIN_STEP = 4            # px between recorded points
STROKE_GRACE = 0.18            # s of lost "point" pose before stroke ends
DRAW_COLORS = [(80, 200, 255), (255, 90, 210), (255, 200, 60),
               (120, 255, 150), (255, 255, 255)]

# ---- magical explosion ----------------------------------------------------
SPIRAL_ARM_TIME = 6.0          # s you have to tap after finishing a spiral
TAP_ZONE_RADIUS = 110
TAP_MIN_AMPLITUDE = 26         # px of out-and-back finger motion for a tap

# ---- hollow purple --------------------------------------------------------
HP_SPAWN_HOLD = 0.30           # s the pinch must be held to spawn an orb
HP_FUSE_START = 0.42           # hand distance / screen width where fusion starts
HP_FUSE_END = 0.10             # ... and where it is complete
HP_RELEASE_SPREAD = 0.14       # sudden spread (fraction of width) in 0.35 s
HP_LOST_GRACE = 1.0
HP_RELEASE_TIME = 3.0
BIG_GLOW_MAX = 700             # px cap for giant glows (performance)

# ---- abilities shown in the control panel --------------------------------
ENABLED = {"Air Drawing": True, "Magical Explosion": True, "Hollow Purple": True}

CALIBRATABLE = ("PINCH_ON", "PINCH_OFF", "EXT_MARGIN", "LM_SMOOTH")
CALIBRATION_FILE = os.path.join(BASE_DIR, "calibration.json")


def save_calibration():
    data = {k: globals()[k] for k in CALIBRATABLE}
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_calibration():
    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k in CALIBRATABLE:
            if k in data:
                globals()[k] = float(data[k])
    except (OSError, ValueError):
        pass


load_calibration()
