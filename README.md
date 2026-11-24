# Gesture VFX - webcam-controlled supernatural effects

Real-time hand tracking (MediaPipe) drives cinematic additive-glow effects (Pygame) on top of your mirrored webcam image.

## Status (honest)
| Stage | Content | State |
|---|---|---|
| 1-2 | Webcam, MediaPipe, landmark pose recognition, skeleton | done |
| 3 | Air drawing + circle recognition | done |
| 4 | Particle engine + spiral/tap magical explosion | done |
| 5 | Blue / red orbs, fusion, **Hollow Purple** release | done |
| 6 | Fireball, Lightning, Black Hole, Shield | **not built yet** (next step) |
| 7 | HUD, ability panel, calibration | done for the abilities above |

All gesture logic (pose recognition, circle/spiral detection, tap detection, orb spawn, fusion, release trigger, fade-out) is covered by headless tests using synthetic hand landmarks (`python tests/test_motion.py`, `python tests/test_pipeline.py`).
The Pygame drawing and the live webcam/MediaPipe path could not be run in the environment where this was written, so expect to tune visuals and thresholds on your machine (calibration mode helps).

## Install (Windows, Python 3.11)
```
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
(MediaPipe 0.10.14 supports Python 3.8-3.11; use 3.11, not 3.12+.)

## Run
```
python main.py
```
Options: `--windowed`, `--camera 1`.

## Controls
`ESC` quit | `F` fullscreen | `1/2/3` toggle abilities (or click the panel) | `K` change drawing colour | `H` skeleton | `C` calibration (`[ ]` pinch, `- =` finger strictness, `, .` smoothing, `S` save, `R` reset)

## Gestures
Stand ~0.6-1 m back so both hands are fully visible, with even lighting.

* **Air Drawing** - extend only your index finger (other fingers curled) and move. Draw a closed circle (about one lap) -> glowing energy ring. Strokes stay ~3 s, then fade.
* **Magical Explosion** - draw a spiral (2+ turns, in or out). A pulsing marker appears at its centre; within 6 s poke/dab your index finger quickly out-and-back at the centre -> explosion.
* **Hollow Purple**
  1. "Circle" gesture = **OK sign**: touch thumb and index tips, keep other fingers extended. Hold ~0.3 s.
  2. Do it on the **left** of the screen (hand x < 40%) -> blue sphere; on the **right** (x > 60%) -> red sphere. Spheres follow your hands.
  3. Slowly bring your hands together: colours blend, purple core and rings grow.
  4. When mostly fused, **suddenly spread your hands apart** -> ~3 s purple-white release. Spreading slowly does nothing.
  5. Make a fist for ~0.5 s to dismiss a sphere.

## Adding an ability
Subclass `effects/base_effect.BaseEffect` (`handle`, `update`, `draw`), add it to the `effects` list in `main.py`. Use `ctx.hands`, `ctx.emit()/events_named()`, `ctx.shared`, and the `Renderer` primitives (`glow`, `ring`, `glow_polyline`, `add_flash/zoom/shake`).

## Troubleshooting
* **No webcam**: close Zoom/Teams/Camera app; try `python main.py --camera 1`; Windows Settings > Privacy > Camera must allow desktop apps. The app retries automatically.
* **`module 'mediapipe' has no attribute 'solutions'`** or install errors: use Python 3.11 and exactly the pinned versions; `pip uninstall opencv-python opencv-python-headless` then reinstall requirements (duplicate OpenCV packages conflict).
* **Hands not detected / jittery**: more light, plain background, hands fully in frame; raise smoothing in calibration or lower `DETECT_CONF` in `config.py`.
* **OK-sign not triggering**: press `C`, make the sign, watch "pinch ratio"; raise the pinch threshold with `]`.
* **Low FPS**: `RENDER_SIZE=(960,540)`, `CAM_SIZE=(640,480)`, `BIG_GLOW_MAX=450` in `config.py`; keep `MODEL_COMPLEXITY=0`.
