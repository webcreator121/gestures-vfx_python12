import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from effects.air_drawing import AirDrawing
from effects.magical_explosion import MagicalExplosion
from effects.hollow_purple import HollowPurple
from tracking.gesture_manager import GestureManager
from tracking.gesture_recognition import GestureRecognizer
from tests.synth import make_hand, Sim, FakeRenderer
import config

W, H = 1280, 720

def new_sim():
    fx = [AirDrawing(), MagicalExplosion(), HollowPurple()]
    return Sim(GestureManager(fx, (W, H))), fx

def draw_all(sim, r):
    sim.m.draw(r)

# ---- 0. pose recognition on synthetic hands ---------------------------------
rec = GestureRecognizer()
import time
for pose in ("POINT", "OPEN", "FIST", "PINCH"):
    t = 0.0; last = None
    for i in range(10):
        t += 0.033
        last = rec.process([make_hand(5, pose, (640, 360))], (W, H), t)[0]
    assert last.pose == pose, f"{pose} recognised as {last.pose} (ratio {last.pinch_ratio:.2f}, fingers {last.fingers})"
print("pose recognition OK")

# ---- 1. air drawing: circle -> energy ring ------------------------------------
sim, (air, boom, hp) = new_sim()
r = FakeRenderer()
for i in range(8):
    sim.step([make_hand(1, "POINT", (640 + 120, 360), anchor_is="tip")])
for i in range(70):
    a = 2 * math.pi * 1.05 * i / 69
    sim.step([make_hand(1, "POINT", (640 + 120 * math.cos(a), 360 + 120 * math.sin(a)), anchor_is="tip")])
    draw_all(sim, r); 
for i in range(12):
    sim.step([make_hand(1, "OPEN", (640, 500))]); draw_all(sim, r)
assert "circle" in sim.log, sim.log
assert len(air.rings) == 1, len(air.rings)
print("air drawing circle -> ring OK")
# strokes fade after ~3.6 s
for i in range(150):
    sim.step([]); air.update(0)
assert not air.strokes, len(air.strokes)
print("stroke fade-out OK")

# ---- 2. spiral -> tap centre -> explosion -------------------------------------
sim, (air, boom, hp) = new_sim()
c = np.array([640.0, 360.0])
for i in range(110):
    u = i / 109; a = 2 * math.pi * 2.6 * u; rad = 170 * (1 - u) + 22 * u
    sim.step([make_hand(1, "POINT", c + rad * np.array([math.cos(a), math.sin(a)]), anchor_is="tip")])
for i in range(10):
    sim.step([make_hand(1, "OPEN", (900, 600))])
assert "spiral" in sim.log, sim.log
assert len(boom.armed) == 1
for i in range(12):    # hover at centre
    sim.step([make_hand(1, "POINT", c + (2 * (i % 2), 0), anchor_is="tip")])
assert not boom.blasts
for i in range(10):    # dab down and back
    off = 45 * math.sin(i / 9 * math.pi)
    sim.step([make_hand(1, "POINT", c + (0, off), anchor_is="tip")])
assert len(boom.blasts) == 1, f"blasts={len(boom.blasts)} armed={len(boom.armed)}"
r = FakeRenderer()
for i in range(100):
    sim.step([]); draw_all(sim, r)
assert not boom.blasts and boom.ps.n == 0
assert r.calls.get("add_flash") and r.calls.get("ring")
print("spiral + tap -> explosion OK")

# ---- 3. hollow purple -----------------------------------------------------------
sim, (air, boom, hp) = new_sim()
r = FakeRenderer()
L, R = np.array([250.0, 380.0]), np.array([1030.0, 380.0])
for i in range(25):    # hold pinch left + right
    sim.step([make_hand(1, "PINCH", L), make_hand(2, "PINCH", R)]); draw_all(sim, r)
assert set(hp.orbs) == {"blue", "red"}, hp.orbs
assert hp.orbs["blue"].hand_id == 1 and hp.orbs["red"].hand_id == 2
for i in range(45):    # slowly bring hands together
    u = i / 44; d = (1 - u) * 780 + u * 90
    sim.step([make_hand(1, "PINCH", (640 - d / 2, 380)), make_hand(2, "PINCH", (640 + d / 2, 380))]); draw_all(sim, r)
assert hp.fusion > 0.85, hp.fusion
assert hp.rel_t is None
for i in range(20):    # hold fused
    sim.step([make_hand(1, "PINCH", (595, 380)), make_hand(2, "PINCH", (685, 380))]); draw_all(sim, r)
for i in range(8):     # sudden spread
    u = (i + 1) / 8; d = 90 + u * 560
    sim.step([make_hand(1, "OPEN", (640 - d / 2, 380)), make_hand(2, "OPEN", (640 + d / 2, 380))]); draw_all(sim, r)
assert hp.rel_t is not None, f"release not triggered (fusion {hp.fusion:.2f})"
print("hollow purple spawn + fusion + release trigger OK")
flashes = []
t_end = 0
while hp.rel_t is not None and t_end < 6:
    sim.step([]); r2 = FakeRenderer(); draw_all(sim, r2); t_end += sim.dt
    flashes.append(r2.calls.get("add_flash", 0))
assert 2.9 < t_end < 3.6, t_end
print(f"release lasted {t_end:.2f}s then faded back OK")
assert not hp.orbs and hp.cooldown > 0

# slow spread must NOT release
sim, (air, boom, hp) = new_sim()
for i in range(25):
    sim.step([make_hand(1, "PINCH", L), make_hand(2, "PINCH", R)])
for i in range(60):
    u = i / 59; d = (1 - u) * 780 + u * 90
    sim.step([make_hand(1, "PINCH", (640 - d / 2, 380)), make_hand(2, "PINCH", (640 + d / 2, 380))])
for i in range(90):
    u = (i + 1) / 90; d = 90 + u * 600
    sim.step([make_hand(1, "OPEN", (640 - d / 2, 380)), make_hand(2, "OPEN", (640 + d / 2, 380))])
assert hp.rel_t is None, "slow spread wrongly triggered release"
print("slow spread correctly ignored OK")

# orb dies when hand lost
for i in range(60):
    sim.step([])
assert not hp.orbs
print("orb cleanup after lost hands OK")
print("ALL PIPELINE TESTS PASSED")
