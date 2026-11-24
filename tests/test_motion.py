import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tracking.motion import classify_stroke, TapDetector

rng = np.random.default_rng(1)

def circle(n=60, r=120, turns=1.05, noise=2.5, c=(600, 360)):
    t = np.linspace(0, 2 * math.pi * turns, n)
    return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], 1) + rng.normal(0, noise, (n, 2))

def spiral(n=100, r0=160, r1=20, turns=2.6, noise=2.0, c=(600, 360), start=0.0):
    t = np.linspace(0, 2 * math.pi * turns, n)
    r = np.linspace(r0, r1, n)
    return np.stack([c[0] + r * np.cos(t + start), c[1] + r * np.sin(t + start)], 1) + rng.normal(0, noise, (n, 2))

ok = 0
for i in range(20):
    s = classify_stroke(circle(r=rng.uniform(60, 180)))
    assert s and s.kind == "circle", f"circle {i}: {s}"
    s = classify_stroke(circle(turns=1.9, r=100))
    assert s and s.kind == "circle", f"double circle {i}: {s}"
    s = classify_stroke(spiral(start=rng.uniform(0, 6)))
    assert s and s.kind == "spiral", f"spiral {i}: {s}"
    assert np.hypot(*(s.center - np.array([600, 360]))) < 40, f"spiral centre off: {s.center}"
    s = classify_stroke(spiral(r0=20, r1=160))   # outward spiral
    assert s and s.kind == "spiral" and not s.inward
    ok += 1
line = np.stack([np.linspace(100, 900, 60), np.linspace(300, 350, 60)], 1)
assert classify_stroke(line) is None
assert classify_stroke(circle(turns=0.5, r=120)) is None   # half circle
assert classify_stroke(circle(r=8, n=20)) is None            # tiny
print("shape recognition OK", ok)

import config
td = TapDetector()
t = 0.0; fired = 0
base = np.array([640.0, 360.0])
for i in range(200):          # 30 fps: hover, then one dab at ~1.0s
    t += 1 / 30
    off = 0.0
    if 1.0 < t < 1.3:
        off = 45 * math.sin((t - 1.0) / 0.3 * math.pi)
    p = base + np.array([rng.normal(0, 1.2), off + rng.normal(0, 1.2)])
    if td.update(t, p):
        fired += 1
assert fired == 1, fired
td = TapDetector(); fired = 0; t = 0
for i in range(300):          # pure jitter must not fire
    t += 1 / 30
    if td.update(t, base + rng.normal(0, 2.0, 2)):
        fired += 1
assert fired == 0, fired
print("tap detection OK")
