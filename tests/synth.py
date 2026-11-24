"""Synthetic hand landmarks + fake renderer for headless testing."""
import numpy as np
from tracking.hand_tracker import TrackedHand, TrackerResult

S = 140.0   # pixels per palm unit
_MCP = {1: (-0.32, -0.95), 2: (0.0, -1.0), 3: (0.30, -0.95), 4: (0.55, -0.85)}  # idx,mid,ring,pinky


def _finger(mcp, extended):
    x, y = mcp
    if extended:
        return [(x, y), (x, y - 0.45), (x, y - 0.70), (x, y - 0.90)]
    return [(x, y), (x, y - 0.40), (x, y - 0.05), (x, y + 0.20)]


def make_hand(hid, pose, anchor, W=1280, H=720, anchor_is="palm"):
    """pose: POINT/OPEN/FIST/PINCH. anchor: palm centre (or index tip if anchor_is='tip')."""
    ext = {"POINT": (1, 0, 0, 0), "OPEN": (1, 1, 1, 1), "FIST": (0, 0, 0, 0),
           "PINCH": (0, 1, 1, 1)}[pose]
    pts = np.zeros((21, 2))
    for fi, base in enumerate((5, 9, 13, 17)):
        chain = _finger(_MCP[fi + 1], ext[fi])
        for j in range(4):
            pts[base + j] = chain[j]
    pts[0] = (0, 0)
    if pose == "PINCH":
        pts[8] = (-0.70, -1.20)                      # index curls to meet thumb
        pts[7], pts[6] = (-0.55, -1.30), (-0.32, -1.35)
        pts[1:5] = [(-0.35, -0.25), (-0.6, -0.55), (-0.66, -0.9), (-0.62, -1.15)]
    elif pose in ("POINT", "FIST"):
        pts[1:5] = [(-0.35, -0.25), (-0.45, -0.5), (-0.38, -0.65), (-0.3, -0.7)]
    else:
        pts[1:5] = [(-0.35, -0.25), (-0.6, -0.55), (-0.85, -0.8), (-1.1, -1.1)]
    if anchor_is == "tip":
        off = pts[8]
    else:
        off = pts[[0, 5, 9, 13, 17]].mean(axis=0)
    px = (pts - off) * S + np.asarray(anchor, float)
    lm = np.zeros((21, 3))
    lm[:, 0], lm[:, 1] = px[:, 0] / W, px[:, 1] / H
    return TrackedHand(hid, lm)


class Sim:
    """Feeds frames into a GestureManager at a fixed frame rate."""

    def __init__(self, manager, fps=30):
        self.m, self.dt, self.t, self.seq = manager, 1.0 / fps, 100.0, 0
        self.log = []

    def step(self, hands):
        self.t += self.dt
        self.seq += 1
        res = TrackerResult(seq=self.seq, t=self.t, hands=hands, camera_ok=True)
        ctx = self.m.update(res, self.dt, self.t)
        self.log.extend(n for n, _ in ctx.events)
        return ctx


class FakeRenderer:
    W, H = 1280, 720

    def __init__(self):
        self.calls = {}

    def __getattr__(self, name):
        def fn(*a, **k):
            self.calls[name] = self.calls.get(name, 0) + 1
            for v in a:
                arr = np.asarray(v, dtype=float) if not isinstance(v, (str, tuple)) or isinstance(v, tuple) else None
                if arr is not None and arr.dtype != object:
                    assert np.all(np.isfinite(arr)), f"{name}: non-finite arg {v}"
            if name == "glow_batch":
                p, rr, c = a
                assert len(p) == len(rr) == len(c), "glow_batch length mismatch"
        return fn
