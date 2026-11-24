"""Motion analysis: stroke shape recognition (circle / spiral) and tap detection."""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class StrokeShape:
    kind: str                 # "circle" | "spiral"
    center: np.ndarray
    radius: float
    turns: float
    inward: bool = False


def path_length(pts):
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def resample(pts, n=64):
    """Resample a polyline to n points evenly spaced by arc length."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] < 1e-6:
        return None
    u = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(u, cum, pts[:, 0]), np.interp(u, cum, pts[:, 1])], 1)


def classify_stroke(points, min_length=200.0):
    """Return a StrokeShape for a circle or spiral, else None."""
    pts = np.asarray(points, float)
    if len(pts) < 12 or path_length(pts) < min_length:
        return None
    rs = resample(pts, 96)
    if rs is None:
        return None
    c = rs.mean(axis=0)
    d = rs - c
    r = np.hypot(d[:, 0], d[:, 1])
    if r.min() < 1e-3:
        return None
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    th = ang - ang[0]
    turns = abs(th[-1]) / (2 * math.pi)
    if turns < 0.8 or np.ptp(th) < 1e-6:
        return None
    k = max(4, int(0.15 * len(r)))
    r_start, r_end = float(r[:k].mean()), float(r[-k:].mean())
    ratio = max(r_start, r_end) / max(1.0, min(r_start, r_end))
    corr = abs(float(np.corrcoef(th, r)[0, 1]))
    # Spiral: radius changes steadily with angle, and several turns.
    if turns >= 1.5 and ratio >= 1.7 and corr >= 0.7:
        inward = r_end < r_start
        inner = rs[-k:] if inward else rs[:k]
        center = inner.mean(axis=0)
        return StrokeShape("spiral", center, float(r.max()), turns, inward)
    # Circle: near-constant radius, mostly closed.
    closure = float(np.hypot(*(rs[0] - rs[-1])))
    if ratio < 1.45 and r.std() / r.mean() < 0.22 and closure < 0.7 * r.mean() \
            and r.mean() > 25:
        return StrokeShape("circle", c, float(r.mean()), turns)
    return None


class TapDetector:
    """Detects a small out-and-back finger motion ("dab") within a short window."""

    def __init__(self, window=0.5, cooldown=0.5):
        self.window, self.cooldown = window, cooldown
        self.hist = deque()
        self.last_tap = -1e9

    def reset(self):
        self.hist.clear()

    def update(self, t, p):
        self.hist.append((t, float(p[0]), float(p[1])))
        while self.hist and t - self.hist[0][0] > self.window:
            self.hist.popleft()
        if len(self.hist) < 5 or t - self.last_tap < self.cooldown:
            return False
        a = np.array([(x, y) for _, x, y in self.hist])
        d = np.linalg.norm(a - a[0], axis=1)
        i = int(np.argmax(d))
        out = d[i]
        back = float(np.hypot(*(a[i] - a[-1])))
        if out >= config.TAP_MIN_AMPLITUDE and back >= 0.6 * out and 0 < i < len(a) - 1:
            self.last_tap = t
            self.hist.clear()
            return True
        return False
