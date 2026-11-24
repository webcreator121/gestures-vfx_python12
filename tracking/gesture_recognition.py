"""Landmark-based hand pose recognition (no finger counting shortcuts).

For every tracked hand we compute: pixel landmarks, palm centre/size, per-finger
extension (distance-from-wrist test, rotation invariant), finger curl angles,
hand orientation, thumb-index pinch ratio, velocities and a debounced pose.
"""
import math

import numpy as np

import config

FINGERS = {"index": (5, 6, 8), "middle": (9, 10, 12),
           "ring": (13, 14, 16), "pinky": (17, 18, 20)}
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def _d(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class Hand:
    """Per-frame processed hand. Coordinates are in render pixels."""

    def __init__(self):
        self.id = 0
        self.pts = None           # (21, 2)
        self.palm = None
        self.size = 1.0           # wrist -> middle MCP distance
        self.tip = None           # index fingertip
        self.fingers = [False] * 5   # thumb, index, middle, ring, pinky extended
        self.curl = [180.0] * 4   # PIP angles (180 = straight)
        self.angle = 0.0          # orientation of wrist->middle MCP, degrees
        self.pinch_ratio = 1.0
        self.pose = "OTHER"       # POINT | OPEN | FIST | PINCH | OTHER
        self.pose_since = 0.0
        self.vel = np.zeros(2)
        self.tip_vel = np.zeros(2)
        self.side = "C"           # L | C | R screen zone
        self.handedness = "?"

    def hold(self, t):
        """Seconds the current pose has been held."""
        return max(0.0, t - self.pose_since)

    @property
    def pinch_center(self):
        return (self.pts[4] + self.pts[8]) / 2


class _State:
    def __init__(self, t):
        self.pose, self.pose_since = "OTHER", t
        self.cand, self.cand_since = None, t
        self.pinching = False
        self.prev_palm = self.prev_tip = self.prev_t = None
        self.vel = np.zeros(2)
        self.tip_vel = np.zeros(2)


class GestureRecognizer:
    def __init__(self):
        self._state = {}

    def process(self, tracked, size, t):
        W, H = size
        scale = np.array([W, H], float)
        hands, seen = [], set()
        for th in tracked:
            seen.add(th.id)
            st = self._state.get(th.id)
            if st is None:
                st = self._state[th.id] = _State(t)
            hands.append(self._build(th, st, scale, W, t))
        for hid in list(self._state):
            if hid not in seen:
                del self._state[hid]
        return hands

    def _build(self, th, st, scale, W, t):
        h = Hand()
        h.id, h.handedness = th.id, th.handedness
        p = h.pts = th.lm[:, :2] * scale
        h.palm = p[[0, 5, 9, 13, 17]].mean(axis=0)
        h.size = max(_d(p[0], p[9]), 1.0)
        h.tip = p[8].copy()
        v = p[9] - p[0]
        h.angle = math.degrees(math.atan2(v[1], v[0]))
        # finger extension: tip farther from wrist than the PIP joint
        for i, (mcp, pip, tip) in enumerate(FINGERS.values(), start=1):
            h.fingers[i] = _d(p[tip], p[0]) > _d(p[pip], p[0]) * config.EXT_MARGIN
            a, b = p[mcp] - p[pip], p[tip] - p[pip]
            cosv = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
            h.curl[i - 1] = math.degrees(math.acos(max(-1.0, min(1.0, cosv))))
        h.fingers[0] = _d(p[4], p[17]) > _d(p[3], p[17]) * 1.1
        h.pinch_ratio = _d(p[4], p[8]) / h.size
        # pinch (thumb+index circle) with hysteresis
        limit = config.PINCH_OFF if st.pinching else config.PINCH_ON
        # OK-sign: thumb+index touch while at least one other finger is extended
        # (keeps a closed fist, where the thumb also touches the index, a FIST)
        st.pinching = h.pinch_ratio < limit and any(h.fingers[2:])
        n = sum(h.fingers[1:])
        if st.pinching:
            raw = "PINCH"
        elif h.fingers[1] and n == 1:
            raw = "POINT"
        elif n == 4:
            raw = "OPEN"
        elif n == 0:
            raw = "FIST"
        else:
            raw = "OTHER"
        # time-based debounce
        if raw != st.pose:
            if raw != st.cand:
                st.cand, st.cand_since = raw, t
            elif t - st.cand_since >= config.POSE_STABLE_TIME:
                st.pose, st.pose_since, st.cand = raw, t, None
        else:
            st.cand = None
        h.pose, h.pose_since = st.pose, st.pose_since
        # velocities
        if st.prev_t is not None and t - st.prev_t > 1e-3:
            dt = t - st.prev_t
            st.vel = 0.5 * st.vel + 0.5 * (h.palm - st.prev_palm) / dt
            st.tip_vel = 0.5 * st.tip_vel + 0.5 * (h.tip - st.prev_tip) / dt
        st.prev_palm, st.prev_tip, st.prev_t = h.palm.copy(), h.tip.copy(), t
        h.vel, h.tip_vel = st.vel.copy(), st.tip_vel.copy()
        fx = h.palm[0] / W
        h.side = "L" if fx < config.ZONE_LEFT else "R" if fx > config.ZONE_RIGHT else "C"
        return h
