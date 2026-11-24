"""Ability 1: Air Drawing - glowing finger trails with circle recognition."""
import math

import numpy as np

import config
from effects.base_effect import BaseEffect
from rendering.animations import ease_out_back, ellipse_points
from rendering.particles import ParticleSystem, rng
from tracking.motion import classify_stroke


class Stroke:
    def __init__(self, color):
        self.color = color
        self.pts, self.times = [], []

    def add(self, p, t):
        self.pts.append(np.array(p, float))
        self.times.append(t)

    @property
    def last_t(self):
        return self.times[-1] if self.times else 0.0


class EnergyRing:
    LIFE = 3.0

    def __init__(self, center, radius, color):
        self.center, self.radius, self.color, self.age = center, radius, color, 0.0

    @property
    def dead(self):
        return self.age > self.LIFE


class AirDrawing(BaseEffect):
    name = "Air Drawing"
    layer = 10

    def __init__(self):
        super().__init__()
        self.active_strokes = {}   # hand id -> Stroke
        self.strokes = []          # finished, fading strokes
        self.rings = []
        self.sparks = ParticleSystem(1500)
        self.color_index = 0
        self.now = 0.0
        self._smooth, self._seen = {}, {}

    @property
    def color(self):
        return config.DRAW_COLORS[self.color_index % len(config.DRAW_COLORS)]

    def cycle_color(self):
        self.color_index += 1

    def reset(self):
        self.active_strokes.clear(); self.strokes.clear(); self.rings.clear()
        self.sparks.clear(); self._smooth.clear(); self._seen.clear()

    @property
    def active(self):
        return bool(self.active_strokes or self.rings)

    def status(self):
        return "drawing" if self.active_strokes else ("ring" if self.rings else "")

    # -- gestures ---------------------------------------------------------
    def handle(self, ctx):
        self.now = ctx.t
        zones = ctx.shared.get("no_draw_zones", ())
        pointing = set()
        for h in ctx.hands:
            if h.pose != "POINT":
                continue
            pointing.add(h.id)
            prev = self._smooth.get(h.id)
            sm = h.tip.copy() if prev is None else prev + (h.tip - prev) * 0.65
            self._smooth[h.id] = sm
            st = self.active_strokes.get(h.id)
            if st is None:
                if any(np.hypot(*(sm - np.asarray(c))) < rad for c, rad in zones):
                    continue          # tapping inside an armed spiral: don't draw
                st = self.active_strokes[h.id] = Stroke(self.color)
            self._seen[h.id] = ctx.t
            if not st.pts or np.hypot(*(sm - st.pts[-1])) > config.STROKE_MIN_STEP:
                st.add(sm, ctx.t)
                if len(st.pts) > 700:
                    st.pts, st.times = st.pts[::2], st.times[::2]
            self._emit_sparks(sm, st.color)
        for hid in list(self.active_strokes):
            if hid not in pointing and ctx.t - self._seen.get(hid, 0) > config.STROKE_GRACE:
                self._finish(hid, ctx)

    def _emit_sparks(self, p, color):
        k = 2
        self.sparks.add(np.tile(p, (k, 1)) + rng.normal(0, 3, (k, 2)),
                        rng.normal(0, 40, (k, 2)), rng.uniform(0.35, 0.8, k),
                        rng.uniform(3, 6, k), color, 2.0, 25.0)

    def _finish(self, hid, ctx):
        st = self.active_strokes.pop(hid)
        self._smooth.pop(hid, None)
        if len(st.pts) < 2:
            return
        self.strokes.append(st)
        # Ignore the last ~0.12 s: points recorded while the pose debounce was
        # catching up with the finger relaxing would distort the shape.
        keep = [q for q, t in zip(st.pts, st.times) if t <= st.last_t - 0.12]
        shape = classify_stroke(np.array(keep if len(keep) >= 12 else st.pts))
        if shape is None:
            return
        if shape.kind == "circle":
            self.rings.append(EnergyRing(shape.center, shape.radius, st.color))
            ctx.emit("circle", center=shape.center, radius=shape.radius, hand_id=hid)
            self.sparks.burst(shape.center, 30, (80, 260), (0.4, 1.0), (3, 6),
                              [st.color, (255, 255, 255)], 2.0)
        else:
            ctx.emit("spiral", center=shape.center, radius=shape.radius, hand_id=hid)

    # -- animation --------------------------------------------------------
    def update(self, dt):
        for ring in self.rings:
            ring.age += dt
            if ring.age < ring.LIFE - 0.6:
                n = 3
                a = rng.uniform(0, 2 * math.pi, n)
                pos = ring.center + np.stack([np.cos(a), np.sin(a)], 1) * ring.radius
                self.sparks.add(pos, np.stack([np.cos(a), np.sin(a)], 1) * 40
                                + rng.normal(0, 20, (n, 2)),
                                rng.uniform(0.4, 0.9, n), rng.uniform(2, 5, n), ring.color, 2.0)
        self.rings = [r for r in self.rings if not r.dead]
        life = config.STROKE_HOLD + config.STROKE_FADE
        self.strokes = [s for s in self.strokes if self.now - s.last_t < life]
        self.sparks.update(dt)

    # -- rendering --------------------------------------------------------
    def _alpha(self, age):
        if age < config.STROKE_HOLD:
            return 1.0
        return max(0.0, 1.0 - (age - config.STROKE_HOLD) / config.STROKE_FADE)

    def draw(self, r):
        self.sparks.draw(r)
        for st in list(self.strokes) + list(self.active_strokes.values()):
            n = len(st.pts)
            for i in range(1, n):
                a = self._alpha(self.now - st.times[i])
                if a > 0.02:
                    r.glow_segment(st.pts[i - 1], st.pts[i], st.color, 5, a)
        for st in self.active_strokes.values():
            if st.pts:
                r.glow(st.pts[-1], 34, st.color, 0.9)
        for ring in self.rings:
            age = ring.age
            s = ease_out_back(min(age / 0.3, 1.0))
            fade = 1.0 if age < ring.LIFE - 0.7 else max(0.0, (ring.LIFE - age) / 0.7)
            R = ring.radius * s
            r.glow(ring.center, R * 1.5, ring.color, 0.25 * fade)
            r.ring(ring.center, R, 6, ring.color, fade)
            r.ring(ring.center, R * 0.86, 2, ring.color, 0.6 * fade)
            for k in range(3):                       # rotating bright arcs
                a0 = age * 2.2 + k * 2 * math.pi / 3
                t = np.linspace(a0, a0 + 0.6, 14)
                pts = ring.center + np.stack([np.cos(t), np.sin(t)], 1) * R
                r.glow_polyline(pts, (255, 255, 255), 4, 0.9 * fade)
