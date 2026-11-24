"""Ability 2: Magical Explosion - draw a spiral, then tap its centre."""
import math

import numpy as np

import config
from effects.base_effect import BaseEffect
from rendering.animations import Shockwave, ease_out_cubic, ellipse_points
from rendering.particles import ParticleSystem
from tracking.motion import TapDetector

GOLD, VIOLET, WHITE, CYAN = (255, 200, 80), (170, 90, 255), (255, 255, 255), (90, 200, 255)


class _Armed:
    def __init__(self, center, t):
        self.center, self.t0 = np.asarray(center, float), t
        self.taps = {}


class _Blast:
    def __init__(self, pos):
        self.pos = np.asarray(pos, float)
        self.age = 0.0
        self.waves = [Shockwave(self.pos, 720, 14, WHITE, 1.3, 0.0),
                      Shockwave(self.pos, 560, 10, GOLD, 1.5, 0.12),
                      Shockwave(self.pos, 420, 8, VIOLET, 1.7, 0.25)]

    @property
    def dead(self):
        return self.age > 2.0


class MagicalExplosion(BaseEffect):
    name = "Magical Explosion"
    layer = 20

    def __init__(self):
        super().__init__()
        self.armed, self.blasts = [], []
        self.ps = ParticleSystem(3000)
        self.now = 0.0
        self.clock = 0.0

    def reset(self):
        self.armed.clear(); self.blasts.clear(); self.ps.clear()

    @property
    def active(self):
        return bool(self.armed or self.blasts)

    def status(self):
        return "tap the spiral centre" if self.armed else ("exploding" if self.blasts else "")

    def handle(self, ctx):
        self.now = ctx.t
        for ev in ctx.events_named("spiral"):
            self.armed.append(_Armed(ev["center"], ctx.t))
        self.armed = [a for a in self.armed if ctx.t - a.t0 < config.SPIRAL_ARM_TIME]
        ctx.shared["no_draw_zones"] = [(a.center, config.TAP_ZONE_RADIUS) for a in self.armed]
        for a in list(self.armed):
            for h in ctx.hands:
                td = a.taps.setdefault(h.id, TapDetector())
                inside = h.fingers[1] and np.hypot(*(h.tip - a.center)) < config.TAP_ZONE_RADIUS
                if not inside:
                    td.reset()
                    continue
                if td.update(ctx.t, h.tip):
                    self._explode(a.center)
                    if a in self.armed:
                        self.armed.remove(a)
                    break

    def _explode(self, pos):
        self.blasts.append(_Blast(pos))
        self.ps.burst(pos, 450, (150, 900), (0.8, 1.9), (3, 9), [GOLD, WHITE, VIOLET, CYAN], 2.2)
        self.ps.burst(pos, 150, (30, 250), (1.0, 2.0), (6, 14), [GOLD, VIOLET], 1.5, -40)

    def update(self, dt):
        self.clock += dt
        for b in self.blasts:
            b.age += dt
            for w in b.waves:
                w.update(dt)
        self.blasts = [b for b in self.blasts if not b.dead]
        self.ps.update(dt)

    def draw(self, r):
        for a in self.armed:                       # pulsing "tap here" marker
            u = (self.now - a.t0) / config.SPIRAL_ARM_TIME
            pulse = 0.5 + 0.5 * math.sin(self.clock * 8)
            r.glow(a.center, 40 + 10 * pulse, VIOLET, 0.8 * (1 - u * 0.5))
            r.ring(a.center, 26 + 8 * pulse, 2, WHITE, 0.7)
        for b in self.blasts:
            t = b.age
            r.add_flash(0.6 * math.exp(-t * 7))
            r.add_zoom(0.035 * math.exp(-t * 5))
            r.add_shake(8 * math.exp(-t * 6))
            core = max(0.0, 1 - t / 0.7)
            r.glow(b.pos, 90 + 260 * ease_out_cubic(t / 0.5), GOLD, 0.9 * core)
            r.glow(b.pos, 50 + 140 * ease_out_cubic(t / 0.5), WHITE, 0.9 * core)
            for w in b.waves:
                w.draw(r)
            for i in range(2):                     # spinning energy rings
                u = ease_out_cubic(t / 1.5)
                fade = max(0.0, 1 - t / 1.5)
                pts = ellipse_points(b.pos, 300 * u + 20, (300 * u + 20) * 0.35,
                                     t * (3 + i) + i * 1.2)
                r.glow_polyline(pts, VIOLET if i else GOLD, 4, fade)
        self.ps.draw(r)
