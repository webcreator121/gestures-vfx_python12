"""Ability 3: Hollow Purple.

Pinch (thumb+index circle) on the LEFT of the screen -> blue sphere ("Blue").
Same gesture on the RIGHT -> red sphere ("Red"). Bring the hands together to
fuse them into purple, then spread them apart suddenly to release.
"""
import math
from collections import deque

import numpy as np

import config
from effects.base_effect import BaseEffect
from rendering.animations import (Shockwave, clamp, ease_out_back, ease_out_cubic,
                                  ellipse_points, lerp_color, smoothstep)
from rendering.particles import ParticleSystem, rng

BLUE, RED, PURPLE = (50, 130, 255), (255, 50, 45), (190, 70, 255)
WHITE, LILAC = (255, 255, 255), (225, 170, 255)
N_ORBIT = 34


class Orb:
    def __init__(self, kind, hand_id, pos, radius):
        self.kind, self.hand_id, self.radius = kind, hand_id, radius
        self.pos = np.array(pos, float)
        self.age, self.lost, self.fade, self.dying = 0.0, 0.0, 1.0, False
        self.hand = None
        self.ang = rng.uniform(0, 2 * math.pi, N_ORBIT)
        self.rr = rng.uniform(1.05, 1.9, N_ORBIT)
        self.spd = rng.choice([-1.0, 1.0], N_ORBIT) * rng.uniform(2.5, 6.0, N_ORBIT)
        self.tilt = rng.uniform(0, math.pi, N_ORBIT)
        self.squash = rng.uniform(0.35, 1.0, N_ORBIT)

    @property
    def base(self):
        return BLUE if self.kind == "blue" else RED


class HollowPurple(BaseEffect):
    name = "Hollow Purple"
    layer = 30

    def __init__(self):
        super().__init__()
        self.orbs = {}                       # "blue" / "red" -> Orb
        self.fusion = 0.0
        self.hist = deque()                  # (t, distance_fraction, fusion)
        self.ps = ParticleSystem(3500)
        self.waves = []
        self.rel_t = None                    # None = no release running
        self.rel_center = np.zeros(2)
        self.cooldown = 0.0
        self.clock = 0.0
        self.mid = np.zeros(2)
        self._emit_acc = 0.0

    # ------------------------------------------------------------------
    def reset(self):
        self.orbs.clear(); self.fusion = 0.0; self.hist.clear(); self.ps.clear()
        self.waves.clear(); self.rel_t = None

    @property
    def active(self):
        return bool(self.orbs) or self.rel_t is not None

    def status(self):
        if self.rel_t is not None:
            return "RELEASE"
        if len(self.orbs) == 2:
            return f"fusing {int(self.fusion * 100)}%"
        return " + ".join(self.orbs) if self.orbs else ""

    # ------------------------------------------------------------------
    def handle(self, ctx):
        W, H = ctx.size
        dt, hands = ctx.dt, {h.id: h for h in ctx.hands}
        bound = {o.hand_id for o in self.orbs.values()}
        if self.cooldown > 0:
            self.cooldown -= dt

        # spawn orbs on a held pinch in the left / right zone
        if self.rel_t is None and self.cooldown <= 0:
            for h in ctx.hands:
                if h.id in bound or h.pose != "PINCH" or h.hold(ctx.t) < config.HP_SPAWN_HOLD:
                    continue
                kind = "blue" if h.side == "L" else "red" if h.side == "R" else None
                if kind and kind not in self.orbs:
                    radius = float(np.clip(0.9 * h.size + 20, 38, 90))
                    self.orbs[kind] = Orb(kind, h.id, h.palm, radius)
                    bound.add(h.id)
                    self.ps.burst(h.palm, 40, (80, 300), (0.4, 0.9), (3, 6),
                                  [self.orbs[kind].base, WHITE], 2.5)

        # follow hands / lose hands / dismiss with a fist
        for kind, orb in list(self.orbs.items()):
            orb.hand = hands.get(orb.hand_id)
            if orb.hand is not None:
                orb.lost = 0.0
                if orb.hand.pose == "FIST" and orb.hand.hold(ctx.t) > 0.4:
                    orb.dying = True
            else:
                orb.lost += dt
                if orb.lost > config.HP_LOST_GRACE:
                    orb.dying = True
            if orb.dying:
                orb.fade -= dt * 2.5
                if orb.fade <= 0:
                    del self.orbs[kind]

        live = [o for o in self.orbs.values() if not o.dying]
        blue, red = self.orbs.get("blue"), self.orbs.get("red")
        d_frac, target_f = None, 0.0
        if blue and red and blue.hand is not None and red.hand is not None and len(live) == 2:
            d_frac = float(np.hypot(*(blue.hand.palm - red.hand.palm))) / W
            target_f = clamp((config.HP_FUSE_START - d_frac)
                             / (config.HP_FUSE_START - config.HP_FUSE_END))
        k = 1 - math.exp(-(5 if target_f > self.fusion else 9) * dt)
        self.fusion += (target_f - self.fusion) * k
        if self.fusion < 0.001:
            self.fusion = 0.0

        # orb positions: follow hand, pulled toward the midpoint as fusion grows
        if blue and red:
            self.mid = (blue.pos + red.pos) / 2
        for orb in self.orbs.values():
            tgt = orb.hand.palm if orb.hand is not None else orb.pos
            if blue and red:
                tgt = tgt + (self.mid - tgt) * 0.5 * self.fusion
            orb.pos += (tgt - orb.pos) * (1 - math.exp(-14 * dt))

        # release detection: fused (recently) + sudden spread
        if d_frac is not None:
            self.hist.append((ctx.t, d_frac, self.fusion))
        while self.hist and ctx.t - self.hist[0][0] > 0.6:
            self.hist.popleft()
        if self.rel_t is None and d_frac is not None and len(self.hist) >= 3:
            f_recent = max(f for _, _, f in self.hist)
            recent = [(t, d) for t, d, _ in self.hist if ctx.t - t <= 0.35]
            if f_recent > 0.6 and recent and \
                    d_frac - min(d for _, d in recent) > config.HP_RELEASE_SPREAD:
                self._release(self.mid.copy())

    # ------------------------------------------------------------------
    def _release(self, center):
        self.rel_t, self.rel_center = 0.0, center
        self.orbs.clear(); self.fusion = 0.0; self.hist.clear()
        self.waves = [Shockwave(center, 1500, 22, WHITE, 1.8, 0.0),
                      Shockwave(center, 1200, 16, PURPLE, 2.0, 0.15),
                      Shockwave(center, 900, 12, LILAC, 2.2, 0.3),
                      Shockwave(center, 700, 8, PURPLE, 2.4, 0.5)]
        self.ps.burst(center, 900, (200, 1500), (1.0, 2.5), (3, 9),
                      [PURPLE, LILAC, WHITE, (140, 60, 255)], 1.8)

    def update(self, dt):
        self.clock += dt
        for orb in self.orbs.values():
            orb.age += dt
        # energy streams flowing from each orb to the fusion point
        if len(self.orbs) == 2 and self.fusion > 0.05:
            self._emit_acc += dt * (60 + 260 * self.fusion)
            n = int(self._emit_acc)
            self._emit_acc -= n
            for orb in self.orbs.values():
                if n <= 0:
                    break
                p = orb.pos + rng.normal(0, orb.radius * 0.4, (n, 2))
                dirv = self.mid - p
                dirv /= np.linalg.norm(dirv, axis=1, keepdims=True) + 1e-6
                col = np.array(lerp_color(orb.base, PURPLE, self.fusion))
                self.ps.add(p, dirv * (200 + 350 * self.fusion) + rng.normal(0, 40, (n, 2)),
                            rng.uniform(0.3, 0.7, n), rng.uniform(3, 6, n), col, 1.0)
        if self.rel_t is not None:
            self.rel_t += dt
            if self.rel_t < 0.9:
                n = int(260 * dt) + 1
                self.ps.burst(self.rel_center, n, (300, 1200), (0.6, 1.6), (3, 8),
                              [PURPLE, LILAC, WHITE], 1.8)
            for w in self.waves:
                w.update(dt)
            if self.rel_t > config.HP_RELEASE_TIME + 0.3:
                self.rel_t, self.cooldown = None, 2.0
                self.waves.clear()
        self.ps.update(dt)

    # ------------------------------------------------------------------
    def _draw_orb(self, r, orb, alpha):
        s = ease_out_back(min(orb.age / 0.45, 1.0))
        R = orb.radius * s * (1 + 0.07 * math.sin(self.clock * 9 + orb.ang[0]))
        a = alpha * orb.fade
        col = lerp_color(orb.base, PURPLE, self.fusion * 0.8)
        r.glow(orb.pos, R * 3.4, col, 0.35 * a)
        r.glow(orb.pos, R * 1.9, col, 0.7 * a)
        r.glow(orb.pos, R * 1.15, lerp_color(col, WHITE, 0.35), a)
        r.glow(orb.pos, R * 0.6, WHITE, 0.9 * a)
        th0 = orb.ang + orb.spd * orb.age                   # swirling particles with trails
        for k in range(4):
            th = th0 - k * orb.spd * 0.035
            lx, ly = np.cos(th) * orb.rr * R, np.sin(th) * orb.rr * R * orb.squash
            c, sn = np.cos(orb.tilt), np.sin(orb.tilt)
            pos = orb.pos + np.stack([lx * c - ly * sn, lx * sn + ly * c], 1)
            fade = (1 - k / 4) ** 1.5 * a
            r.glow_batch(pos, np.full(N_ORBIT, max(2.0, 6 - 1.2 * k)),
                         np.tile(np.array(lerp_color(col, WHITE, 0.4)) * fade, (N_ORBIT, 1)))
        r.ring(orb.pos, R * 1.05, 3, col, 0.6 * a)

    def _draw_fusion(self, r):
        f = self.fusion
        blue, red = self.orbs["blue"], self.orbs["red"]
        if f > 0.05:                                         # crackling trail between orbs
            n = 12
            t = np.linspace(0, 1, n)[:, None]
            pts = blue.pos + (red.pos - blue.pos) * t
            amp = 16 * (1 - f) + 3
            pts[1:-1] += rng.normal(0, amp, (n - 2, 2))
            r.glow_polyline(pts, lerp_color(BLUE, RED, 0.5 + 0.0 * f), 3, 0.7 * f)
        R0 = 0.5 * (blue.radius + red.radius)
        g = smoothstep(0.3, 0.9, f)
        if g > 0:
            R = R0 * (1 + 0.8 * f) * (1 + 0.06 * math.sin(self.clock * 11))
            r.glow(self.mid, R * (2 + 3 * f), PURPLE, 0.45 * g)
            r.glow(self.mid, R * 1.4, PURPLE, 0.9 * g)
            r.glow(self.mid, R * 0.8, LILAC, g)
            r.glow(self.mid, R * 0.45, WHITE, g)
            for i in range(2 + int(f * 3)):                  # rotating energy rings
                rot = self.clock * (2.5 + i) + i * math.pi / 3
                r.glow_polyline(ellipse_points(self.mid, R * (1.5 + 0.25 * i),
                                               R * 0.5 * (1.5 + 0.25 * i), rot),
                                LILAC if i % 2 else PURPLE, 3, 0.8 * g, True)
        r.add_zoom(0.012 * f * (0.5 + 0.5 * math.sin(self.clock * 8)))   # pulsing distortion
        r.add_shake(2.5 * f * f)

    def _draw_release(self, r):
        t = self.rel_t
        c = self.rel_center
        g = 1 - smoothstep(2.2, config.HP_RELEASE_TIME, t)
        R = 80 + 760 * ease_out_cubic(min(t / 1.1, 1.0))
        r.glow(c, R * 1.6, PURPLE, 0.55 * g)
        r.glow(c, R, LILAC, 0.9 * g)
        r.glow(c, R * 0.6, WHITE, g * (1 - 0.5 * smoothstep(0.6, 1.6, t)))
        for w in self.waves:
            w.draw(r, g)
        for i in range(3):
            u = ease_out_cubic(min(t / 1.6, 1.0))
            rad = (150 + 450 * u) * (0.6 + 0.3 * i)
            r.glow_polyline(ellipse_points(c, rad, rad * 0.4, t * (3 + i) + i),
                            LILAC if i % 2 else PURPLE, 5, g * (1 - 0.5 * u), True)
        flash = t / 0.15 if t < 0.15 else math.exp(-(t - 0.15) * 3.2)
        r.add_flash(flash)
        r.add_zoom(0.07 * math.exp(-3 * t))
        r.add_shake(14 * math.exp(-2.5 * t))

    def draw(self, r):
        both = len(self.orbs) == 2
        oa = 1 - smoothstep(0.65, 0.95, self.fusion) if both else 1.0
        for orb in self.orbs.values():
            self._draw_orb(r, orb, oa)
        if both:
            self._draw_fusion(r)
        self.ps.draw(r)
        if self.rel_t is not None:
            self._draw_release(r)
