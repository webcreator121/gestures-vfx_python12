"""Easing helpers and small reusable animated entities."""
import math

import numpy as np


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c0, c1, t):
    t = clamp(t)
    return tuple(lerp(c0[i], c1[i], t) for i in range(3))


def smoothstep(a, b, x):
    t = clamp((x - a) / (b - a)) if b != a else 1.0
    return t * t * (3 - 2 * t)


def ease_out_cubic(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_out_back(t):
    t = clamp(t)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ellipse_points(center, rx, ry, rot, n=40):
    """Points of a rotated ellipse (used for spinning energy rings)."""
    t = np.linspace(0, 2 * math.pi, n)
    x, y = rx * np.cos(t), ry * np.sin(t)
    c, s = math.cos(rot), math.sin(rot)
    return np.stack([center[0] + x * c - y * s, center[1] + x * s + y * c], 1)


class Shockwave:
    """An expanding, fading ring."""

    def __init__(self, pos, max_r=700, width=12, color=(255, 255, 255),
                 life=1.4, delay=0.0):
        self.pos, self.max_r, self.width = pos, max_r, width
        self.color, self.life, self.delay = color, life, delay
        self.age = 0.0

    @property
    def dead(self):
        return self.age - self.delay >= self.life

    def update(self, dt):
        self.age += dt

    def draw(self, r, alpha=1.0):
        a = self.age - self.delay
        if a < 0 or a >= self.life:
            return
        u = a / self.life
        radius = self.max_r * ease_out_cubic(u)
        r.ring(self.pos, radius, max(2, self.width * (1 - 0.6 * u)),
               self.color, alpha * (1 - u) ** 1.4)
