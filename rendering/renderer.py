"""Pygame renderer: additive-glow compositing, camera layer, screen-space FX.

Every effect draws onto a black ``fx`` layer using additive blending; the layer
is then added on top of the dimmed webcam image. Screen flash, zoom pulse and
shake are applied once per frame at the end.
"""
import numpy as np
import pygame

import config


class Renderer:
    def __init__(self, size, fullscreen=True):
        self.W, self.H = self.size = size
        flags = pygame.SCALED | (pygame.FULLSCREEN if fullscreen else 0)
        try:
            self.display = pygame.display.set_mode(size, flags, vsync=1)
        except pygame.error:
            self.display = pygame.display.set_mode(size, flags)
        pygame.display.set_caption("Gesture VFX")
        self.canvas = pygame.Surface(size).convert()
        self.fx = pygame.Surface(size).convert()
        self.cam = None
        self._cache = {}
        self.flash = self.zoom = self.shake = 0.0

    # ---- camera ----------------------------------------------------------
    def set_camera(self, frame_bgr):
        import cv2
        img = frame_bgr
        if (img.shape[1], img.shape[0]) != (self.W, self.H):
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        img = cv2.convertScaleAbs(img, alpha=config.CAMERA_DIM)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.cam = pygame.image.frombuffer(rgb.tobytes(), (self.W, self.H), "RGB")

    # ---- frame lifecycle -------------------------------------------------
    def begin_frame(self):
        self.fx.fill((0, 0, 0))
        self.flash = self.zoom = self.shake = 0.0

    def end_frame(self):
        if self.cam is not None:
            self.canvas.blit(self.cam, (0, 0))
        else:
            self.canvas.fill((8, 8, 14))
        self.canvas.blit(self.fx, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        if self.flash > 0.01:
            v = int(min(1.0, self.flash) * 255)
            self.canvas.fill((v, v, v), special_flags=pygame.BLEND_RGB_ADD)
        if self.zoom > 0.002 or self.shake > 0.5:
            dz = int(self.W * self.zoom / 2)
            sx, sy = (np.random.uniform(-self.shake, self.shake, 2) if self.shake > 0.5
                      else (0, 0))
            if dz > 0:
                big = pygame.transform.scale(self.canvas, (self.W + 2 * dz, self.H + 2 * dz * self.H // self.W))
                self.display.blit(big, (-dz + int(sx), -dz * self.H // self.W + int(sy)))
            else:
                self.display.blit(self.canvas, (int(sx), int(sy)))
        else:
            self.display.blit(self.canvas, (0, 0))

    # ---- screen-space modifiers (max over the frame) ---------------------
    def add_flash(self, v):
        self.flash = max(self.flash, v)

    def add_zoom(self, v):
        self.zoom = max(self.zoom, v)

    def add_shake(self, px):
        self.shake = max(self.shake, px)

    # ---- sprites ---------------------------------------------------------
    @staticmethod
    def _quant_radius(r):
        if r <= 12:
            return r
        if r <= 32:
            return (r + 1) // 2 * 2
        if r <= 64:
            return (r + 3) // 4 * 4
        return (r + 7) // 8 * 8

    def _build(self, r, col):
        yy, xx = np.mgrid[-r:r, -r:r] + 0.5
        d = np.sqrt(xx * xx + yy * yy) / r
        soft = np.clip(1 - d, 0, 1) ** 2
        core = np.clip(1 - d / 0.35, 0, 1) ** 2 * 0.6
        arr = np.empty((2 * r, 2 * r, 3), np.uint8)
        for i in range(3):
            arr[..., i] = np.clip(soft * col[i] + core * 255 * (col[i] / 255.0 * 0.6 + 0.4)
                                  * (max(col) / 255.0), 0, 255)
        return pygame.surfarray.make_surface(arr.swapaxes(0, 1))

    def sprite(self, radius, col):
        """Cached glow sprite; large radii are scaled from a 256px base."""
        radius = max(1, min(int(radius), config.BIG_GLOW_MAX))
        q = tuple(min(255, (int(c) >> 4) * 16 + 8) for c in col)
        if radius > 128:
            key = (256, q)
            base = self._cache.get(key)
            if base is None:
                base = self._cache[key] = self._build(256, q)
            return pygame.transform.scale(base, (2 * radius, 2 * radius))
        rq = self._quant_radius(radius)
        key = (rq, q)
        spr = self._cache.get(key)
        if spr is None:
            if len(self._cache) > 3000:
                self._cache.clear()
            spr = self._cache[key] = self._build(rq, q)
        return spr

    # ---- drawing primitives (all additive onto the fx layer) -------------
    def glow(self, pos, radius, color, alpha=1.0):
        a = min(1.0, alpha)
        if a < 0.02 or radius < 1:
            return
        col = (color[0] * a, color[1] * a, color[2] * a)
        if max(col) < 6:
            return
        spr = self.sprite(radius, col)
        w, h = spr.get_size()
        self.fx.blit(spr, (int(pos[0]) - w // 2, int(pos[1]) - h // 2),
                     special_flags=pygame.BLEND_RGB_ADD)

    def glow_batch(self, pos, radii, colors):
        """Draw many glows; colours are already multiplied by alpha (0..255)."""
        seq = []
        for (x, y), rad, col in zip(pos, radii, colors):
            if rad < 1 or max(col) < 6:
                continue
            spr = self.sprite(rad, col)
            w, h = spr.get_size()
            seq.append((spr, (int(x) - w // 2, int(y) - h // 2), None,
                        pygame.BLEND_RGB_ADD))
        if seq:
            self.fx.blits(seq, False)

    @staticmethod
    def _scale(color, k):
        return (max(0, min(255, int(color[0] * k))), max(0, min(255, int(color[1] * k))),
                max(0, min(255, int(color[2] * k))))

    def _passes(self, color, alpha, width):
        a = min(1.0, alpha)
        white = tuple(c + (255 - c) * 0.6 for c in color)
        return [(self._scale(color, 0.25 * a), max(1, int(width * 3))),
                (self._scale(color, 0.6 * a), max(1, int(width * 1.8))),
                (self._scale(color, a), max(1, int(width))),
                (self._scale(white, a), max(1, int(width * 0.4)))]

    def ring(self, center, radius, width, color, alpha=1.0):
        if radius < 1 or alpha < 0.02:
            return
        c = (int(center[0]), int(center[1]))
        for col, w in self._passes(color, alpha, width):
            pygame.draw.circle(self.fx, col, c, int(radius), min(w, int(radius)))

    def glow_polyline(self, pts, color, width, alpha=1.0, closed=False):
        if len(pts) < 2 or alpha < 0.02:
            return
        p = [(float(x), float(y)) for x, y in pts]
        for col, w in self._passes(color, alpha, width):
            pygame.draw.lines(self.fx, col, closed, p, w)

    def glow_segment(self, p0, p1, color, width, alpha=1.0):
        if alpha < 0.02:
            return
        for col, w in self._passes(color, alpha, width):
            pygame.draw.line(self.fx, col, p0, p1, w)
            if w > 2:
                pygame.draw.circle(self.fx, col, (int(p1[0]), int(p1[1])), w // 2)

    def line(self, p0, p1, color, width=1):
        pygame.draw.line(self.fx, color, p0, p1, width)
