"""Minimal futuristic HUD: status, ability panel, calibration overlay, hand skeleton."""
import numpy as np
import pygame

import config
from tracking.gesture_recognition import HAND_CONNECTIONS

CYAN, DIM, OK, BAD = (90, 220, 255), (120, 150, 170), (110, 255, 170), (255, 110, 110)


def draw_hands(r, hands):
    """Subtle skeleton + glowing fingertip dots, drawn on the additive fx layer."""
    for h in hands:
        for a, b in HAND_CONNECTIONS:
            r.line(h.pts[a], h.pts[b], (25, 70, 90), 2)
        tips = h.pts[[4, 8, 12, 16, 20]]
        r.glow_batch(h.pts, np.full(21, 5.0), np.tile((60, 150, 190), (21, 1)))
        r.glow_batch(tips, np.full(5, 11.0), np.tile((120, 230, 255), (5, 1)))


class HUD:
    def __init__(self, display, effects):
        self.d, self.effects = display, effects
        self.f = pygame.font.SysFont("consolas,couriernew,monospace", 16)
        self.fs = pygame.font.SysFont("consolas,couriernew,monospace", 13)
        self.fb = pygame.font.SysFont("consolas,couriernew,monospace", 30, bold=True)
        self.rects = []

    def _text(self, s, pos, color=CYAN, font=None):
        surf = (font or self.f).render(s, True, color)
        self.d.blit(surf, pos)
        return surf.get_width()

    def click(self, pos):
        """Return the effect index whose panel row was clicked, else None."""
        for i, rect in enumerate(self.rects):
            if rect.collidepoint(pos):
                return i
        return None

    def draw(self, fps, result, manager, calibrating, show_skeleton):
        W, H = self.d.get_size()
        self._text(f"FPS {fps:5.1f}", (14, 10), OK if fps >= 40 else BAD)
        track_ok = result.camera_ok and bool(manager.hands)
        self._text(f"TRACKING: {result.status.upper()}", (14, 30),
                   OK if track_ok else (BAD if not result.camera_ok else DIM))
        if result.camera_ok:
            self._text(f"detect {result.detect_ms:4.1f} ms", (14, 50), DIM, self.fs)
        # detected gestures + active abilities
        y = 74
        for h in manager.hands:
            self._text(f"HAND {h.id} [{h.side}] {h.pose}", (14, y), CYAN)
            y += 20
        act = manager.active_names()
        self._text("ACTIVE: " + (", ".join(act) if act else "-"), (14, y + 6), (255, 210, 120))
        for e in manager.effects:
            if e.enabled and e.status():
                y += 20
                self._text(f"  {e.name}: {e.status()}", (14, y + 6), DIM, self.fs)
        # control panel (click rows or press 1..n)
        self.rects = []
        px, py = W - 260, 10
        self._text("ABILITIES", (px, py), DIM, self.fs)
        for i, e in enumerate(self.effects):
            row = pygame.Rect(px - 6, py + 20 + i * 26, 252, 24)
            self.rects.append(row)
            pygame.draw.rect(self.d, (20, 40, 55) if e.enabled else (35, 25, 25), row, border_radius=4)
            pygame.draw.rect(self.d, CYAN if e.enabled else (90, 60, 60), row, 1, border_radius=4)
            self._text(f"{i + 1}  {e.name}", (px, row.y + 4), CYAN if e.enabled else DIM)
            self._text("ON" if e.enabled else "OFF", (row.right - 38, row.y + 4),
                       OK if e.enabled else BAD)
        self._text("C calib | H skeleton | K colour | F fullscreen | ESC quit",
                   (14, H - 22), DIM, self.fs)
        if not result.camera_ok:
            msg = self.fb.render("NO WEBCAM / WAITING FOR CAMERA", True, BAD)
            self.d.blit(msg, ((W - msg.get_width()) // 2, H // 2 - 20))
            sub = self.f.render(result.status, True, DIM)
            self.d.blit(sub, ((W - sub.get_width()) // 2, H // 2 + 20))
        if calibrating:
            self._calibration(manager, W, H)

    def _calibration(self, manager, W, H):
        box = pygame.Rect(W // 2 - 300, 70, 600, 210 + 62 * len(manager.hands))
        s = pygame.Surface(box.size, pygame.SRCALPHA)
        s.fill((5, 15, 25, 215))
        self.d.blit(s, box.topleft)
        pygame.draw.rect(self.d, CYAN, box, 1, border_radius=6)
        x, y = box.x + 16, box.y + 12
        self._text("CALIBRATION MODE", (x, y), (255, 210, 120), self.fb)
        rows = [f"[ / ]  pinch threshold   {config.PINCH_ON:.2f}  (release {config.PINCH_OFF:.2f})",
                f"- / =  finger-extend margin {config.EXT_MARGIN:.2f}  (higher = stricter)",
                f", / .  landmark smoothing  {config.LM_SMOOTH:.2f}  (higher = snappier)",
                "S  save to calibration.json   R  reset defaults"]
        for i, t in enumerate(rows):
            self._text(t, (x, y + 46 + i * 20), CYAN, self.fs)
        y += 140
        if not manager.hands:
            self._text("Show a hand: try POINT, OPEN, FIST and an OK-sign (pinch).", (x, y), DIM, self.fs)
        for h in manager.hands:
            fl = "".join(n if on else "." for n, on in zip("TIMRP", h.fingers))
            self._text(f"hand {h.id}: pose={h.pose:<6} fingers={fl}  palm={h.size:5.1f}px",
                       (x, y), CYAN, self.fs)
            ok = h.pinch_ratio < config.PINCH_ON
            self._text(f"   pinch ratio {h.pinch_ratio:4.2f}  (on<{config.PINCH_ON:.2f})  curl "
                       + " ".join(f"{c:3.0f}" for c in h.curl), (x, y + 18),
                       OK if ok else DIM, self.fs)
            y += 44
