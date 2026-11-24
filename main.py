"""Gesture-controlled supernatural VFX - entry point.

    python main.py [--windowed] [--camera N]
"""
import argparse
import sys
import time

import pygame

import config
from effects.air_drawing import AirDrawing
from effects.hollow_purple import HollowPurple
from effects.magical_explosion import MagicalExplosion
from rendering.hud import HUD, draw_hands
from rendering.renderer import Renderer
from tracking.gesture_manager import GestureManager
from tracking.hand_tracker import HandTracker


def parse_args():
    ap = argparse.ArgumentParser(description="Gesture VFX")
    ap.add_argument("--windowed", action="store_true", help="run in a window")
    ap.add_argument("--camera", type=int, default=None, help="webcam index (default 0)")
    return ap.parse_args()


def main():
    args = parse_args()
    pygame.init()
    renderer = Renderer(config.RENDER_SIZE, fullscreen=config.FULLSCREEN and not args.windowed)
    tracker = HandTracker(args.camera)
    tracker.start()

    effects = [AirDrawing(), MagicalExplosion(), HollowPurple()]   # add new abilities here
    for e in effects:
        e.enabled = config.ENABLED.get(e.name, True)
    manager = GestureManager(effects, config.RENDER_SIZE)
    hud = HUD(renderer.display, effects)
    air = effects[0]

    clock = pygame.time.Clock()
    show_skeleton, calibrating = True, False
    last_seq, fps = -1, 0.0
    running = True
    try:
        while running:
            dt = min(clock.tick(config.TARGET_FPS) / 1000.0, 0.05)
            fps = fps * 0.9 + (1.0 / max(dt, 1e-4)) * 0.1 if fps else 60.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    i = hud.click(ev.pos)
                    if i is not None:
                        manager.set_enabled(effects[i], not effects[i].enabled)
                elif ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k == pygame.K_ESCAPE:
                        running = False
                    elif k == pygame.K_f:
                        pygame.display.toggle_fullscreen()
                    elif k == pygame.K_c:
                        calibrating = not calibrating
                    elif k == pygame.K_h:
                        show_skeleton = not show_skeleton
                    elif k == pygame.K_k:
                        air.cycle_color()
                    elif pygame.K_1 <= k <= pygame.K_9 and k - pygame.K_1 < len(effects):
                        e = effects[k - pygame.K_1]
                        manager.set_enabled(e, not e.enabled)
                    elif calibrating:
                        _calibration_key(k)

            res = tracker.latest()
            if res.frame is not None and res.seq != last_seq:
                last_seq = res.seq
                renderer.set_camera(res.frame)

            renderer.begin_frame()
            manager.update(res, dt, time.perf_counter())
            if show_skeleton or calibrating:
                draw_hands(renderer, manager.hands)
            manager.draw(renderer)
            renderer.end_frame()
            hud.draw(fps, res, manager, calibrating, show_skeleton)
            pygame.display.flip()
    finally:
        tracker.stop()
        pygame.quit()


def _calibration_key(k):
    if k == pygame.K_LEFTBRACKET:
        config.PINCH_ON = max(0.10, config.PINCH_ON - 0.02)
    elif k == pygame.K_RIGHTBRACKET:
        config.PINCH_ON = min(0.70, config.PINCH_ON + 0.02)
    elif k == pygame.K_MINUS:
        config.EXT_MARGIN = max(0.90, config.EXT_MARGIN - 0.01)
    elif k == pygame.K_EQUALS:
        config.EXT_MARGIN = min(1.40, config.EXT_MARGIN + 0.01)
    elif k == pygame.K_COMMA:
        config.LM_SMOOTH = max(0.20, config.LM_SMOOTH - 0.05)
    elif k == pygame.K_PERIOD:
        config.LM_SMOOTH = min(0.95, config.LM_SMOOTH + 0.05)
    elif k == pygame.K_s:
        config.save_calibration()
    elif k == pygame.K_r:
        config.PINCH_ON, config.PINCH_OFF, config.EXT_MARGIN, config.LM_SMOOTH = 0.32, 0.45, 1.05, 0.55
        return
    config.PINCH_OFF = config.PINCH_ON + 0.13


if __name__ == "__main__":
    sys.exit(main())
