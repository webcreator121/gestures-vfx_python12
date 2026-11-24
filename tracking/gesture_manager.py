"""Central gesture manager: runs recognition once, then drives every ability."""
import config
from effects.base_effect import GestureContext
from tracking.gesture_recognition import GestureRecognizer


class GestureManager:
    def __init__(self, effects, size):
        self.effects = sorted(effects, key=lambda e: e.layer)
        self.size = size
        self.recognizer = GestureRecognizer()
        self.shared = {}
        self.hands = []
        self.ctx = None
        self._seq = -1

    def set_enabled(self, effect, on):
        effect.enabled = on
        config.ENABLED[effect.name] = on
        if not on:
            effect.reset()

    def update(self, result, dt, t):
        if result.seq != self._seq:
            self._seq = result.seq
            self.hands = self.recognizer.process(result.hands, self.size, t)
        if not result.camera_ok or t - result.t > config.STALE_TIME:
            self.hands = []
        ctx = GestureContext(t, dt, self.size, self.hands, self.shared)
        for e in self.effects:
            if e.enabled:
                e.handle(ctx)
        for e in self.effects:
            if e.enabled:
                e.update(dt)
        self.ctx = ctx
        return ctx

    def draw(self, renderer):
        for e in self.effects:
            if e.enabled:
                e.draw(renderer)

    def active_names(self):
        return [e.name for e in self.effects if e.enabled and e.active]
