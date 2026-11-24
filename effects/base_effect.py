"""Base class and per-frame context shared by all abilities."""
from dataclasses import dataclass, field


@dataclass
class GestureContext:
    t: float                       # time (perf_counter seconds)
    dt: float                      # frame delta time
    size: tuple                    # (W, H) render size
    hands: list                    # list[Hand]
    shared: dict                   # persists across frames; used for cross-effect state
    events: list = field(default_factory=list)

    def emit(self, name, **data):
        self.events.append((name, data))

    def events_named(self, name):
        return [d for n, d in self.events if n == name]


class BaseEffect:
    """Subclass this to add a new ability.

    handle(ctx)  - read gestures, change state          (every frame)
    update(dt)   - advance animations / particles        (every frame)
    draw(r)      - render with the Renderer API          (every frame)
    """
    name = "Effect"
    layer = 0                      # draw order, low first

    def __init__(self):
        self.enabled = True

    def handle(self, ctx):
        pass

    def update(self, dt):
        pass

    def draw(self, r):
        pass

    def reset(self):
        """Clear all live state (called when the ability is disabled)."""

    @property
    def active(self):
        return False

    def status(self):
        return ""
