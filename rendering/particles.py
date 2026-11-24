"""NumPy-backed particle system (additive glow sprites)."""
import numpy as np

rng = np.random.default_rng()


class ParticleSystem:
    def __init__(self, capacity=2500):
        self.cap = capacity
        self.n = 0
        self.pos = np.zeros((capacity, 2))
        self.vel = np.zeros((capacity, 2))
        self.life = np.zeros(capacity)
        self.maxlife = np.ones(capacity)
        self.size = np.ones(capacity)
        self.color = np.zeros((capacity, 3))
        self.drag = np.zeros(capacity)
        self.grav = np.zeros(capacity)

    def clear(self):
        self.n = 0

    def add(self, pos, vel, life, size, color, drag=1.5, grav=0.0):
        """Add k particles. All arguments may be arrays of length k or scalars."""
        pos = np.atleast_2d(np.asarray(pos, float))
        k = len(pos)
        take = min(k, self.cap - self.n)
        if take <= 0:
            return
        s = slice(self.n, self.n + take)
        self.pos[s] = pos[:take]
        self.vel[s] = np.broadcast_to(np.asarray(vel, float), (k, 2))[:take]
        lf = np.broadcast_to(np.asarray(life, float), (k,))[:take]
        self.life[s] = lf
        self.maxlife[s] = lf
        self.size[s] = np.broadcast_to(np.asarray(size, float), (k,))[:take]
        self.color[s] = np.broadcast_to(np.asarray(color, float), (k, 3))[:take]
        self.drag[s] = np.broadcast_to(np.asarray(drag, float), (k,))[:take]
        self.grav[s] = np.broadcast_to(np.asarray(grav, float), (k,))[:take]
        self.n += take

    def burst(self, pos, count, speed=(100, 500), life=(0.6, 1.6), size=(3, 8),
              colors=((255, 255, 255),), drag=1.5, grav=0.0, spread=0.0,
              angle=(0, 2 * np.pi)):
        """Radial burst of ``count`` particles with random palette colours."""
        if count <= 0:
            return
        ang = rng.uniform(angle[0], angle[1], count)
        spd = rng.uniform(speed[0], speed[1], count)
        vel = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1)
        p = np.tile(np.asarray(pos, float), (count, 1))
        if spread:
            p = p + rng.normal(0, spread, p.shape)
        pal = np.asarray(colors, float)
        col = pal[rng.integers(0, len(pal), count)]
        self.add(p, vel, rng.uniform(life[0], life[1], count),
                 rng.uniform(size[0], size[1], count), col, drag, grav)

    def update(self, dt):
        n = self.n
        if n == 0:
            return
        self.pos[:n] += self.vel[:n] * dt
        self.vel[:n] *= np.exp(-self.drag[:n] * dt)[:, None]
        self.vel[:n, 1] += self.grav[:n] * dt
        self.life[:n] -= dt
        keep = np.nonzero(self.life[:n] > 0)[0]
        m = len(keep)
        if m != n:
            for arr in (self.pos, self.vel, self.life, self.maxlife, self.size,
                        self.color, self.drag, self.grav):
                arr[:m] = arr[keep]
            self.n = m

    def draw(self, r, bright=1.0):
        n = self.n
        if n == 0:
            return
        u = np.clip(self.life[:n] / self.maxlife[:n], 0, 1)
        cols = np.clip(self.color[:n] * (u ** 1.3 * bright)[:, None], 0, 255)
        r.glow_batch(self.pos[:n], self.size[:n] * (0.45 + 0.55 * u), cols)
