"""The rice bowl: N identical soft grains in a 2D circular bowl, stirred, rubbing, forming diamonds.

Every grain has the same physics (radius, mass, stiffness, friction) and its own hidden value, a
small random vector that the physics never reads. Stirring drives the grains into each other; the
tangential friction between two touching grains dissipates power (|friction force| x |slip
speed|). That power is the only thing that makes diamonds:

- each grain keeps a decaying friction *heat* (the power it took part in, tau = heat_tau);
- each pair keeps a decaying friction *work* (tau = pair_tau) and the time their latest contact
  began.

Every `check_every` steps the hottest grain over `heat_threshold` seeds a cluster: the grain, then
repeatedly the grain that has rubbed hardest against any member (pair work > pair_min), up to
k_max. A cluster of at least k_min grains whose summed heat reaches `cluster_heat` becomes a
diamond: its grains' hidden values, the pair work between them, the order in which their contacts
began, their peak contact forces and the local stir state are written into a *record*. The grains
are spent: their heat and pair memory go to zero and they get fresh hidden values from the
episode's RNG, so the bowl keeps renewing itself.

Determinism: one episode is fully set by (episode seed, stir pattern, config). All randomness comes
from a numpy Generator seeded with the episode seed, and the simulation uses only dense, ordered
torch ops (no atomics, no scatter-add), so the same episode replays bit for bit on the same device.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

STIRS = ("none", "circle", "spiral", "figure8", "shake", "jolt", "mixed")


@dataclass
class BowlConfig:
    n: int = 2000
    hidden_dim: int = 12
    radius: float = 1.0  # grain radius
    packing: float = 0.78  # area fraction of the bowl the grains fill (2D random close packing ~0.84)
    k_n: float = 200.0  # normal contact stiffness
    c_n: float = 2.0  # normal contact damping
    mu: float = 0.5  # Coulomb friction coefficient
    gamma_t: float = 5.0  # tangential viscous cap (friction below the Coulomb limit)
    drag: float = 0.1  # air drag
    dt: float = 0.01
    spoon_radius: float = 5.0
    spoon_k: float = 400.0
    stir_speed: float = 1.0  # multiplies every stir pattern's speed / strength
    heat_tau: float = 1.0
    pair_tau: float = 2.0
    heat_threshold: float = 15.0  # ~ the top 1% of grains under a spiral stir at speed 1
    cluster_heat: float = 80.0
    pair_min: float = 2.0
    pair_rel: float = 0.3  # a grain joins only if it rubbed >= 30% as hard as the cluster's strongest pair
    k_min: int = 4
    k_max: int = 12
    check_every: int = 10  # also how often the neighbour table is rebuilt
    neighbours: int = 20
    skin: float = 1.0
    max_per_check: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


def bowl_radius(cfg: BowlConfig) -> float:
    return math.sqrt(cfg.n / cfg.packing) * cfg.radius


class Bowl:
    def __init__(self, seed: int, stir: str, cfg: BowlConfig | None = None, device: str | None = None):
        if stir not in STIRS:
            raise ValueError(f"stir must be one of {STIRS}")
        self.cfg = cfg = cfg or BowlConfig()
        self.seed = int(seed)
        self.stir = stir
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(self.seed)
        self.R = bowl_radius(cfg)
        n = cfg.n

        # Grains on a jittered hex lattice inside the bowl, n of its sites chosen at random.
        s = 2.05 * cfg.radius
        pts = []
        rows = int(2 * self.R / (s * 0.866)) + 2
        for a in range(-rows, rows + 1):
            for b in range(-rows, rows + 1):
                x = s * (b + 0.5 * (a & 1))
                y = s * 0.866 * a
                if x * x + y * y < (self.R - 1.2 * cfg.radius) ** 2:
                    pts.append((x, y))
        pts = np.array(pts)
        pick = self.rng.choice(len(pts), size=n, replace=False)
        pos = pts[np.sort(pick)] + self.rng.uniform(-0.02, 0.02, size=(n, 2))
        self.x = torch.tensor(pos, dtype=torch.float32, device=self.dev)
        self.v = torch.zeros_like(self.x)
        self.hidden = self.rng.standard_normal((n, cfg.hidden_dim)).astype(np.float32)

        self.heat = torch.zeros(n, device=self.dev)
        self.peak = torch.zeros(n, device=self.dev)
        self.pair = torch.zeros(n, n, device=self.dev)
        self.touch = torch.full((n, n), -1.0, device=self.dev)  # time the latest contact began
        self.incontact = torch.zeros(n, n, dtype=torch.bool, device=self.dev)
        self.eye = torch.eye(n, dtype=torch.bool, device=self.dev)

        self.step_i = 0
        self.t = 0.0
        # stir state
        self.theta = 0.0
        self.phi = 0.0
        self.dir = 1.0
        self.spoon = None  # (x, y, vx, vy)
        self._spoon_prev = None
        self.jolt_next = self.rng.exponential(1.0)
        self.formed: list[dict] = []
        self.nbr = None
        self._rebuild()

    # --- stirring ------------------------------------------------------------------------------

    def _stir(self) -> tuple[torch.Tensor | None, tuple[float, float] | None]:
        """Advance the stir; returns the spoon centre (or None) and a body acceleration (or None)."""
        cfg, R, dt, t = self.cfg, self.R, self.cfg.dt, self.t
        v = 6.0 * cfg.stir_speed
        spoon = None
        accel = None
        pat = self.stir
        if pat == "circle":
            rho = 0.55 * R
            self.theta += dt * v / rho
            spoon = (rho * math.cos(self.theta), rho * math.sin(self.theta))
        elif pat in ("spiral", "mixed"):
            rho = R * (0.5 + 0.3 * math.sin(2 * math.pi * t / 8.0))
            if int(t // 12.0) % 2 == 1:
                self.dir = -1.0
            else:
                self.dir = 1.0
            self.theta += dt * self.dir * v / rho
            spoon = (rho * math.cos(self.theta), rho * math.sin(self.theta))
        elif pat == "figure8":
            a = 0.6 * R
            self.phi += dt * v / a
            spoon = (a * math.sin(self.phi), a * math.sin(self.phi) * math.cos(self.phi))
        elif pat == "shake":
            amp = 40.0 * cfg.stir_speed * math.sin(2 * math.pi * 0.8 * t)
            psi = 0.3 * t
            accel = (amp * math.cos(psi), amp * math.sin(psi))
        if pat in ("jolt", "mixed") and t >= self.jolt_next:
            self.jolt_next = t + self.rng.exponential(1.0)
            c = self.rng.uniform(-0.7, 0.7, size=2) * R
            ang = self.rng.uniform(0, 2 * math.pi)
            kick = torch.tensor([math.cos(ang), math.sin(ang)], dtype=torch.float32, device=self.dev)
            ct = torch.tensor(c, dtype=torch.float32, device=self.dev)
            near = ((self.x - ct) ** 2).sum(1) < 64.0
            self.v = self.v + near[:, None].float() * kick * (10.0 * cfg.stir_speed)
        if spoon is not None:
            prev = self._spoon_prev or spoon
            self.spoon = (spoon[0], spoon[1], (spoon[0] - prev[0]) / dt, (spoon[1] - prev[1]) / dt)
            self._spoon_prev = spoon
        else:
            self.spoon = None
        return spoon, accel

    # --- one step ------------------------------------------------------------------------------

    def _rebuild(self) -> None:
        """Write the neighbour table's pair memory back to the dense matrices, then rebuild it.
        Every row's neighbours are distinct, so the scatters never collide (deterministic)."""
        cfg = self.cfg
        if self.nbr is not None:
            self.pair = self.pair * math.exp(-cfg.check_every * cfg.dt / cfg.pair_tau)
            self.pair.scatter_(1, self.nbr, self.pair.gather(1, self.nbr) + self.acc_t)
            self.touch.scatter_(1, self.nbr, self.touch_t)
            self.incontact.scatter_(1, self.nbr, self.inc_t)
        x = self.x
        d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1) + self.eye * 1e9
        vals, nbr = torch.topk(-d2, cfg.neighbours, dim=1)
        self.nbr = nbr
        self.valid = (-vals) < (2 * cfg.radius + cfg.skin) ** 2
        self.touch_t = self.touch.gather(1, nbr)
        self.inc_t = self.incontact.gather(1, nbr)
        self.acc_t = torch.zeros_like(self.touch_t)

    def step(self) -> None:
        cfg = self.cfg
        r2 = 2 * cfg.radius
        spoon, accel = self._stir()
        x, v = self.x, self.v
        nbr = self.nbr

        xj, vj = x[nbr], v[nbr]  # [n, K, 2]
        dx = x[:, None, 0] - xj[..., 0]
        dy = x[:, None, 1] - xj[..., 1]
        d = torch.sqrt(dx * dx + dy * dy + 1e-12)
        ov = torch.relu(r2 - d) * self.valid
        contact = ov > 0
        nx, ny = dx / d, dy / d
        dvx = v[:, None, 0] - vj[..., 0]
        dvy = v[:, None, 1] - vj[..., 1]
        vn = dvx * nx + dvy * ny
        fn = torch.relu(cfg.k_n * ov - cfg.c_n * vn * contact)
        tx, ty = dvx - vn * nx, dvy - vn * ny
        vt = torch.sqrt(tx * tx + ty * ty + 1e-12)
        ft = torch.minimum(cfg.mu * fn, cfg.gamma_t * vt) * contact
        fx = (fn * nx - ft * tx / vt).sum(1)
        fy = (fn * ny - ft * ty / vt).sum(1)
        power = ft * vt  # friction power per pair (each pair seen from both sides)

        # wall
        rad = torch.sqrt((x * x).sum(1) + 1e-12)
        wov = torch.relu(rad + cfg.radius - self.R)
        wn = x / rad[:, None]
        wvn = (v * wn).sum(1)
        wf = torch.relu(cfg.k_n * wov + cfg.c_n * wvn * (wov > 0))
        wt = v - wvn[:, None] * wn
        wts = torch.sqrt((wt * wt).sum(1) + 1e-12)
        wft = torch.minimum(cfg.mu * wf, cfg.gamma_t * wts) * (wov > 0)
        f = torch.stack([fx, fy], 1) - wf[:, None] * wn - wft[:, None] * wt / wts[:, None]

        # spoon (a kinematic disc)
        if spoon is not None:
            sp = torch.tensor(self.spoon, dtype=torch.float32, device=self.dev)
            sd = x - sp[:2]
            sdist = torch.sqrt((sd * sd).sum(1) + 1e-12)
            sov = torch.relu(cfg.radius + cfg.spoon_radius - sdist)
            sn = sd / sdist[:, None]
            rv = v - sp[2:]
            svn = (rv * sn).sum(1)
            sf = torch.relu(cfg.spoon_k * sov - cfg.c_n * svn * (sov > 0))
            st = rv - svn[:, None] * sn
            sts = torch.sqrt((st * st).sum(1) + 1e-12)
            sft = torch.minimum(cfg.mu * sf, cfg.gamma_t * sts) * (sov > 0)
            f = f + sf[:, None] * sn - sft[:, None] * st / sts[:, None]
        if accel is not None:
            f = f + torch.tensor(accel, dtype=torch.float32, device=self.dev)

        f = f - cfg.drag * v
        self.v = v + cfg.dt * f
        self.x = x + cfg.dt * self.v

        # friction memory (pair work is decayed per block of check_every steps, in _rebuild)
        dt = cfg.dt
        self.heat = self.heat * math.exp(-dt / cfg.heat_tau) + dt * 0.5 * power.sum(1)
        self.acc_t = self.acc_t + dt * power
        self.peak = torch.maximum(self.peak * math.exp(-dt / cfg.heat_tau), fn.sum(1))
        began = contact & ~self.inc_t
        self.touch_t = torch.where(began, torch.full_like(self.touch_t, self.t), self.touch_t)
        self.inc_t = contact
        self.last_power = power

        self.step_i += 1
        self.t = self.step_i * dt
        if self.step_i % cfg.check_every == 0:
            self._rebuild()
            self._check()

    # --- diamonds ------------------------------------------------------------------------------

    def _check(self) -> None:
        cfg = self.cfg
        heat = self.heat.cpu().numpy()
        order = np.argsort(-heat, kind="stable")
        made = 0
        for g in order[:16]:
            if heat[g] < cfg.heat_threshold or made >= cfg.max_per_check:
                break
            members = self._cluster(int(g))
            if members is None:
                continue
            if heat[members].sum() < cfg.cluster_heat:
                continue
            self._form(members)
            heat[members] = 0.0
            made += 1
        if made:  # the spent grains' contact times were reset in the dense matrix
            self.touch_t = self.touch.gather(1, self.nbr)

    def _cluster(self, g: int) -> list[int] | None:
        cfg = self.cfg
        members = [g]
        best = self.pair[g].clone()
        best[g] = -1.0
        strongest = None
        while len(members) < cfg.k_max:
            j = int(torch.argmax(best))
            w = float(best[j])
            strongest = strongest or w
            if w < max(cfg.pair_min, cfg.pair_rel * strongest):
                break
            members.append(j)
            best = torch.maximum(best, self.pair[j])
            best[members] = -1.0
        if len(members) < cfg.k_min:
            return None
        return members

    def _form(self, members: list[int]) -> None:
        cfg = self.cfg
        idx = torch.tensor(members, device=self.dev)
        W = self.pair[idx][:, idx].cpu().numpy().astype(np.float64)
        T = self.touch[idx][:, idx].cpu().numpy().astype(np.float64)
        heat = self.heat[idx].cpu().numpy().astype(np.float64)
        peak = self.peak[idx].cpu().numpy().astype(np.float64)
        xs = self.x[idx].cpu().numpy().astype(np.float64)
        vs = self.v[idx].cpu().numpy().astype(np.float64)
        k = len(members)

        # The order of contacts: every rubbing pair in the cluster, by when its latest contact began.
        events = []
        for a in range(k):
            for b in range(a + 1, k):
                if W[a, b] >= cfg.pair_min and T[a, b] >= 0:
                    events.append((T[a, b], a, b, W[a, b]))
        events.sort(key=lambda e: (e[0], e[1], e[2]))
        first = {}
        for t, a, b, _ in events:
            for q in (a, b):
                first.setdefault(q, len(first))
        for q in range(k):  # members that only rubbed before their latest contact window
            first.setdefault(q, len(first))
        perm = sorted(range(k), key=lambda q: first[q])
        inv = {q: i for i, q in enumerate(perm)}

        c = xs.mean(0)
        vbar = vs.mean(0)
        rel_r, rel_v = xs - c, vs - vbar
        L = float(np.sum(rel_r[:, 0] * rel_v[:, 1] - rel_r[:, 1] * rel_v[:, 0]))
        denom = float(np.sum(np.linalg.norm(rel_r, axis=1) * np.linalg.norm(rel_v, axis=1))) + 1e-9
        forces = peak
        rec = {
            "nodes": [int(members[q]) for q in perm],
            "hidden": [[float(h) for h in self.hidden[members[q]]] for q in perm],
            "pair_work": [[float(W[perm[i], perm[j]]) for j in range(k)] for i in range(k)],
            "events": [[inv[a], inv[b], float(t), float(w)] for t, a, b, w in events],
            "heat": [float(heat[q]) for q in perm],
            "peak_force": [float(peak[q]) for q in perm],
            "stir": {
                "pattern": self.stir,
                "speed": cfg.stir_speed,
                "time": round(self.t, 6),
                "spoon": None if self.spoon is None else [float(s) for s in self.spoon],
                "swirl": abs(L) / denom,
                "mean_speed": float(np.linalg.norm(vs, axis=1).mean()),
                "force_cv": float(forces.std() / (forces.mean() + 1e-9)),
                "centroid": [float(c[0]), float(c[1])],
            },
            "step": self.step_i,
        }
        self.formed.append(rec)

        # The grains are spent.
        self.heat[idx] = 0.0
        self.peak[idx] = 0.0
        self.pair[idx, :] = 0.0
        self.pair[:, idx] = 0.0
        self.touch[idx, :] = -1.0
        self.touch[:, idx] = -1.0
        self.hidden[members] = self.rng.standard_normal((k, cfg.hidden_dim)).astype(np.float32)

    def run(self, steps: int, until_diamonds: int | None = None) -> list[dict]:
        for _ in range(steps):
            self.step()
            if until_diamonds is not None and len(self.formed) >= until_diamonds:
                break
        return self.formed
