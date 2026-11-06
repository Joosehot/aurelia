"""Artificial particles and the world they assemble in.

A diamond is a particle species. Its definition (from the bowl's friction record, deterministically):

- k *ports* on its rim, one per grain, in the order the grains' contacts began. Each port has
  - a charge: a 4-dim unit vector from its grain's hidden value. Two ports attract when their charges
    point opposite ways (affinity s = -q_a . q_b > 0) and repel when they point the same way;
  - a dead zone tau, a bias and a gain (see diamond.py);
- an inner structure A: how a push on one port moves the others (the friction work between the
  grains, directed by contact order, see diamond.py);
- a slip rate alpha and a number of sweeps T;
- a mass (1 + its friction heat / cluster_heat: hotter diamonds are heavier) and a radius
  (0.8 + 0.12 k: more ports, bigger particle).

A port is a stick-slip contact: it holds a position p and moves only when the drive on it leaves
its dead zone (the play operator). So a particle has memory.

The world: P copies of one species in a disc, moving and turning under their ports' attraction and
repulsion, soft body contact and Brownian kicks that cool to zero (annealing). When the world has
cooled, two ports of different particles that sit together (distance < BOND_D) with positive affinity
and are each other's nearest such partner become a *bond*. Bonds join particles into molecules.

Signals then travel inside a particle through A and between particles through bonds (weight
0.6 s), so the assembled matter is one big stick-slip network whose wiring nobody designed: the
particles' charges and shapes decided it.

Every world is seeded on its own (from the run seed and the particle's definition hash), so a
world assembles the same whichever batch it is in.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .diamond import K_MAX, definition_from_record, digest

QDIM = 4
P = 32  # particles per world
ASSEMBLY_STEPS = 1500
DT = 0.02
RC = 0.5  # port interaction range
EPS = 4.0  # port force strength
T0 = 0.5  # starting temperature
BOND_D = 0.35
BOND_S = 0.3
BOND_W = 0.6
FILL = 0.35  # fraction of the world's area the particles cover
INPUT_FRACTION = 0.25


def particle_from_record(rec: dict, cluster_heat: float) -> dict:
    d = definition_from_record(rec, cluster_heat)
    H = np.array(rec["hidden"], dtype=np.float64)
    k = d["k"]
    q = H[:, 4:4 + QDIM]
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    phase = math.atan2(float(H[:, 8].sum()), float(H[:, 9].sum()))
    d.update({
        "form": "aurelia-particle/v1",
        "charge": [[float(v) for v in row] for row in q],
        "angle": [float(phase + 2 * math.pi * i / k) for i in range(k)],
        "mass": float(1.0 + sum(rec["heat"]) / cluster_heat),
        "radius": float(0.8 + 0.12 * k),
    })
    return d


def world_seed(run_seed: int, defn: dict, s: int) -> int:
    return (int(digest(defn)[:8], 16) * 31 + run_seed * 7919 + s * 104729) % (2**31)


def _union_sizes(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    sizes = {}
    for a in range(n):
        r = find(a)
        sizes[r] = sizes.get(r, 0) + 1
    return sorted(sizes.values(), reverse=True)


class Worlds:
    """M worlds side by side; world m holds P copies of species defs[m], seeded by seeds[m]."""

    def __init__(self, defs: list[dict], seeds: list[int], dev, record_frames: int = 0):
        self.defs, self.seeds, self.dev = defs, seeds, dev
        M, K, N = len(defs), K_MAX, P * K_MAX
        self.M, self.N = M, N
        f32 = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=dev)
        mask = np.zeros((M, K), np.float32)
        q = np.zeros((M, K, QDIM), np.float32)
        ang = np.zeros((M, K), np.float32)
        A = np.zeros((M, K, K), np.float32)
        tau = np.zeros((M, K), np.float32)
        bias = np.zeros((M, K), np.float32)
        gain = np.zeros((M, K), np.float32)
        for m, d in enumerate(defs):
            k = d["k"]
            mask[m, :k] = 1
            q[m, :k] = d["charge"]
            ang[m, :k] = d["angle"]
            A[m, :k, :k] = d["A"]
            tau[m, :k], bias[m, :k], gain[m, :k] = d["tau"], d["bias"], d["gain"]
        self.mass = f32([d["mass"] for d in defs])
        self.rad = f32([d["radius"] for d in defs])
        self.Rw = self.rad * math.sqrt(P / FILL)
        self.alpha = np.array([d["alpha"] for d in defs], np.float32)
        self.T = np.array([d["T"] for d in defs], np.int64)
        self.port_mask = f32(np.repeat(mask[:, None, :], P, 1).reshape(M, N))
        self.tau = f32(np.tile(tau, (1, P)))
        self.bias = f32(np.tile(bias, (1, P)))
        self.gain = f32(np.tile(gain, (1, P)))
        self.q = f32(np.tile(q, (1, P, 1)))  # [M,N,4]
        self.ang = f32(ang)
        self.A = A

        # Per-world randomness: start positions, orientations, Brownian kicks, input weights.
        pos0 = np.zeros((M, P, 2), np.float32)
        th0 = np.zeros((M, P), np.float32)
        noise = np.zeros((ASSEMBLY_STEPS, M, P, 3), np.float32)
        self.rngs = []
        for m, s in enumerate(seeds):
            g = np.random.default_rng(s)
            pts = []
            while len(pts) < P:
                p = g.uniform(-1, 1, 2)
                if p @ p < 0.85:
                    pts.append(p)
            pos0[m] = np.array(pts) * float(self.Rw[m])
            th0[m] = g.uniform(0, 2 * math.pi, P)
            noise[:, m] = g.standard_normal((ASSEMBLY_STEPS, P, 3))
            self.rngs.append(g)
        self.pos, self.th = f32(pos0), f32(th0)
        self.noise = f32(noise)
        pid = np.repeat(np.arange(P), K)
        self.same = torch.tensor(pid[:, None] == pid[None, :], device=dev)
        self.frames = []
        self.record_frames = record_frames

    def ports(self):
        a = self.th[:, :, None] + self.ang[:, None, :]  # [M,P,K]
        off = self.rad[:, None, None, None] * torch.stack([torch.cos(a), torch.sin(a)], -1)
        return self.pos[:, :, None, :] + off, off  # [M,P,K,2]

    @torch.no_grad()
    def assemble(self) -> None:
        M, K, N = self.M, K_MAX, self.N
        vel = torch.zeros_like(self.pos)
        om = torch.zeros_like(self.th)
        inertia = 0.5 * self.mass * self.rad ** 2
        S = -(self.q @ self.q.transpose(1, 2))  # affinity [M,N,N]
        valid = (self.port_mask[:, :, None] * self.port_mask[:, None, :]) * (~self.same).float()
        S = S * valid
        gamma = 1.0
        for step in range(ASSEMBLY_STEPS):
            temp = T0 * (1 - step / ASSEMBLY_STEPS) ** 2
            pp, off = self.ports()
            ppf = pp.reshape(M, N, 2)
            diff = ppf[:, :, None, :] - ppf[:, None, :, :]
            d2 = (diff * diff).sum(-1)
            fm = EPS * S * torch.exp(-d2 / RC ** 2)
            fport = -(fm[..., None] * diff).sum(2).view(M, P, K, 2)  # attraction toward partners
            fbody = fport.sum(2)
            torque = (off[..., 0] * fport[..., 1] - off[..., 1] * fport[..., 0]).sum(2)
            # soft body contact
            bd = self.pos[:, :, None, :] - self.pos[:, None, :, :]
            bdist = torch.sqrt((bd * bd).sum(-1) + 1e-9)
            ov = torch.relu(2 * self.rad[:, None, None] - bdist) * (1 - torch.eye(P, device=self.dev))
            fbody = fbody + (20.0 * ov / bdist)[..., None].mul(bd).sum(2)
            # wall
            r = torch.sqrt((self.pos * self.pos).sum(-1) + 1e-9)
            wov = torch.relu(r + self.rad[:, None] - self.Rw[:, None])
            fbody = fbody - (20.0 * wov / r)[..., None] * self.pos
            nz = self.noise[step]
            kick = torch.sqrt(torch.tensor(2 * gamma * temp * DT))
            vel = vel + DT * (fbody / self.mass[:, None, None] - gamma * vel) + kick * nz[..., :2] / torch.sqrt(self.mass)[:, None, None]
            om = om + DT * (torque / inertia[:, None] - gamma * om) + kick * nz[..., 2] / torch.sqrt(inertia)[:, None]
            self.pos = self.pos + DT * vel
            self.th = self.th + DT * om
            if self.record_frames and step % self.record_frames == 0:
                self.frames.append((self.pos[0].cpu().numpy().copy(), self.th[0].cpu().numpy().copy()))
        self._bond(S)

    def _bond(self, S):
        M, N = self.M, self.N
        pp, _ = self.ports()
        ppf = pp.reshape(M, N, 2)
        d = torch.cdist(ppf, ppf)
        ok = (S > BOND_S) & (d < BOND_D)
        dm = torch.where(ok, d, torch.full_like(d, 1e9))
        near = dm.argmin(2)  # [M,N]
        has = dm.min(2).values < 1e8
        mutual = has & (near.gather(1, near) == torch.arange(N, device=self.dev)[None, :])
        C = torch.zeros(M, N, N, device=self.dev)
        A = torch.tensor(self.A, device=self.dev)
        for a in range(P):
            sl = slice(a * K_MAX, (a + 1) * K_MAX)
            C[:, sl, sl] = A
        self.C_inner = C.clone()
        i_idx = torch.arange(N, device=self.dev)[None, :].expand(M, N)
        w = BOND_W * S.gather(2, near[..., None])[..., 0] * mutual
        Cb = torch.zeros_like(C)
        Cb.scatter_(2, near[..., None], w[..., None])  # row i gets its partner; mutual => symmetric
        self.C_bonds = Cb
        self.bonds = []
        self.stats = []
        mu = mutual.cpu().numpy()
        nr = near.cpu().numpy()
        for m in range(M):
            idx = np.nonzero(mu[m])[0]
            pairs = sorted({(min(i, nr[m, i]), max(i, nr[m, i])) for i in idx})
            self.bonds.append(pairs)
            edges = [(i // K_MAX, j // K_MAX) for i, j in pairs]
            sizes = _union_sizes(P, edges)
            self.stats.append({"bonds": len(pairs), "bonds_per_particle": round(2 * len(pairs) / P, 3),
                               "molecules": len(sizes), "largest": sizes[0],
                               "free": sum(1 for s in sizes if s == 1)})
        del i_idx

    def coupling(self, variant: str) -> torch.Tensor:
        """The matter's coupling matrix: 'matter' (inner structure + bonds), 'nobond' (inner only)
        or 'shuffled' (the same number of bonds, same weights, between random ports)."""
        if variant == "matter":
            C = self.C_inner + self.C_bonds
        elif variant == "nobond":
            C = self.C_inner.clone()
        else:
            C = self.C_inner.clone()
            for m in range(self.M):
                g = np.random.default_rng(self.seeds[m] + 1)
                act = np.nonzero(self.port_mask[m].cpu().numpy())[0]
                for i, j in self.bonds[m]:
                    w = float(self.C_bonds[m, i, j])
                    while True:
                        a, b = g.choice(act, 2, replace=False)
                        if a // K_MAX != b // K_MAX:
                            break
                    C[m, a, b] += w
                    C[m, b, a] += w
        n = torch.linalg.matrix_norm(C, ord=2)
        return C * torch.clamp(0.95 / (n + 1e-9), max=1.0)[:, None, None]

    def input_weights(self, din: int) -> torch.Tensor:
        """Inputs push the ports of the first quarter of the particles (fixed per world)."""
        W = np.zeros((self.M, din, self.N), np.float32)
        n_in = int(P * INPUT_FRACTION) * K_MAX
        for m in range(self.M):
            g = np.random.default_rng(self.seeds[m] + 2)
            W[m, :, :n_in] = g.uniform(-1, 1, (din, n_in))
        return torch.tensor(W, device=self.dev) * self.port_mask[:, None, :]
