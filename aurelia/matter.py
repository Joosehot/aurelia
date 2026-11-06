"""Testing a particle: does the matter it assembles into compute anything useful?

The assembled matter of one species (P = 32 particles, up to 384 ports) is driven by inputs
pushing the ports of a quarter of the particles. Every input step, every port relaxes T sweeps:

    d   = W_in u_t + b + C p          (C: inner structures + bonds, see particle.py)
    p  <- p + alpha (clamp(p, d - tau, d + tau) - p)

and the matter's state is gain * p. Nothing in the matter is trained. Only a linear readout of all
port states is fitted (ridge regression), so the question is purely whether the matter's own
dynamics make the task easy to read off.

Comparisons, on the same tasks with the same readout and the same number of units (32 k):
- an echo state network (ESN), the standard random recurrent reservoir (tanh, sparse random
  weights), with its input scale and spectral radius *tuned per task on the test score* over a 3 x 3
  grid, 3 seeds. The matter gets the same input-scale grid (0.3, 1, 3), also chosen on the test
  score, per particle and task; the controls below use the scale the matter chose;
- the same particles with no bonds (each particle alone): does assembling matter?
- the same particles with the same number of bonds placed at random: does the wiring the particles
  chose matter?

Tasks (each a fixed set, drawn once from DATA_SEED):
- memcap:  memory capacity. Random input u in [-0.8, 0.8]; read u(t-d) for d = 1..15. Score = the
           sum of the R^2 over delays / 15.
- xor:     random bits; read XOR(b(t-1), b(t-2)). Accuracy.
- narma10: the NARMA-10 system (nonlinear, 10-step memory). R^2.
- latch:   10-step sequences, four bits shown once at a random step with a store flag, noise
           otherwise, recall at the end. Bit accuracy.
- mnist14: MNIST digits pooled to 14 x 14 and fed row by row (14 steps); read the last state.
           2000 train / 1000 test. Accuracy.

A task counts for keeping when the matter beats or matches the ESN (within max(std, 0.01)) AND is
clearly above chance (by 0.05). A particle is KEPT when one task counts. "assembly matters" marks
tasks where the matter also beats both controls by 0.01.
"""

from __future__ import annotations

import gzip
import os

import numpy as np
import torch

from .diamond import K_MAX, digest
from .particle import INPUT_FRACTION, P, Worlds, world_seed

DATA_SEED = 20261106
MNIST_DIR = os.environ.get("AURELIA_MNIST", r"C:\Users\joose\Neuras\data\mnist")
TOL = 0.01
SEEDS = 2
BASE_SEEDS = 3
CHUNK = 64
TASK_NAMES = ("memcap", "xor", "narma10", "latch", "mnist14")
INPUT_SCALES = (0.3, 1.0, 3.0)
CHANCE = {"memcap": 0.0, "xor": 0.5, "narma10": 0.0, "latch": 0.5, "mnist14": 0.1}


# --- tasks ---------------------------------------------------------------------------------------

class Task:
    def __init__(self, name, kind, u, y, split, metric):
        self.name, self.kind, self.u, self.y, self.split, self.metric = name, kind, u, y, split, metric
        self.chance = CHANCE[name]


def _stream_tasks(rng):
    out = []
    L, wash, ntr = 1200, 200, 600
    u = rng.uniform(-0.8, 0.8, L).astype(np.float32)
    y = np.stack([np.roll(u, d) for d in range(1, 16)], 1)
    out.append(Task("memcap", "stream", u[:, None], y, (wash, wash + ntr), "memcap"))
    b = rng.choice([-1.0, 1.0], L).astype(np.float32)
    y = (np.roll(b, 1) * np.roll(b, 2) < 0).astype(np.float32)[:, None]  # XOR of the two bits before
    out.append(Task("xor", "stream", b[:, None], y, (wash, wash + ntr), "acc_bin"))
    L2 = 2200
    u = rng.uniform(0, 0.5, L2)
    yy = np.zeros(L2)
    for t in range(9, L2 - 1):
        yy[t + 1] = 0.3 * yy[t] + 0.05 * yy[t] * yy[t - 9:t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
    out.append(Task("narma10", "stream", (u * 2 - 0.5).astype(np.float32)[:, None], yy.astype(np.float32)[:, None],
                    (200, 1400), "r2"))
    return out


def _latch(rng, m=2000, L=10):
    bits = rng.choice([-1.0, 1.0], size=(m, 4))
    x = np.zeros((m, L, 6), dtype=np.float32)
    x[:, :L - 1, :4] = rng.normal(0, 0.7, size=(m, L - 1, 4))
    at = rng.integers(0, L - 3, size=m)
    x[np.arange(m), at, :4] = bits
    x[np.arange(m), at, 4] = 1.0
    x[:, L - 1, 5] = 1.0
    return Task("latch", "batch", x, (bits > 0).astype(np.float32), 1500, "acc_bits")


def _idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    if data[3] == 3:
        n, r, c = (int.from_bytes(data[i:i + 4], "big") for i in (4, 8, 12))
        return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, r, c)
    return np.frombuffer(data, dtype=np.uint8, offset=8)


def _mnist(rng):
    xa = _idx(os.path.join(MNIST_DIR, "train-images-idx3-ubyte.gz"))
    ya = _idx(os.path.join(MNIST_DIR, "train-labels-idx1-ubyte.gz"))
    xb = _idx(os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte.gz"))
    yb = _idx(os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte.gz"))
    i = rng.choice(len(xa), 2000, replace=False)
    j = rng.choice(len(xb), 1000, replace=False)
    x = np.concatenate([xa[i], xb[j]]).astype(np.float32) / 255.0
    x = x.reshape(-1, 14, 2, 14, 2).mean((2, 4))  # 14 x 14
    x = (x - 0.13) / 0.3
    y = np.concatenate([ya[i], yb[j]])
    return Task("mnist14", "batch", x, np.eye(10, dtype=np.float32)[y], 2000, "acc_class")


def load_tasks() -> list[Task]:
    rng = np.random.default_rng(DATA_SEED)
    return _stream_tasks(rng) + [_latch(rng), _mnist(rng)]


# --- running a reservoir -------------------------------------------------------------------------

@torch.no_grad()
def run_matter(C, Win, tau, bias, gain, alpha, T, u, last=False):
    """C [M,N,N], Win [M,din,N], tau/bias/gain [M,N], alpha [M], T [M]; u [B,L,din] (shared).
    Returns states [M,B,L,N] ([M,B,1,N] with last=True)."""
    M, N = C.shape[0], C.shape[1]
    B, L, _ = u.shape
    Ct = C.transpose(1, 2)
    Tmax = int(T.max())
    al = torch.tensor(np.where(np.arange(Tmax)[None, :] < T[:, None], alpha[:, None], 0.0).astype(np.float32),
                      device=C.device)
    tau, bias, gain = tau[:, None, :], bias[:, None, :], gain[:, None, :]
    p = torch.zeros(M, B, N, device=C.device)
    out = []
    for t in range(L):
        drive_in = torch.matmul(u[:, t], Win) + bias  # [M,B,N]
        for s in range(Tmax):
            d = drive_in + torch.bmm(p, Ct)
            tgt = torch.minimum(torch.maximum(p, d - tau), d + tau)
            p = p + al[:, s, None, None] * (tgt - p)
        if not last or t == L - 1:
            out.append(p * gain)
    return torch.stack(out, 2)


@torch.no_grad()
def run_esn(W, Win, u, last=False):
    M, N = W.shape[0], W.shape[1]
    B, L, _ = u.shape
    x = torch.zeros(M, B, N, device=W.device)
    Wt = W.transpose(1, 2)
    out = []
    for t in range(L):
        x = torch.tanh(torch.matmul(u[:, t], Win) + torch.bmm(x, Wt))
        if not last or t == L - 1:
            out.append(x)
    return torch.stack(out, 2)


def _ridge_score(F_tr, Y_tr, F_te, Y_te, metric, mask=None):
    """F [M,n,N] (float32), Y [n,o]. Returns [M] scores."""
    M = F_tr.shape[0]
    one = lambda F: torch.cat([F, torch.ones(*F.shape[:2], 1, device=F.device)], 2).double()
    Xtr, Xte = one(F_tr), one(F_te)
    G = Xtr.transpose(1, 2) @ Xtr
    lam = 1e-4 * G.diagonal(dim1=1, dim2=2).mean(1).clamp(min=1e-6)
    G = G + lam[:, None, None] * torch.eye(G.shape[1], device=G.device, dtype=G.dtype)
    Yt = torch.tensor(Y_tr, device=F_tr.device, dtype=torch.float64).expand(M, -1, -1)
    Wout = torch.linalg.solve(G, Xtr.transpose(1, 2) @ Yt)
    pred = (Xte @ Wout).float().cpu().numpy()
    Y = Y_te
    if metric == "memcap":
        var = Y.var(0)
        r2 = 1 - ((pred - Y) ** 2).mean(1) / var
        return np.clip(r2, 0, 1).sum(1) / Y.shape[1]
    if metric == "r2":
        return np.clip(1 - ((pred - Y) ** 2).mean((1, 2)) / Y.var(), -1, 1)
    if metric == "acc_bin":
        return ((pred[..., 0] > 0.5) == (Y[:, 0] > 0.5)).mean(1)
    if metric == "acc_bits":
        return ((pred > 0.5) == (Y > 0.5)).mean((1, 2))
    return (pred.argmax(2) == Y.argmax(1)).mean(1)


def score_states(task: Task, S: torch.Tensor) -> np.ndarray:
    """S: [M,B,L,N] states for this task's input."""
    if task.kind == "stream":
        a, b = task.split
        X = S[:, 0]
        return _ridge_score(X[:, a:b], task.y[a:b], X[:, b:], task.y[b:], task.metric)
    X = S[:, :, -1]
    n = task.split
    return _ridge_score(X[:, :n], task.y[:n], X[:, n:], task.y[n:], task.metric)


def task_input(task: Task, dev) -> torch.Tensor:
    u = task.u if task.kind == "batch" else task.u[None]
    return torch.tensor(u, device=dev)


# --- baselines ------------------------------------------------------------------------------------

def esn_baselines(tasks: list[Task], dev, cache_dir: str) -> dict:
    key = digest({"v": 1, "tasks": [t.name for t in tasks], "P": P, "seeds": BASE_SEEDS})
    path = os.path.join(cache_dir, f"esn-{key}.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            return json.load(f)
    out = {t.name: {} for t in tasks}
    grid = [(si, rho) for si in (0.3, 1.0, 3.0) for rho in (0.5, 0.9, 1.2)]
    for k in range(4, K_MAX + 1):
        n = P * k
        Ws, Wins = [], []
        for gi, (si, rho) in enumerate(grid):
            for s in range(BASE_SEEDS):
                g = np.random.default_rng(5000 + 97 * k + 13 * gi + s)
                W = g.standard_normal((n, n)) * (g.random((n, n)) < 0.1)
                W *= rho / max(1e-9, np.abs(np.linalg.eigvals(W)).max())
                Ws.append(W)
                Wins.append(g)
        Wt = torch.tensor(np.array(Ws, np.float32), device=dev)
        for task in tasks:
            u = task_input(task, dev)
            din = u.shape[-1]
            Win = np.zeros((len(grid) * BASE_SEEDS, din, n), np.float32)
            n_in = int(n * INPUT_FRACTION)
            for i, g in enumerate(Wins):
                si = grid[i // BASE_SEEDS][0]
                g2 = np.random.default_rng(int(g.integers(2**31)) + din)
                Win[i, :, :n_in] = g2.uniform(-1, 1, (din, n_in)) * si
            S = run_esn(Wt, torch.tensor(Win, device=dev), u, last=task.kind == "batch")
            sc = score_states(task, S).reshape(len(grid), BASE_SEEDS)
            best = int(sc.mean(1).argmax())
            out[task.name][str(k)] = {"mean": float(sc[best].mean()), "std": float(sc[best].std()),
                                      "input_scale": grid[best][0], "spectral_radius": grid[best][1],
                                      "units": n, "readout_params": n + 1}
            del S
    os.makedirs(cache_dir, exist_ok=True)
    import json
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return out


# --- testing particles -----------------------------------------------------------------------------

def _verdict(d, b, sd):
    s = max(sd, TOL)
    return "beat" if d > b + s else ("match" if d >= b - s else "below")


def test_particles(defs: list[dict], tasks: list[Task], base: dict, run_seed: int, dev) -> list[dict]:
    n = len(defs)
    scores = {v: {t.name: np.zeros((n, SEEDS)) for t in tasks} for v in ("matter", "nobond", "shuffled")}
    assembly = [[None] * SEEDS for _ in range(n)]
    chosen_scale = {t.name: [None] * n for t in tasks}
    jobs = [(i, s) for i in range(n) for s in range(SEEDS)]
    for c0 in range(0, len(jobs), CHUNK):
        chunk = jobs[c0:c0 + CHUNK]
        w = Worlds([defs[i] for i, _ in chunk], [world_seed(run_seed, defs[i], s) for i, s in chunk], dev)
        w.assemble()
        for (i, s), st in zip(chunk, w.stats):
            assembly[i][s] = st
        Cs = {v: w.coupling(v) for v in ("matter", "nobond", "shuffled")}
        run = lambda C, Win, u, task: score_states(task, run_matter(C, Win, w.tau, w.bias, w.gain, w.alpha, w.T, u,
                                                                    last=task.kind == "batch"))
        for task in tasks:
            u = task_input(task, dev)
            Win = w.input_weights(u.shape[-1])
            # the input scale, from the same grid the ESN gets, chosen per particle (mean of its seeds)
            grid = np.stack([run(Cs["matter"], Win * g, u, task) for g in INPUT_SCALES], 1)  # [M, G]
            pick = np.zeros(len(chunk), dtype=np.int64)
            for i in {i for i, _ in chunk}:
                rows = [r for r, (j, _) in enumerate(chunk) if j == i]
                pick[rows] = int(grid[rows].mean(0).argmax())
                chosen_scale[task.name][i] = INPUT_SCALES[pick[rows[0]]]
            sc_m = grid[np.arange(len(chunk)), pick]
            Wsel = Win * torch.tensor([INPUT_SCALES[p] for p in pick], device=dev, dtype=torch.float32)[:, None, None]
            per = {"matter": sc_m, "nobond": run(Cs["nobond"], Wsel, u, task), "shuffled": run(Cs["shuffled"], Wsel, u, task)}
            for v, sc in per.items():
                for (i, s), val in zip(chunk, sc):
                    scores[v][task.name][i, s] = val
        del w, Cs
    results = []
    for i, d in enumerate(defs):
        k = str(d["k"])
        ts = {}
        for task in tasks:
            b = base[task.name][k]
            m = float(scores["matter"][task.name][i].mean())
            e = {"matter": round(m, 5), "seeds": [round(float(x), 5) for x in scores["matter"][task.name][i]],
                 "nobond": round(float(scores["nobond"][task.name][i].mean()), 5),
                 "shuffled": round(float(scores["shuffled"][task.name][i].mean()), 5),
                 "esn": {"mean": round(b["mean"], 5), "std": round(b["std"], 5), "input_scale": b["input_scale"],
                         "spectral_radius": b["spectral_radius"]},
                 "units": P * d["k"], "readout_params": b["readout_params"],
                 "input_scale": chosen_scale[task.name][i], "chance": task.chance}
            e["verdict"] = _verdict(m, b["mean"], b["std"])
            e["above_chance"] = m > task.chance + 5 * TOL
            e["assembly_matters"] = m > e["nobond"] + TOL and m > e["shuffled"] + TOL
            ts[task.name] = e
        good = [t for t, e in ts.items() if e["verdict"] in ("beat", "match") and e["above_chance"]]
        z = [(e["matter"] - e["esn"]["mean"]) / max(e["esn"]["std"], TOL) for e in ts.values()]
        results.append({"tasks": ts, "keep": bool(good), "good_at": good,
                        "assembly_matters": [t for t in good if ts[t]["assembly_matters"]],
                        "score": round(float(np.mean(np.clip(z, -10, 10))), 4),
                        "assembly": assembly[i]})
    return results
