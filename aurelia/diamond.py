"""From a friction record to a particle's inner structure: the *frictional knot*.

(particle.py adds the rest of the particle: charged ports, mass, radius. What follows describes
the knot inside it. The paragraph about using it "like an activation" is how v0 first tested it; the
particle is now tested only as matter, in matter.py.)

Design
------
A diamond is born from k grains that rubbed against each other. Friction is stick-slip: a contact
holds (sticks) until the force on it exceeds a threshold, then slips, and where it stops depends on
where it was. The particle keeps exactly that behaviour as computation.

A knot has k nodes, one per grain, each with a *position* p_i (its state). Given a drive u (k
numbers, from whatever layer feeds the particle) it relaxes for T sweeps:

    d_i  = u_i + b_i + sum_j A_ij p_j                (drive plus the pull of the other grains)
    p_i <- p_i + alpha * (clamp(p_i, d_i - tau_i, d_i + tau_i) - p_i)

and emits y_i = g_i * p_i. The clamp is the *play operator* (backlash) of friction: a node does
not move while the drive is within tau_i of where it rests, and when pushed further it is dragged
along at distance tau_i. From rest, one node alone is a dead zone (soft shrinkage). Coupled through
A it becomes a small network of grains that drag each other, and because p is a state, a knot
that is not reset between time steps *remembers*: a pushed node stays where it was left until it is
pushed past its dead zone again (hysteresis). A knot has no trainable parameters at all; a model
uses it between trainable linear maps, like an activation, but it is a fixed little dynamical
system with memory, invented by the bowl.

Where each part comes from (all from the record, deterministically):

- nodes and their order: the grains, ordered by when their contacts began ("order of contacts");
- A_ij: friction work between grains i and j, normalised, signed by whether their hidden values
  agree (sign of h_i . h_j). Node i feels a node that touched earlier (j before i) fully and a later
  one only by beta, so the contact order makes the coupling directed;
- beta: the cluster's swirl (how coherently it was rotating when it formed): a swirling cluster
  couples both ways, a jolted one cascades one way;
- the size of A (largest singular value) rho: from the cluster's total heat, in [0.3, 0.95], so the
  linear part always contracts;
- tau_i: from the grain's hidden value and how hard it was pressed (peak contact force);
- b_i, g_i: from the grain's hidden value;
- alpha (how far a node slips per sweep): from how fast the cluster was moving;
- T (sweeps): from how unevenly the forces were spread (force CV), 2..6.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

K_MAX = 12


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def definition_from_record(rec: dict, cluster_heat: float) -> dict:
    H = np.array(rec["hidden"], dtype=np.float64)
    W = np.array(rec["pair_work"], dtype=np.float64)
    k = H.shape[0]
    np.fill_diagonal(W, 0.0)
    Wn = W / (W.max() + 1e-12)
    S = np.sign(H @ H.T)
    S[S == 0] = 1.0
    st = rec["stir"]
    beta = float(np.clip(st["swirl"], 0.05, 1.0))
    order = np.arange(k)  # nodes are stored in contact order already
    earlier = order[None, :] < order[:, None]  # j touched before i
    A = S * Wn * np.where(earlier, 1.0, beta)
    np.fill_diagonal(A, 0.0)
    total_heat = float(np.sum(rec["heat"]))
    rho = 0.3 + 0.65 * math.tanh(total_heat / (3.0 * cluster_heat))
    smax = float(np.linalg.svd(A, compute_uv=False)[0]) if np.any(A) else 0.0
    if smax > 0:
        A = A * (rho / smax)
    peak = np.array(rec["peak_force"], dtype=np.float64)
    rel = peak / (peak.mean() + 1e-12)
    tau = np.clip([0.08 + 0.6 * _sig(H[i, 0]) * rel[i] for i in range(k)], 0.05, 1.5)
    bias = 0.25 * np.tanh(H[:, 1])
    gain = 1.0 + 0.5 * np.tanh(H[:, 2])
    alpha = float(np.clip(0.25 + 0.75 * (1 - math.exp(-st["mean_speed"] / 2.0)), 0.25, 1.0))
    T = int(2 + round(4 * min(1.0, max(0.0, st["force_cv"]))))
    return {
        "form": "frictional-knot/v0",
        "k": k,
        "A": [[float(a) for a in row] for row in A],
        "tau": [float(t) for t in tau],
        "bias": [float(b) for b in bias],
        "gain": [float(g) for g in gain],
        "alpha": alpha,
        "T": T,
        "beta": beta,
        "rho": float(rho),
    }


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def knot_numpy(defn: dict, u: np.ndarray, p: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation (one input vector, float64): returns (output, state)."""
    A = np.array(defn["A"])
    tau, b, g = np.array(defn["tau"]), np.array(defn["bias"]), np.array(defn["gain"])
    p = np.zeros(defn["k"]) if p is None else p.copy()
    for _ in range(defn["T"]):
        d = u + b + A @ p
        p = p + defn["alpha"] * (np.clip(p, d - tau, d + tau) - p)
    return g * p, p


def anatomy(defn: dict) -> dict:
    """Plain mechanical facts about a knot, for kinds and reports."""
    A = np.array(defn["A"])
    k = defn["k"]
    off = ~np.eye(k, dtype=bool)
    nz = np.abs(A[off]) > 1e-3 * (np.abs(A).max() + 1e-12)
    neg = (A[off] < 0) & nz
    tau = np.array(defn["tau"])
    row = np.abs(A).sum(1)
    # loops: a cycle exists in the strong-edge graph if the strong part is not nilpotent
    strong = (np.abs(A) > 0.15 * (np.abs(A).max() + 1e-12)).astype(float)
    cyc = bool(np.abs(np.linalg.eigvals(strong)).max() > 1e-6) if k else False
    return {
        "k": k,
        "edges": int(nz.sum()),
        "density": float(nz.mean()) if nz.size else 0.0,
        "inhibitory": float(neg.sum() / max(1, nz.sum())),
        "coupling": float(row.mean()),
        "tau_mean": float(tau.mean()),
        "beta": defn["beta"],
        "T": defn["T"],
        "alpha": defn["alpha"],
        "loops": cyc,
    }


def kind(defn: dict) -> str:
    a = anatomy(defn)
    parts = []
    if a["tau_mean"] > 0.5:
        parts.append("sticky")
    elif a["tau_mean"] < 0.2:
        parts.append("slippery")
    if a["coupling"] < 0.15:
        parts.append("loose")
    elif a["inhibitory"] > 0.6:
        parts.append("inhibitory")
    elif a["inhibitory"] < 0.25:
        parts.append("excitatory")
    shape = "cascade" if a["beta"] < 0.3 else ("knot" if a["loops"] else "chain")
    parts.append(shape)
    return " ".join(parts) + f" (k={a['k']})"
