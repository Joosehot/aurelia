"""The vault: kept particle species as JSON files (record + definition + scores + report)."""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from .diamond import anatomy, kind
from .matter import TOL

USES = {
    "memcap": "a fading memory of a signal (short-term memory for a sequence model)",
    "xor": "combining recent events nonlinearly (temporal feature detection)",
    "narma10": "modelling a nonlinear system with a long memory",
    "latch": "holding a value until it is overwritten, with nothing trained but the readout: a "
             "physical memory element for a recurrent or reservoir system",
    "mnist14": "turning a scanned pattern into readable features",
}


def report(entry: dict) -> str:
    d = entry["definition"]
    a = anatomy(d)
    sc = entry["scores"]
    asm = [s for s in sc["assembly"] if s]
    q = np.array(d["charge"])
    aff = -(q @ q.T)
    attract = int((aff[np.triu_indices(len(q), 1)] > 0.3).sum()) if len(q) > 1 else 0
    lines = [f"Particle {entry['id']}: {entry['kind']}", ""]
    lines.append(
        f"What it is: a particle of mass {d['mass']:.1f} and radius {d['radius']:.2f} with {a['k']} ports on its rim. "
        f"Each port is a stick-slip contact with a dead zone (mean {a['tau_mean']:.2f}) and a 4-dim charge; "
        f"{attract} of its port pairs attract each other strongly (so copies can bind port to port). Inside, "
        f"{a['edges']} couplings ({a['inhibitory']:.0%} inhibitory) pass a push from port to port; it settles in "
        f"{a['T']} sweeps with slip rate {a['alpha']:.2f} and keeps its state (memory).")
    if asm:
        lines.append(
            f"How it assembles (32 copies, {len(asm)} worlds): {np.mean([s['bonds_per_particle'] for s in asm]):.2f} "
            f"bonds per particle, largest molecule {np.mean([s['largest'] for s in asm]):.1f} particles, "
            f"{np.mean([s['free'] for s in asm]):.1f} left free.")
    lines.append("")
    lines.append("Its matter vs an ESN of the same size (readout only, both with tuned input scale); "
                 "controls: no bonds, random bonds:")
    for n, t in sc["tasks"].items():
        e = t["esn"]
        lines.append(f"  {n:8s} {t['matter']:.3f} vs ESN {e['mean']:.3f}±{e['std']:.3f}  [{t['verdict']}"
                     f"{'' if t['above_chance'] else ', at chance'}]  no bonds {t['nobond']:.3f}  random bonds "
                     f"{t['shuffled']:.3f}")
    lines.append("")
    good = sc["good_at"]
    if good:
        lines.append("Good at: " + ", ".join(good) + ".")
        am = sc["assembly_matters"]
        if am:
            lines.append("Its assembly matters on " + ", ".join(am) + ": the bonds its copies chose beat both no "
                         "bonds and random bonds.")
        else:
            lines.append("Its assembly does not matter: the particles do as well unbonded, so the useful part is "
                         "the single particle's own stick-slip memory.")
        lines.append("Use (a guess): " + "; ".join(USES[g] for g in good) + ".")
    else:
        lines.append("Good at: nothing.")
    return "\n".join(lines)


class Vault:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path(self, did: str) -> str:
        return os.path.join(self.root, f"{did}.json")

    def add(self, entry: dict) -> None:
        entry["kind"] = kind(entry["definition"])
        entry["report"] = report(entry)
        with open(self.path(entry["id"]), "w") as f:
            json.dump(entry, f, indent=1)

    def get(self, did: str) -> dict:
        p = self.path(did)
        if not os.path.exists(p):
            hits = [q for q in glob.glob(os.path.join(self.root, "*.json")) if os.path.basename(q).startswith(did)]
            if len(hits) != 1:
                raise SystemExit(f"no single particle matches {did!r} in {self.root}")
            p = hits[0]
        with open(p) as f:
            return json.load(f)

    def all(self) -> list[dict]:
        out = []
        for p in sorted(glob.glob(os.path.join(self.root, "*.json"))):
            with open(p) as f:
                out.append(json.load(f))
        return out
