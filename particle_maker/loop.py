"""The loop: stir -> diamonds (particle species) form -> their matter is tested -> keep -> repeat."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from .bowl import Bowl, BowlConfig
from .diamond import digest, kind
from .matter import esn_baselines, load_tasks, test_particles
from .particle import particle_from_record
from .vault import Vault

EPISODE_STEPS = 3000


def episode_seed(run_seed: int, episode: int) -> int:
    return (run_seed * 100003 + episode) % (2**31)


class DiamondSource:
    """Runs bowl episodes (a fresh bowl every EPISODE_STEPS steps, seeded from the run seed) and hands
    out diamonds in the order they form, with their provenance."""

    def __init__(self, run_seed: int, stir: str, cfg: BowlConfig, dev):
        self.run_seed, self.stir, self.cfg, self.dev = run_seed, stir, cfg, dev
        self.episode = -1
        self.bowl = None
        self.taken = 0
        self.sim_steps = 0

    def _new_episode(self):
        self.episode += 1
        self.bowl = Bowl(episode_seed(self.run_seed, self.episode), self.stir, self.cfg, self.dev)
        self.taken = 0

    def take(self, n: int) -> list[dict]:
        out = []
        while len(out) < n:
            if self.bowl is None or self.bowl.step_i >= EPISODE_STEPS:
                self._new_episode()
            b = self.bowl
            while len(b.formed) <= self.taken and b.step_i < EPISODE_STEPS:
                b.step()
                self.sim_steps += 1
            while self.taken < len(b.formed) and len(out) < n:
                rec = b.formed[self.taken]
                out.append({
                    "id": f"A{self.run_seed}-{self.stir}-e{self.episode:03d}-d{self.taken:03d}",
                    "record": rec,
                    "provenance": {"run_seed": self.run_seed, "stir": self.stir, "episode": self.episode,
                                   "episode_seed": b.seed, "index": self.taken, "step": rec["step"],
                                   "bowl": self.cfg.to_dict()},
                })
                self.taken += 1
        return out


def shuffled_twin(rec: dict, rng: np.random.Generator) -> dict:
    """The control species: the same grains' values and stir, but the friction structure scrambled
    (pair work moved to random pairs, the grains' order shuffled)."""
    k = len(rec["hidden"])
    perm = rng.permutation(k)
    W = np.array(rec["pair_work"])
    iu = np.triu_indices(k, 1)
    vals = W[iu].copy()
    rng.shuffle(vals)
    W2 = np.zeros((k, k))
    W2[iu] = vals
    twin = dict(rec)
    twin["hidden"] = [rec["hidden"][i] for i in perm]
    twin["heat"] = [rec["heat"][i] for i in perm]
    twin["peak_force"] = [rec["peak_force"][i] for i in perm]
    twin["pair_work"] = (W2 + W2.T).tolist()
    return twin


def _entry(item: dict, defn: dict, res: dict) -> dict:
    return {"id": item["id"], "provenance": item["provenance"], "record": item["record"],
            "record_hash": digest(item["record"]), "definition": defn, "definition_hash": digest(defn),
            "scores": res, "created": time.strftime("%Y-%m-%d %H:%M:%S")}


def run(minutes: float, stir: str, seed: int, batch: int = 32, vault_dir: str = "vault",
        runs_dir: str = "runs", control: bool = True) -> dict:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = BowlConfig()
    run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{stir}-s{seed}"
    rdir = os.path.join(runs_dir, run_id)
    os.makedirs(rdir, exist_ok=True)
    log = open(os.path.join(rdir, "log.jsonl"), "w")
    disc = open(os.path.join(rdir, "discarded.jsonl"), "w")
    vault = Vault(vault_dir)
    print(f"Particle maker run {run_id}: {minutes} min, stir {stir}, seed {seed}, device {dev}")
    tasks = load_tasks()
    t_b = time.time()
    base = esn_baselines(tasks, dev, os.path.join(runs_dir, "cache"))
    print(f"ESN baselines ready ({time.time() - t_b:.0f} s; cached after the first run)")
    t0 = time.time()
    deadline = t0 + minutes * 60
    src = DiamondSource(seed, stir, cfg, dev)
    names = [t.name for t in tasks]
    stats = {"tested": 0, "kept": 0, "discarded": 0, "kinds": {}, "kept_kinds": {}, "good_at": {},
             "assembly_matters": {}, "assembly_any": 0, "best": {n: None for n in names}, "batches": 0}
    control_stats = None
    last_batch = 0.0
    while stats["batches"] == 0 or time.time() + last_batch <= deadline:
        tb = time.time()
        items = src.take(batch)
        defs = [particle_from_record(it["record"], cfg.cluster_heat) for it in items]
        results = test_particles(defs, tasks, base, seed, dev)
        if control and control_stats is None:
            rng = np.random.default_rng(seed)
            tdefs = [particle_from_record(shuffled_twin(it["record"], rng), cfg.cluster_heat) for it in items]
            tres = test_particles(tdefs, tasks, base, seed, dev)
            control_stats = {"n": len(tres), "twin_kept": sum(r["keep"] for r in tres),
                             "real_kept": sum(r["keep"] for r in results),
                             "per_task": {n: {"twin": float(np.mean([r["tasks"][n]["matter"] for r in tres])),
                                              "real": float(np.mean([r["tasks"][n]["matter"] for r in results]))}
                                          for n in names}}
            print("control: the first batch's species with their friction structure scrambled")
            for n, v in control_stats["per_task"].items():
                print(f"  {n:8s} real {v['real']:.3f}  scrambled {v['twin']:.3f}")
        for it, d, r in zip(items, defs, results):
            short = kind(d).split(" (")[0]
            stats["tested"] += 1
            stats["kinds"][short] = stats["kinds"].get(short, 0) + 1
            for n, t in r["tasks"].items():
                b = stats["best"][n]
                if b is None or t["matter"] > b["matter"]:
                    stats["best"][n] = {"id": it["id"], "matter": t["matter"], "esn": t["esn"]["mean"]}
            if r["keep"]:
                stats["kept"] += 1
                stats["kept_kinds"][short] = stats["kept_kinds"].get(short, 0) + 1
                for g in r["good_at"]:
                    stats["good_at"][g] = stats["good_at"].get(g, 0) + 1
                for g in r["assembly_matters"]:
                    stats["assembly_matters"][g] = stats["assembly_matters"].get(g, 0) + 1
                stats["assembly_any"] += bool(r["assembly_matters"])
                vault.add(_entry(it, d, r))
            else:
                stats["discarded"] += 1
                disc.write(json.dumps({"id": it["id"], "kind": kind(d), "score": r["score"],
                                       "tasks": {n: [t["matter"], t["verdict"]] for n, t in r["tasks"].items()}}) + "\n")
        stats["batches"] += 1
        last_batch = time.time() - tb
        el = (time.time() - t0) / 60
        line = {"minutes": round(el, 3), "tested": stats["tested"], "kept": stats["kept"],
                "discarded": stats["discarded"], "assembly_matters": stats["assembly_any"],
                "diamonds_per_min": round(stats["tested"] / el, 1), "batch_seconds": round(last_batch, 1),
                "sim_steps": src.sim_steps, "episodes": src.episode + 1,
                "best": {n: (v["matter"] if v else None) for n, v in stats["best"].items()}}
        log.write(json.dumps(line) + "\n")
        log.flush()
        disc.flush()
        print(f"  {el:5.1f} min  tested {stats['tested']:4d}  kept {stats['kept']:4d}  assembly matters "
              f"{stats['assembly_any']:3d}  {line['diamonds_per_min']:.0f}/min  best " +
              " ".join(f"{n} {v:.3f}" for n, v in line["best"].items()), flush=True)
    summary = {"run_id": run_id, "minutes": round((time.time() - t0) / 60, 2), "stir": stir, "seed": seed,
               "bowl": cfg.to_dict(), "episodes": src.episode + 1, "sim_steps": src.sim_steps,
               "esn": base, "control": control_stats, **stats}
    with open(os.path.join(rdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log.close()
    disc.close()
    print(f"done: tested {stats['tested']}, kept {stats['kept']}, discarded {stats['discarded']}; summary in {rdir}")
    return summary


def replay(entry: dict, retest: bool = True) -> dict:
    """Rebuild a particle from its record, re-simulate its bowl episode from the seed, and re-test it."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pv = entry["provenance"]
    cfg = BowlConfig(**pv["bowl"])
    out = {}
    defn = particle_from_record(entry["record"], cfg.cluster_heat)
    out["definition_from_record"] = digest(defn) == entry["definition_hash"]
    t0 = time.time()
    b = Bowl(pv["episode_seed"], pv["stir"], cfg, dev)
    while len(b.formed) <= pv["index"] and b.step_i < EPISODE_STEPS:
        b.step()
    rec = b.formed[pv["index"]] if len(b.formed) > pv["index"] else None
    out["record_from_seed"] = rec is not None and digest(rec) == entry["record_hash"]
    out["resim_seconds"] = round(time.time() - t0, 1)
    if rec is not None:
        out["definition_from_seed"] = digest(particle_from_record(rec, cfg.cluster_heat)) == entry["definition_hash"]
    if retest:
        tasks = load_tasks()
        base = esn_baselines(tasks, dev, os.path.join("runs", "cache"))
        r = test_particles([defn], tasks, base, pv["run_seed"], dev)[0]
        out["scores"] = {n: (entry["scores"]["tasks"][n]["matter"], t["matter"]) for n, t in r["tasks"].items()}
        out["assembly_same"] = r["assembly"] == entry["scores"]["assembly"]
        out["keep_again"] = r["keep"]
    return out
