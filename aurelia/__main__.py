"""Aurelia command line.

  python -m aurelia run --minutes 30 --stir spiral --seed 42
  python -m aurelia vault list | vault show <id> | vault replay <id>
  python -m aurelia bowl --stir all --seconds 20 --seed 1     (how many diamonds each stir makes)
  python -m aurelia viz --stir spiral --seed 42 --seconds 12  (docs/bowl.gif and docs/bowl.png)
  python -m aurelia viz --particle <id>                       (docs/assembly.gif: its copies binding)
"""

from __future__ import annotations

import argparse
import json
import sys

from .bowl import STIRS


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="aurelia")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="stir, form, test, keep, repeat")
    r.add_argument("--minutes", type=float, default=30)
    r.add_argument("--stir", choices=STIRS, default="spiral")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--vault", default="vault")
    r.add_argument("--no-control", action="store_true")
    v = sub.add_parser("vault", help="the kept diamonds")
    v.add_argument("action", choices=("list", "show", "replay"))
    v.add_argument("id", nargs="?")
    v.add_argument("--vault", default="vault")
    v.add_argument("--no-retest", action="store_true")
    b = sub.add_parser("bowl", help="stir a bowl and count the diamonds (no testing)")
    b.add_argument("--stir", choices=STIRS + ("all",), default="all")
    b.add_argument("--seconds", type=float, default=20)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--speed", type=float, default=1.0)
    z = sub.add_parser("viz", help="render the bowl being stirred")
    z.add_argument("--stir", choices=STIRS, default="spiral")
    z.add_argument("--seed", type=int, default=42)
    z.add_argument("--seconds", type=float, default=12)
    z.add_argument("--out", default="docs/bowl")
    z.add_argument("--particle", help="a vault id: render 32 copies of it assembling instead of the bowl")
    z.add_argument("--vault", default="vault")
    a = ap.parse_args(argv)

    if a.cmd == "run":
        from .loop import run
        run(a.minutes, a.stir, a.seed, a.batch, a.vault, control=not a.no_control)
    elif a.cmd == "vault":
        from .vault import Vault
        vault = Vault(a.vault)
        if a.action == "list":
            es = sorted(vault.all(), key=lambda e: -e["scores"]["score"])
            if not es:
                print("the vault is empty")
            print(f"{len(es)} kept particles (best first)")
            print(f"{'id':30s} {'score':>6s}  {'good at':22s} {'assembly matters':17s} kind")
            for e in es:
                sm = ",".join(e["scores"].get("assembly_matters", [])) or "-"
                print(f"{e['id']:30s} {e['scores']['score']:6.2f}  {','.join(e['scores']['good_at']):22s} {sm:17s} {e['kind']}")
        elif a.action == "show":
            if not a.id:
                sys.exit("vault show needs an id")
            e = vault.get(a.id)
            print(e["report"])
            print()
            print("definition:", json.dumps(e["definition"]))
            print("provenance:", json.dumps({k: v for k, v in e["provenance"].items() if k != "bowl"}))
        else:
            if not a.id:
                sys.exit("vault replay needs an id")
            from .loop import replay
            e = vault.get(a.id)
            out = replay(e, retest=not a.no_retest)
            print(f"replay {e['id']}")
            print(f"  definition rebuilt from the saved record: {'identical' if out['definition_from_record'] else 'DIFFERENT'}")
            print(f"  bowl episode re-simulated from seed {e['provenance']['episode_seed']} to step "
                  f"{e['provenance']['step']} ({out['resim_seconds']} s): record "
                  f"{'identical' if out['record_from_seed'] else 'DIFFERENT'}, definition "
                  f"{'identical' if out.get('definition_from_seed') else 'DIFFERENT'}")
            if "scores" in out:
                print("  re-tested (saved -> now):")
                for k, (s0, s1) in out["scores"].items():
                    print(f"    {k:9s} {s0:.4f} -> {s1:.4f}")
                print(f"  assembly (bonds, molecules) the same: {out['assembly_same']}")
                print(f"  kept again: {out['keep_again']}")
    elif a.cmd == "bowl":
        import numpy as np
        from .bowl import Bowl, BowlConfig
        from .diamond import definition_from_record, kind
        cfg = BowlConfig(stir_speed=a.speed)
        for s in (STIRS if a.stir == "all" else (a.stir,)):
            bw = Bowl(a.seed, s, cfg)
            heat = 0.0
            steps = int(a.seconds / cfg.dt)
            for _ in range(steps):
                bw.step()
                heat += float(bw.last_power.sum()) * 0.5 * cfg.dt
            ks = [len(r["nodes"]) for r in bw.formed]
            kinds = {}
            for r in bw.formed:
                kk = kind(definition_from_record(r, cfg.cluster_heat)).split(" (")[0]
                kinds[kk] = kinds.get(kk, 0) + 1
            top = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda x: -x[1])[:4])
            print(f"{s:8s} friction work {heat:9.0f}  diamonds {len(bw.formed):4d}  "
                  f"mean k {np.mean(ks) if ks else 0:4.1f}  {top}")
    elif a.cmd == "viz":
        if a.particle:
            from .vault import Vault
            from .viz import render_assembly
            e = Vault(a.vault).get(a.particle)
            render_assembly(e, a.out if a.out != "docs/bowl" else "docs/assembly")
        else:
            from .viz import render
            render(a.stir, a.seed, a.seconds, a.out)


if __name__ == "__main__":
    main()
