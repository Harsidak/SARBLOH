"""Replicate the ablation sweep's two live findings across seeds.

The sweep ran one seed. Two results moved the score:

    breaker off -> lp85 banks level 1 at action 66 instead of 95
    graph/cert off -> ls20 banks level 1 at all, which the baseline never does

The first is mechanism-explained (identical deaths, fewer detours); the second
could be seed luck -- one of the three banked it on action 1499 of 1500. This
script re-runs both against fresh seeds. A finding that only holds on seed 0 is
not a finding.

    python scratchpad/seedcheck.py <stream> <game> <seed> <CONFIG> [CONFIG ...]

CONFIG is a switch name, or "base" for the untouched agent.
"""
import os
import sys
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "seedcheck")


def main():
    stream, game, seed = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(OUT, exist_ok=True)
    for cfg in sys.argv[4:]:
        tag = f"{game}_s{seed}_{cfg.replace('ARC_NO_', 'no_').lower()}"
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONHASHSEED"] = "0"
        if cfg != "base":
            env[cfg] = "1"
        t0 = time.time()
        print(f"[{stream}] START {tag}", flush=True)
        with open(os.path.join(OUT, tag + ".log"), "w", encoding="utf-8") as fh:
            subprocess.run(
                [sys.executable, "-u", "eval/rhae_eval.py", game,
                 # A timeout no Modern Standby nap can reach. The sweep's
                 # 2000s cap truncated an ls20 run at 69 of 1500 actions when
                 # the laptop suspended for an hour mid-run, and a truncated
                 # run scores 0 for reasons that have nothing to do with the
                 # component under test. Scores are deterministic, so an
                 # oversized cap costs nothing but keeps a nap from voiding
                 # the result. Verify `actions` on every row regardless.
                 "--seed", seed, "--timeout", "30000",
                 "--out", os.path.join(OUT, tag + ".json")],
                cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
            )
        print(f"[{stream}] DONE  {tag}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
