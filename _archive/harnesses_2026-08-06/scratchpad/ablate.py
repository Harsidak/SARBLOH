"""Component ablation sweep.

One tier-1 run per kill switch, against the untouched tier-1 baseline. The
question each run answers is not "is this component clever" but "does turning it
off change the score". Three outcomes:

    removal costs nothing   -> the component is dead weight; delete it or make it
                               opt-in (precedent: ARC_MCTS, 50.5% of think time
                               for 0 rewards)
    removal helps           -> the component is actively harmful
    removal costs levels    -> load-bearing; this is the real short list

Runs are serialised inside a stream because concurrent suites contend for CPU
and a contended run reads as a stall (measured: r11l 79.5s clean vs a 13-minute
"hang" under six concurrent suites). Streams are launched separately.

    python scratchpad/ablate.py <stream-name> <SWITCH> [SWITCH ...]
"""
import os
import sys
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "ablate")
GAMES = ["lp85", "ls20", "r11l", "sb26"]


def main():
    stream = sys.argv[1]
    switches = sys.argv[2:]
    os.makedirs(OUT, exist_ok=True)
    for sw in switches:
        tag = sw.replace("ARC_NO_", "no_").lower()
        log = os.path.join(OUT, tag + ".log")
        js = os.path.join(OUT, tag + ".json")
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONHASHSEED"] = "0"
        env[sw] = "1"
        t0 = time.time()
        print(f"[{stream}] START {sw}", flush=True)
        with open(log, "w", encoding="utf-8") as fh:
            subprocess.run(
                [sys.executable, "-u", "eval/rhae_eval.py", *GAMES,
                 "--seed", "0", "--timeout", "2000", "--out", js],
                cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
            )
        print(f"[{stream}] DONE  {sw}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
