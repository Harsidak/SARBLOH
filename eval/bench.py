"""THE feedback loop. One file, one protocol, one verdict.

This replaces eval/local_eval.py, eval/rhae_eval.py, scratchpad/ablate*.py,
scratchpad/seedcheck.py and scratchpad/route_ab.py. There is now exactly one way
to ask "did that change help?", and it answers with the official number or it
says it does not know.

Scoring is NOT implemented here. It lives in eval/official_score.py, which calls
the calculator the competition ships. Nothing in this repo may transcribe RHAE
again -- every harness that did was wrong by ~400x.

Why this file is shaped the way it is
-------------------------------------
Measured on lp85, the UNTOUCHED agent scores between 0.0679 and 0.4972 honest
across seeds 0-9. The whole ten-switch ablation sweep at seed 0 spanned 0.3592.
The noise is larger than every effect we have ever "measured". So:

  * seeds are plural by default and a single seed is refused for a verdict;
  * A/B is PAIRED on (game, seed) and run in one invocation (--ablate), so the
    two arms cannot drift apart on anything but the switch;
  * the verdict is allowed to be NO EVIDENCE, and says how many seeds it would
    take to resolve the effect it actually saw. That is the point. A loop that
    always produces a direction is a loop that produces noise.

Validity, so a number is never quietly wrong
--------------------------------------------
  STALL/TIMEOUT  the run was cut short -- the score is INVALID and is dropped
                 from every aggregate (a truncated run once got filed as
                 "score-neutral"; it was 69 of 1500 actions).
  SUSPEND        wall clock advanced while the CPU did not: the laptop slept
                 mid-run (Modern Standby once turned a 390s game into 12997s).
                 The SCORE survives this; the TIMING does not, so timing is
                 dropped and the row is flagged.
  EARLY          the agent stopped before the cap without winning.

Usage
-----
    # baseline, dev split, 3 seeds, parallel
    python eval/bench.py --label base --seeds 0-2

    # did my change help? (compares against the stored baseline run)
    python eval/bench.py --label mychange --seeds 0-2 --vs base

    # is component X earning its keep? (both arms in ONE paired invocation)
    python eval/bench.py --label graph --seeds 0-4 --ablate ARC_NO_GRAPH=1

    # generalization report -- read only, never tune against this
    python eval/bench.py --label mychange --seeds 0-2 --split heldout

Seeding: MyAgent seeds its RNG from ARC_AGENT_SEED + hash(game_id), so
PYTHONHASHSEED is pinned to 0 inside the workers or runs are not reproducible.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# ONE THREAD PER WORKER. Set before numpy/torch are imported anywhere, in the
# parent, so every spawned worker inherits it.
#
# Measured 2026-08-06: with 8 workers each letting BLAS open a thread per core,
# 12 cores were carrying ~96 compute threads. Games that finish in 212s alone
# took 900s+ under that contention -- past the timeout -- so 8 of the first 9
# cells were DROPPED as invalid. The harness was measuring its own oversubscription
# and then throwing the result away. The agent's inner loop is small numpy on
# small grids; it gets nothing from multithreaded BLAS and pays the sync cost on
# every call.
# ---------------------------------------------------------------------------
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "TORCH_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import glob
import json
import math
import time
import random
import logging
import argparse
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REAL_GAMES_DIR = os.path.join(HERE, "real_games")
SCORES_DIR = os.path.join(ROOT, "scores")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

# 80% power, alpha=0.05 two-sided: (z_{a/2} + z_beta) = 1.96 + 0.84
_POWER_Z = 2.80
# A single action cannot legitimately take this long on the dev box (no LLM).
_SUSPEND_WALL_S = 30.0
# ...and if the CPU barely moved during it, the machine was asleep, not working.
_SUSPEND_CPU_FRAC = 0.25


# ---------------------------------------------------------------- discovery
def games_index() -> dict:
    with open(os.path.join(REAL_GAMES_DIR, "games_index.json"), encoding="utf-8") as f:
        return {g["id"]: g for g in json.load(f)}


def discover_games() -> dict:
    """id -> path of the runnable game module."""
    out = {}
    for gdir in sorted(glob.glob(os.path.join(REAL_GAMES_DIR, "*"))):
        if not os.path.isdir(gdir):
            continue
        gid = os.path.basename(gdir)
        hits = glob.glob(os.path.join(gdir, "*", f"{gid}.py"))
        if hits:
            out[gid] = sorted(hits)[0]
    return out


def load_split() -> dict:
    with open(os.path.join(HERE, "splits", "heldout_split.json"), encoding="utf-8") as f:
        d = json.load(f)
    return {"dev": list(d.get("dev") or []), "heldout": list(d.get("heldout") or [])}


# ---------------------------------------------------------------- the run
def _quiet_logger():
    lg = logging.getLogger("bench")
    lg.setLevel(logging.ERROR)
    if not lg.handlers:
        lg.addHandler(logging.StreamHandler(sys.stderr))
    return lg


def _class_name_from_file(path):
    from scoreboard import class_name_from_file
    return class_name_from_file(path)


_AGENT_CACHE = {}


def _agent_class(agent_file: str):
    """MyAgent from `my_agent.py`, or an alternate build by path.

    Cached: re-executing a 7k-line module per game resets its module-level state
    and costs seconds.
    """
    if not agent_file:
        from my_agent import MyAgent
        return MyAgent
    if agent_file not in _AGENT_CACHE:
        from agent_loader import load_agent_module
        _AGENT_CACHE[agent_file] = load_agent_module(agent_file).MyAgent
    return _AGENT_CACHE[agent_file]


def run_game(game_file, seed=0, max_actions=1500, timeout=None, agent_file="",
             meta=None, competition_mode=False):
    """Play one game. Returns the raw ledger; scoring happens in score_row().

    The official number is READ BACK from the competition's own scorecard chain
    (eval/scoreboard.py), not reconstructed here. One card per cell: cells are
    separate processes, so a card cannot span them, and a card holding one
    environment scores that environment identically either way.
    """
    import numpy as np
    from arc_agi.local_wrapper import LocalEnvironmentWrapper
    from official_score import LevelEvents
    from scoreboard import OfficialScorecard, make_env_info, reset_would_wipe

    MyAgent = _agent_class(agent_file)
    game_id = os.path.splitext(os.path.basename(game_file))[0]
    info, base_src = make_env_info(game_file, game_id, meta or {})
    board = OfficialScorecard(competition_mode=competition_mode)
    board.register(info)
    wrapper = LocalEnvironmentWrapper(info, _quiet_logger(), seed=seed,
                                      **board.wrapper_kwargs())
    frame0 = wrapper.observation_space
    if frame0 is None:
        return {"game": game_id, "seed": seed, "error": "failed to load/reset game"}

    win_levels = int(getattr(frame0, "win_levels", 0) or 0)
    os.environ["ARC_AGENT_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    agent = MyAgent(game_id=game_id)
    frames = [frame0]

    ev = LevelEvents()
    ev.observe(getattr(frame0, "levels_completed", 0) or 0, 0)

    # Intent ledger: who spends, who scores, who loses. A score with no
    # attribution cannot steer anything.
    spend, banked, wiped, deaths = Counter(), Counter(), Counter(), Counter()
    changed, novel = Counter(), Counter()      # working vs fidgeting
    route_ms = defaultdict(float)
    seen_frames: set = set()
    resets = game_overs = swallowed_resets = 0

    def _fhash(fr):
        g = getattr(fr, "frame", None)
        try:
            return hash(np.asarray(g, dtype=np.int16).tobytes())
        except Exception:
            return hash(repr(g))

    t0, c0 = time.time(), time.process_time()
    suspend_s, suspend_events = 0.0, 0
    stall = None
    while not agent.is_done(frames, frames[-1]) and agent.action_counter <= max_actions:
        if timeout and (time.time() - t0 - suspend_s) > timeout:
            stall = f"timeout>{timeout}s"
            break
        w_tick, c_tick = time.perf_counter(), time.process_time()
        try:
            action = agent.choose_action(frames, frames[-1])
        except Exception as e:
            stall = f"choose_action raised: {type(e).__name__}: {e}"
            break
        wall = time.perf_counter() - w_tick
        cpu = time.process_time() - c_tick
        # Wall moved, CPU did not => the machine slept. Timing is unusable for
        # this step; the score is not affected, so keep playing and flag it.
        if wall > _SUSPEND_WALL_S and cpu < _SUSPEND_CPU_FRAC * wall:
            suspend_s += wall - cpu
            suspend_events += 1

        route = str(getattr(agent, "_route", "") or "?")
        spend[route] += 1
        route_ms[route] += wall * 1000.0

        data = {}
        if hasattr(action, "is_complex") and action.is_complex():
            data = {"x": int(action.action_data.x), "y": int(action.action_data.y)}
        prev_h = _fhash(frames[-1])
        is_reset = getattr(action, "name", str(action)) == "RESET"
        if competition_mode and is_reset and reset_would_wipe(wrapper):
            # COMPETITION MODE, from arc_agi/api.py:316-334. The RESET that would
            # full_reset() and wipe the banked score is NOT PERFORMED by the hosted
            # API: it returns the current frame and books the update anyway, so the
            # action is spent, the world does not change, and the score survives.
            # Without this branch a local run measures a game the leaderboard will
            # never play -- there, the wipe is impossible.
            resp = wrapper.observation_space
            sc = board.raw()
            if sc is not None and resp is not None:
                try:
                    sc.update_scorecard(wrapper._guid, resp, False)
                except Exception:
                    pass
            swallowed_resets += 1
        else:
            resp = wrapper.step(action, data=data)
        if resp is None:
            stall = "wrapper.step returned None"
            break
        cur_h = _fhash(resp)
        if cur_h != prev_h:
            changed[route] += 1
        if cur_h not in seen_frames:
            novel[route] += 1
            seen_frames.add(cur_h)
        frames.append(resp)
        agent.action_counter += 1

        if str(getattr(agent, "last_action_str", "") or "").startswith("RESET"):
            resets += 1
        if getattr(getattr(resp, "state", None), "name", "") == "GAME_OVER":
            game_overs += 1
            deaths[route] += 1

        before_peak, before_wipes = ev.real_levels, ev.wipes
        if ev.observe(getattr(resp, "levels_completed", 0) or 0, agent.action_counter):
            if ev.real_levels > before_peak:
                banked[route] += ev.real_levels - before_peak
            if ev.wipes > before_wipes:
                wiped[route] += ev.wipes - before_wipes

    last = frames[-1]
    state = getattr(getattr(last, "state", None), "name", str(getattr(last, "state", None)))
    secs = time.time() - t0

    # THE official number, read back from their objects. Never recomputed here.
    card = board.game(game_id)
    return {
        "game": game_id, "seed": seed, "win_levels": win_levels,
        "scorecard": card, "baseline_source": base_src,
        "baseline_official": list(getattr(info, "baseline_actions", None) or []),
        "competition_mode": bool(competition_mode),
        "swallowed_resets": swallowed_resets,
        "actions": agent.action_counter, "state": state, "won": state == "WIN",
        "secs": round(secs, 1), "secs_active": round(secs - suspend_s, 1),
        "cpu_s": round(time.process_time() - c0, 1),
        "suspend_s": round(suspend_s, 1), "suspend_events": suspend_events,
        "stall": stall, "resets": resets, "game_overs": game_overs,
        "level_events": ev.to_json(),
        "spend": dict(spend), "banked": dict(banked), "wiped": dict(wiped),
        "deaths": dict(deaths), "changed": dict(changed), "novel": dict(novel),
        "route_ms": {k: round(v, 1) for k, v in route_ms.items()},
    }


def score_row(res: dict, meta: dict, max_actions: int) -> dict:
    """Attach the official numbers and the validity verdict to a raw run.

    Three scores, and they are NOT interchangeable:

      official     the scorecard's own number, read back from
                   EnvironmentScorecard (max over plays). This is the leaderboard.
      honest       capability, each level counted once -- computed against
                   games_index.json baselines, UNCHANGED, so every previously
                   stored scores/*.json stays comparable to a run made today.
      honest_v2    the same capability metric against the baselines the official
                   scorer actually divides by (metadata.json). Where the two
                   sources agree these are identical; where they disagree (cn04:
                   21 vs 15) honest_v2 is the right one and honest is the
                   comparable one. Re-base deliberately, never silently.

    `official_reconstructed` is eval/official_score.py's reconstruction of the
    leaderboard, computed against the SAME baselines the card used so the only
    thing that can differ is the reset-churn crediting bug. It is kept for one
    release as a cross-check and flagged when it disagrees.
    """
    from official_score import LevelEvents, score_run
    ev = LevelEvents()
    ev.events = [tuple(e) for e in res["level_events"]["events"]]
    ev.first_reach = {int(k): int(v) for k, v in res["level_events"]["first_reach"].items()}
    ev._peak = int(res["level_events"]["real_levels"])

    base_v1 = list(meta.get("baseline_steps") or [])
    base_v2 = list(res.get("baseline_official") or []) or base_v1
    sc = score_run(ev, res["actions"], base_v1, res.get("win_levels"))
    sc_v2 = score_run(ev, res["actions"], base_v2, res.get("win_levels"))

    card = res.get("scorecard") or None
    row = dict(res)
    row.update(sc)                                   # honest / real_levels / wipes
    row["honest_v1"] = sc["honest"]
    row["honest_v2"] = sc_v2["honest"]
    row["official_reconstructed"] = sc_v2["official"]

    flags = []
    if card is None:
        # No card means no leaderboard number. Report zero, but never let it read
        # as "the agent scored nothing".
        row["official"] = 0.0
        row["scorecard_ok"] = False
        flags.append("NO-SCORECARD")
    else:
        row["official"] = float(card.get("score") or 0.0)
        row["scorecard_ok"] = True
        row["plays"] = card.get("plays")
        row["scorer_message"] = card.get("message")
        msg = str(card.get("message") or "")
        if "baseline" in msg.lower() and "mismatch" in msg.lower():
            # The calculator returns 0.0 with this message when the level-event
            # count exceeds the baseline list. A silent zero here is
            # indistinguishable from agent failure, so it is a LOUD flag.
            flags.append("BASELINE-MISMATCH")
        elif msg:
            flags.append("SCORER-MSG")
    hon = row["honest"]
    row["inflation"] = (row["official"] / hon) if hon > 0 else (
        float("inf") if row["official"] > 0 else 0.0)

    if base_v1 and base_v2 and base_v1 != base_v2:
        flags.append("BASELINE-DISAGREE")
    if abs(row["official_reconstructed"] - row["official"]) > 1e-6:
        flags.append("RECON-MISMATCH")
    if res.get("stall"):
        flags.append("TIMEOUT" if "timeout" in str(res["stall"]) else "STALL")
    if res.get("suspend_events"):
        flags.append(f"SUSPEND x{res['suspend_events']}")
    if not res.get("stall") and res["actions"] < max_actions and not res.get("won"):
        flags.append("EARLY")

    row["flags"] = flags
    # A cut-short run has not played the game the score claims it played.
    row["score_valid"] = not bool(res.get("stall"))
    row["timing_valid"] = not res.get("suspend_events")
    return row


# ---------------------------------------------------------------- workers
def _task(payload):
    """One (arm, game, seed) cell. Module-level so Windows spawn can pickle it."""
    (arm, gid, gfile, seed, cap, timeout, agent_file, env, managed,
     meta, comp) = payload
    os.environ["PYTHONHASHSEED"] = "0"
    for k in managed:                     # workers are reused across arms
        os.environ.pop(k, None)
    for k, v in (env or {}).items():
        os.environ[k] = str(v)
    try:
        res = run_game(gfile, seed=seed, max_actions=cap, timeout=timeout,
                       agent_file=agent_file, meta=meta, competition_mode=comp)
    except Exception as e:                                    # never lose a cell
        import traceback
        res = {"game": gid, "seed": seed, "error": f"{type(e).__name__}: {e}",
               "traceback": traceback.format_exc()[-1200:]}
    res["arm"] = arm
    return res


def run_matrix(arms, gids, files, seeds, cap, timeout, agent_file, jobs,
               idx=None, competition_mode=False):
    """arms: {name: env_dict}. Returns raw rows. Ordered by cost, longest first."""
    managed = sorted({k for env in arms.values() for k in env})
    idx = idx or {}
    payloads = [(arm, g, files[g], s, cap, timeout, agent_file, env, managed,
                 idx.get(g, {}), competition_mode)
                for arm, env in arms.items() for g in gids for s in seeds]
    total = len(payloads)
    rows, done, t0 = [], 0, time.time()
    if jobs <= 1:
        for p in payloads:
            rows.append(_task(p))
            done += 1
            _progress(rows[-1], done, total, t0)
        return rows
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(_task, p): p for p in payloads}
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            _progress(rows[-1], done, total, t0)
    return rows


def _progress(res, done, total, t0):
    tag = f"{res.get('arm', '?')}/{res.get('game', '?')}/s{res.get('seed', '?')}"
    if res.get("error"):
        print(f"[{done:>3}/{total}] {tag:<28} ERROR {res['error']}", flush=True)
        return
    lv = res["level_events"]["real_levels"]
    print(f"[{done:>3}/{total}] {tag:<28} {lv}L {res['actions']:>4}a "
          f"{res['secs']:>6.0f}s  ({time.time() - t0:.0f}s elapsed)", flush=True)


# ---------------------------------------------------------------- reporting
def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def report_arm(rows, name, cap):
    """Per-game means over seeds + the overall number + the noise floor."""
    ok = [r for r in rows if r.get("score_valid")]
    bad = [r for r in rows if not r.get("score_valid")]
    if not ok:
        print(f"\n  {name}: NO VALID ROWS ({len(bad)} invalid) -- nothing to report.")
        return None

    by_game = defaultdict(list)
    for r in ok:
        by_game[r["game"]].append(r)

    print(f"\n  {'game':<7}{'lv':>6}{'official':>10}{'recon':>8}{'honest':>9}"
          f"{'hon_v2':>8}{'sd':>7}{'spread':>8}{'wipe':>6}{'acts':>7}  flags")
    spreads = []
    for g in sorted(by_game):
        rs = sorted(by_game[g], key=lambda r: r["seed"])
        hon = [r["honest"] for r in rs]
        sd = statistics.stdev(hon) if len(hon) > 1 else 0.0
        spread = max(hon) - min(hon)
        spreads.append((spread, g))
        fl = sorted({f for r in rs for f in r["flags"]})
        print(f"  {g:<7}{_mean([r['real_levels'] for r in rs]):>4.1f}/"
              f"{rs[0]['win_levels']:<2}{_mean([r['official'] for r in rs]):>9.3f}"
              f"{_mean([r.get('official_reconstructed', 0.0) for r in rs]):>8.3f}"
              f"{_mean(hon):>9.3f}{_mean([r.get('honest_v2', r['honest']) for r in rs]):>8.3f}"
              f"{sd:>7.3f}{spread:>8.3f}"
              f"{_mean([r['wipes'] for r in rs]):>6.1f}"
              f"{_mean([r['actions'] for r in rs]):>7.0f}  {','.join(fl)}")

    off, hon = _mean([r["official"] for r in ok]), _mean([r["honest"] for r in ok])
    recon = _mean([r.get("official_reconstructed", 0.0) for r in ok])
    hon2 = _mean([r.get("honest_v2", r["honest"]) for r in ok])
    print(f"\n  OVERALL {name}:  official {off:.4f}   honest {hon:.4f}"
          f"   honest_v2 {hon2:.4f}   inflation {off / hon if hon else 0:.1f}x")
    if abs(recon - off) > 1e-6:
        print(f"  reconstruction (eval/official_score.py) says {recon:.4f} -- "
              f"{recon / off if off else float('inf'):.1f}x the card. The CARD is "
              f"the leaderboard; the reconstruction credits reset churn as fresh "
              f"levels. Do not quote the reconstruction.")
    swal = sum(int(r.get("swallowed_resets") or 0) for r in ok)
    if ok and ok[0].get("competition_mode"):
        print(f"  competition mode ON: {swal} RESET(s) refused (charged an action, "
              f"world unchanged, score survived).")
    print(f"  valid rows: {len(ok)}/{len(ok) + len(bad)}"
          + (f"   DROPPED {len(bad)} (cut short: "
             f"{sorted({str(r.get('stall') or r.get('error'))[:40] for r in bad})})"
             if bad else ""))
    # Dropping a cut-short row keeps the number honest about that row, but the
    # SURVIVORS are not a random sample: a cell is dropped for being slow, so a
    # change that costs time quietly removes its own worst cases and the mean it
    # leaves behind looks better than the agent is. Say so, loudly.
    drop_rate = len(bad) / max(1, len(ok) + len(bad))
    if drop_rate > 0.2:
        print(f"  ** {drop_rate:.0%} OF CELLS WERE DROPPED. This overall number is a "
              f"SURVIVOR-BIASED subset -- fast cells only. Lower --max-actions or "
              f"raise --timeout until the drop rate is under 20%, then re-run. **")
    if spreads:
        sp, gg = max(spreads)
        print(f"  NOISE FLOOR: widest per-game honest spread across seeds = "
              f"{sp:.3f} on {gg}. Any effect smaller than this is invisible here.")
    return {"official": off, "honest": hon, "honest_v2": hon2,
            "official_reconstructed": recon, "swallowed_resets": swal,
            "n_valid": len(ok), "n_invalid": len(bad)}


def report_routes(rows):
    """Where the budget went and what it bought. Spend alone cannot tell
    'working' from 'fidgeting' -- chg%/new% can."""
    ok = [r for r in rows if r.get("score_valid")]
    if not ok:
        return
    spend, bank, wipe, chg, new = (Counter() for _ in range(5))
    for r in ok:
        spend.update(r["spend"]); bank.update(r["banked"]); wipe.update(r["wiped"])
        chg.update(r.get("changed", {})); new.update(r.get("novel", {}))
    tot = sum(spend.values()) or 1
    print(f"\n  {'route':<22}{'actions':>9}{'share':>8}{'chg%':>7}{'new%':>7}"
          f"{'banked':>8}{'wiped':>7}{'per-1k':>9}")
    for r, n in spend.most_common(10):
        print(f"  {r:<22}{n:>9}{100.0 * n / tot:>7.1f}%"
              f"{100.0 * chg.get(r, 0) / n:>6.1f}%{100.0 * new.get(r, 0) / n:>6.1f}%"
              f"{bank.get(r, 0):>8}{wipe.get(r, 0):>7}"
              f"{1000.0 * bank.get(r, 0) / n:>9.2f}")


def paired_verdict(ctrl_rows, treat_rows, ctrl_name, treat_name, metric="honest"):
    """The only comparison this repo is allowed to make.

    Pairs on (game, seed) so the two arms differ by the switch and nothing else,
    then refuses to call a direction it cannot resolve.
    """
    ck = {(r["game"], r["seed"]): r for r in ctrl_rows if r.get("score_valid")}
    tk = {(r["game"], r["seed"]): r for r in treat_rows if r.get("score_valid")}
    keys = sorted(set(ck) & set(tk))
    print(f"\n{'=' * 74}\nPAIRED: {treat_name} vs {ctrl_name}   metric={metric}")
    dropped = (len(set(ck) | set(tk)) - len(keys))
    if dropped:
        print(f"  {dropped} pair(s) dropped: one side invalid or missing.")
        # Differential dropping is the failure mode that matters: if the treatment
        # is slower, ITS cells time out, those pairs vanish, and the comparison is
        # run on the ground where the treatment happened to be cheap.
        only_ctrl = len(set(ck) - set(tk))
        only_treat = len(set(tk) - set(ck))
        if only_ctrl != only_treat:
            print(f"  ** ASYMMETRIC: {only_treat} cell(s) valid only in {treat_name}, "
                  f"{only_ctrl} only in {ctrl_name}. One arm is being cut short more "
                  f"than the other, so the surviving pairs are not a fair sample. **")
        if dropped > len(keys):
            print(f"  ** More pairs were dropped ({dropped}) than kept ({len(keys)}). "
                  f"Treat any verdict below as untrustworthy. **")
    if len(keys) < 2:
        print(f"  {len(keys)} usable pair(s). VERDICT: NO EVIDENCE (need >= 2).")
        return {"verdict": "NO EVIDENCE", "n": len(keys)}

    deltas = [tk[k][metric] - ck[k][metric] for k in keys]
    mean = _mean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    n = len(deltas)
    t = mean / (sd / math.sqrt(n)) if sd > 0 else (math.inf if mean else 0.0)
    need = math.ceil((_POWER_Z * sd / abs(mean)) ** 2) if mean and sd else n

    print(f"  pairs {n}   mean delta {mean:+.4f}   sd {sd:.4f}   t {t:+.2f}")
    if mean == 0.0 and sd == 0.0:
        # Identical scores. Two very different things look like this, and
        # calling both "no effect" is how a component gets deleted for being
        # inert when it was actually working and merely not paying yet.
        moved = any(ck[k]["spend"] != tk[k]["spend"] for k in keys)
        if moved:
            print(f"  VERDICT: SCORE-NEUTRAL. The arms took DIFFERENT routes "
                  f"through every game and arrived at the same score. The "
                  f"component is live; it is not (yet) paying.")
            v = "SCORE-NEUTRAL"
        else:
            # An identical trace is evidence, not absence of it.
            print(f"  VERDICT: NO-OP. Every pair produced the same trajectory "
                  f"AND the same score on these {len({k[0] for k in keys})} "
                  f"game(s) -- the switch never fired. Either the component "
                  f"cannot trigger here, or it is dead code.")
            v = "NO-OP"
        return {"verdict": v, "n": n, "mean_delta": 0.0, "sd": 0.0,
                "t": 0.0, "needed_pairs": 0, "metric": metric}
    if abs(t) >= 2.0:
        verdict = "BETTER" if mean > 0 else "WORSE"
        print(f"  VERDICT: {verdict} (|t| >= 2). Effect survives the seed noise.")
    else:
        verdict = "NO EVIDENCE"
        print(f"  VERDICT: NO EVIDENCE. The observed difference is inside the "
              f"seed noise.")
        print(f"  To resolve an effect this size at 80% power you need "
              f"n >= {need} pairs ({math.ceil(need / max(1, len({k[0] for k in keys})))} "
              f"seeds over {len({k[0] for k in keys})} games). "
              f"You ran {n}. Do not merge on this.")

    per_game = defaultdict(list)
    for k, d in zip(keys, deltas):
        per_game[k[0]].append(d)
    movers = sorted(per_game.items(), key=lambda kv: -abs(_mean(kv[1])))[:8]
    print(f"\n  {'game':<7}{'mean d':>9}{'seeds':>7}  per-seed deltas")
    for g, ds in movers:
        if abs(_mean(ds)) < 1e-9:
            continue
        print(f"  {g:<7}{_mean(ds):>+9.3f}{len(ds):>7}  "
              + " ".join(f"{d:+.2f}" for d in ds))
    return {"verdict": verdict, "n": n, "mean_delta": mean, "sd": sd,
            "t": t, "needed_pairs": need, "metric": metric}


# ---------------------------------------------------------------- the sweep
def cost_model():
    """(secs, honest-sd) per game, measured from every stored run.

    The budget planner must not guess. If nothing has been stored yet it says
    so and falls back to a flat estimate, flagged as such.
    """
    secs, hon = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for p in glob.glob(os.path.join(SCORES_DIR, "*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        for r in d.get("rows") or []:
            if not r.get("score_valid") or not r.get("timing_valid"):
                continue
            g = r.get("game")
            if not g:
                continue
            secs[g].append(float(r.get("secs_active") or r.get("secs") or 0.0))
            hon[g][r.get("seed")].append(float(r.get("honest") or 0.0))
    cost, sd = {}, {}
    for g, xs in secs.items():
        if xs:
            cost[g] = statistics.median(xs)
    for g, per_seed in hon.items():
        # One value per seed (mean if a seed was run more than once), so the sd
        # is the SEED spread -- the thing that actually limits resolution.
        vals = [_mean(v) for v in per_seed.values() if v]
        if len(vals) > 1:
            sd[g] = statistics.stdev(vals)
    return cost, sd


def plan_budget(budget_h, n_arms, gids, seeds, jobs, cost, sd):
    """Fit (games x seeds) into the budget and SAY what it can resolve.

    Returns (gids, seeds, note). Games are dropped before seeds: seeds are what
    buy resolution, and a sweep that cannot resolve anything is worse than no
    sweep -- it produces a direction anyway and someone merges on it.
    """
    DEFAULT_S = 300.0
    gids, seeds = list(gids), list(seeds)

    def hours(gs, ss):
        per = sum(cost.get(g, DEFAULT_S) for g in gs) * len(ss) * n_arms
        return per / max(1, jobs) / 3600.0

    if budget_h is None:
        return gids, seeds, "no --budget-hours: running the full selection"
    notes = []
    while len(gids) > 5 and hours(gids, seeds) > budget_h:
        gids.sort(key=lambda g: -cost.get(g, DEFAULT_S))
        notes.append(f"dropped {gids[0]} ({cost.get(gids[0], DEFAULT_S):.0f}s/cell)")
        gids.pop(0)
        gids.sort()
    while len(seeds) > 3 and hours(gids, seeds) > budget_h:
        seeds.pop()
        notes.append(f"cut to seeds {seeds}")
    est = hours(gids, seeds)
    if est > budget_h:
        notes.append(f"STILL {est:.1f}h at the floor (5 games x 3 seeds). "
                     f"Raise --budget-hours or accept the overrun.")
    return gids, seeds, "; ".join(notes) or f"fits: est {est:.1f}h"


def sweep_resolution(gids, seeds, sd):
    """The smallest honest delta this selection can resolve, at 80% power.

    Printed BEFORE the run, because afterwards it is too late to learn that the
    answer was going to be NO EVIDENCE whatever happened.
    """
    n = len(gids) * len(seeds)
    have = [sd[g] for g in gids if g in sd]
    if not have or n < 2:
        return None, n, 0.0
    # Paired on (game, seed), so the delta's sd is at most sqrt(2)x the per-arm
    # sd and usually much less. Use the conservative bound.
    sd_pair = math.sqrt(2.0) * _mean(have)
    return _POWER_Z * sd_pair / math.sqrt(n), n, sd_pair


def report_sweep(by_arm, rows_ctrl, comps):
    """Rank every component arm against the shared control."""
    print(f"\n{'=' * 74}\nSWEEP: every component against ONE control arm")
    print(f"  {'component':<20}{'switch':<20}{'dHonest':>9}{'dOffic':>9}"
          f"{'think%':>8}{'pairs':>7}  verdict")
    out = []
    ctrl_ms = sum(sum(r.get("route_ms", {}).values())
                  for r in rows_ctrl if r.get("score_valid")) or 1.0
    for c in comps:
        arm = c["name"].strip()
        trows = by_arm.get(arm) or []
        v = paired_verdict(rows_ctrl, trows, "control", arm, metric="honest")
        vo = paired_verdict(rows_ctrl, trows, "control", arm, metric="official")
        share = 0.0
        if c.get("route"):
            share = 100.0 * sum(r.get("route_ms", {}).get(c["route"], 0.0)
                                for r in rows_ctrl if r.get("score_valid")) / ctrl_ms
        out.append({"component": arm, "switch": c["switch"],
                    "status": c["status"], "route_share": round(share, 1),
                    "honest": v, "official": vo})
    print(f"\n{'=' * 74}\nRANKED (by |dHonest|; the ablation TURNS THE COMPONENT OFF, "
          f"so a NEGATIVE delta means the component was earning its keep)")
    print(f"  {'component':<20}{'switch':<20}{'dHonest':>9}{'dOffic':>9}"
          f"{'think%':>8}{'pairs':>7}  verdict")
    for o in sorted(out, key=lambda o: -abs(o["honest"].get("mean_delta") or 0.0)):
        h, of = o["honest"], o["official"]
        print(f"  {o['component']:<20}{o['switch']:<20}"
              f"{(h.get('mean_delta') or 0.0):>+9.4f}"
              f"{(of.get('mean_delta') or 0.0):>+9.4f}"
              f"{o['route_share']:>7.1f}%{h.get('n', 0):>7}  {h['verdict']}"
              + (f"  (need n>={h['needed_pairs']})"
                 if h["verdict"] == "NO EVIDENCE" and h.get("needed_pairs") else "")
              + (f"  [{o['status']}]" if o["status"] != "live" else ""))
    print("\n  This ranks WHERE TO LOOK. It is not a merge gate, and re-running "
          "it on a schedule buys nothing.")
    return out


# ---------------------------------------------------------------- main
def _parse_seeds(spec: str):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _parse_env(pairs):
    env = {}
    for p in pairs or []:
        if "=" not in p:
            env[p] = "1"
        else:
            k, v = p.split("=", 1)
            env[k] = v
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="*", help="explicit game ids (overrides --split)")
    ap.add_argument("--label", required=True, help="name this run; stored as scores/<label>.json")
    ap.add_argument("--seeds", default="0-2", help="e.g. 0-4 or 0,3,7  (default 0-2)")
    ap.add_argument("--split", choices=["dev", "heldout", "all"], default="dev")
    ap.add_argument("--max-actions", type=int, default=1500)
    ap.add_argument("--timeout", type=float, default=None,
                    help="per-game wall cap in seconds (suspend time excluded)")
    ap.add_argument("--jobs", default="auto", help="parallel games (default: auto)")
    ap.add_argument("--agent-file", default="",
                    help="evaluate an alternate build by path, e.g. 'my_agent original.py'")
    ap.add_argument("--env", action="append", default=[],
                    help="KEY=VAL applied to the run (repeatable)")
    ap.add_argument("--ablate", action="append", default=[],
                    help="KEY=VAL -- run a paired control/treatment A/B in ONE invocation")
    ap.add_argument("--vs", default="",
                    help="label of a previous run to compare against (paired on game,seed)")
    # A4: default OFF. Every stored scores/*.json predates the rule, so a
    # default-ON would silently compare two different rulebooks. Sequence is:
    # land the port -> re-run the baseline arms with --competition-mode -> flip
    # this default -> stamp the jsons rules:"comp-v1".
    ap.add_argument("--competition-mode", dest="competition_mode",
                    action="store_true", default=False,
                    help="refuse the score-wiping RESET, as the hosted API does "
                         "(arc_agi/api.py:316-334). Default off for comparability.")
    ap.add_argument("--no-competition-mode", dest="competition_mode",
                    action="store_false")
    ap.add_argument("--sweep", action="store_true",
                    help="score every registered component against a shared "
                         "control arm (eval/components.py). Dev split only.")
    ap.add_argument("--budget-hours", type=float, default=None,
                    help="with --sweep: fit the selection to this wall budget and "
                         "print the smallest effect it can resolve")
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    files = discover_games()
    idx = games_index()
    split = load_split()

    if args.games:
        gids = [g for g in args.games if g in files]
        missing = [g for g in args.games if g not in files]
        if missing:
            print(f"unknown game ids: {missing}")
    elif args.split == "all":
        gids = sorted(files)
    else:
        gids = sorted(g for g in split[args.split] if g in files)
    if not gids:
        print("no games selected")
        return 1

    if args.split in ("heldout", "all") and not args.games:
        print("!! HELD-OUT DATA. Report the number; do not derive a change from it.")

    # PART F -- split discipline. The 8 held-out games exist to answer "does this
    # generalise?", and a search that reads them has already spent that answer.
    # Stochastic Goose: 12.58% on the preview, 0.25% on the full benchmark.
    if args.sweep:
        touched = set(gids) & set(split["heldout"])
        if touched or args.split in ("heldout", "all"):
            print(f"\nREFUSED: --sweep selects a component by comparing scores, "
                  f"which is tuning. Held-out games in this selection: "
                  f"{sorted(touched) or args.split}. Sweep the dev split; report "
                  f"held-out separately, once, after the decision is made.")
            return 2

    jobs = (max(1, min(8, (os.cpu_count() or 2) - 1)) if args.jobs == "auto"
            else max(1, int(args.jobs)))

    base_env = _parse_env(args.env)
    arms = {"main": base_env}
    comps = []
    if args.ablate:
        arms = {"control": base_env,
                "treatment": {**base_env, **_parse_env(args.ablate)}}
    elif args.sweep:
        from components import sweepable
        comps = sweepable()
        arms = {"control": base_env}
        for c in comps:
            arms[c["name"].strip()] = {**base_env, c["switch"]: c["on"]}
        cost, sd = cost_model()
        gids, seeds, note = plan_budget(args.budget_hours, len(arms), gids, seeds,
                                        jobs, cost, sd)
        mde, npairs, sd_pair = sweep_resolution(gids, seeds, sd)
        print(f"\nSWEEP PLAN: {len(comps)} components + 1 control = {len(arms)} arms")
        print(f"  budget: {note}")
        if mde is None:
            print(f"  NO STORED SEED SPREAD for these games: the smallest "
                  f"resolvable effect is UNKNOWN. Run a plain baseline over "
                  f"several seeds first, or read every verdict below as "
                  f"provisional.")
        else:
            print(f"  {npairs} pairs per component, paired delta sd ~{sd_pair:.3f} "
                  f"(measured). SMALLEST RESOLVABLE dHonest at 80% power: "
                  f"{mde:.4f}.")
            print(f"  Anything smaller than that WILL come back NO EVIDENCE no "
                  f"matter what the component does. Decide now whether that is "
                  f"worth the wall time.")

    print(f"bench: label={args.label}  agent={args.agent_file or 'my_agent.py'}")
    print(f"  split={args.split}  games={len(gids)}  seeds={seeds}  "
          f"cap={args.max_actions}  jobs={jobs}")
    print(f"  rules: {'comp-v1 (score-wiping RESET refused)' if args.competition_mode else 'legacy (RESET always performed)'}")
    for a, e in arms.items():
        print(f"  arm {a}: {e or '(unmodified)'}")
    cells = len(arms) * len(gids) * len(seeds)
    print(f"  {cells} cells to run\n")
    if len(seeds) < 3:
        print("  WARNING: fewer than 3 seeds. This can report a run, but it "
              "cannot support a verdict -- the seed spread is larger than every "
              "effect measured in this project so far.\n")

    t0 = time.time()
    raw = run_matrix(arms, gids, files, seeds, args.max_actions, args.timeout,
                     args.agent_file, jobs, idx=idx,
                     competition_mode=args.competition_mode)

    rows = []
    for r in raw:
        if r.get("error"):
            r.update({"score_valid": False, "timing_valid": False,
                      "flags": ["ERROR"], "official": 0.0, "honest": 0.0})
            rows.append(r)
            continue
        rows.append(score_row(r, idx.get(r["game"], {}), args.max_actions))

    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    summary = {}
    for arm in arms:
        print(f"\n{'=' * 74}\nARM: {arm}   {arms[arm] or '(unmodified)'}")
        summary[arm] = report_arm(by_arm[arm], arm, args.max_actions)
        report_routes(by_arm[arm])

    comparison = None
    sweep = None
    if args.sweep:
        sweep = report_sweep(by_arm, by_arm["control"], comps)
    elif args.ablate:
        comparison = paired_verdict(by_arm["control"], by_arm["treatment"],
                                    "control", f"ablate {arms['treatment']}")
    elif args.vs:
        prev_path = os.path.join(SCORES_DIR, f"{args.vs}.json")
        if not os.path.exists(prev_path):
            print(f"\n--vs: no such run: {prev_path}")
        else:
            with open(prev_path, encoding="utf-8") as f:
                prev = json.load(f)
            if sorted(prev.get("seeds") or []) != seeds:
                print(f"\n  NOTE: baseline seeds {prev.get('seeds')} != {seeds}; "
                      f"only matching (game,seed) pairs are compared.")
            comparison = paired_verdict(prev.get("rows") or [], by_arm["main"],
                                        args.vs, args.label)

    os.makedirs(SCORES_DIR, exist_ok=True)
    out = os.path.join(SCORES_DIR, f"{args.label}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "agent_file": args.agent_file, "split": args.split,
                   "games": gids, "seeds": seeds, "max_actions": args.max_actions,
                   # Stamped so a future --vs can tell whether the two runs were
                   # played under the same rulebook and scored by the same code.
                   "rules": "comp-v1" if args.competition_mode else "legacy",
                   "scorer": "scorecard",     # official read from EnvironmentScorecard
                   "honest_basis": "honest_v1 (games_index.json baselines)",
                   "arms": arms, "summary": summary, "comparison": comparison,
                   "sweep": sweep,
                   "wall_s": round(time.time() - t0, 1),
                   "rows": rows}, f, indent=1)
    print(f"\nwrote {out}   ({time.time() - t0:.0f}s wall)")

    if args.vs:
        prev_path = os.path.join(SCORES_DIR, f"{args.vs}.json")
        if os.path.exists(prev_path):
            with open(prev_path, encoding="utf-8") as f:
                prev = json.load(f)
            if prev.get("scorer") != "scorecard":
                print(f"  ** {args.vs} predates the scorecard port: its `official` "
                      f"is the reconstruction and is too high for any run that "
                      f"resets. `honest` IS comparable. **")
            if (prev.get("rules") or "legacy") != ("comp-v1" if args.competition_mode
                                                   else "legacy"):
                print(f"  ** RULE MISMATCH: {args.vs} was played under "
                      f"'{prev.get('rules') or 'legacy'}' rules, this run under "
                      f"'{'comp-v1' if args.competition_mode else 'legacy'}'. "
                      f"That is a different game, not a different agent. **")
    return 0


if __name__ == "__main__":
    sys.exit(main())
