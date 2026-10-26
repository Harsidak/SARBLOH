"""Collects every number the paper plots into Writing/data/paper_data.json, each with its source.

Sources: the Kaggle submission history (pulled 2026-10-26 with KaggleApi.competition_submissions, copied below), the
best offline run's summary.json, the TAAF benchmark.json on the same server, experiments/E115 and E120 (the Gemini
ladder). Nothing here is typed from memory except where a comment names the experiment file it comes from."""

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/kaggle_sarbloh-experimentation_20261023_115157/prime_run/summary.json"
TAAF = REPO / "runs/TAAF results/benchmark.json"

# Kaggle public leaderboard, every COMPLETE submission of team Banwait_13 (UTC dates), with the kernel it came from.
# ERROR submissions (system errors in the Phase B rerun) are listed with score None.
SUBMISSIONS = [
    ("2026-08-01", 0.07, "v-o-i-d-agent", "legacy agent"),
    ("2026-08-02", 0.15, "v-o-i-d-agent", "legacy agent"),
    ("2026-08-15", 0.00, "v-o-i-d-agent", "legacy agent"),
    ("2026-09-28", 0.10, "sarbloh-prime", "Prime Agent port, Gemma-4-31B, vLLM"),
    ("2026-09-29", 0.00, "sarbloh-prime", "Prime Agent port"),
    ("2026-10-01", 0.00, "sarbloh-prime", "Prime Agent port"),
    ("2026-10-02", 0.27, "sarbloh-prime", "Prime Agent port"),
    ("2026-10-03", 0.19, "sarbloh-experimentation", "first experimentation build (perception engine X8 wired in on 10-03)"),
    ("2026-10-05", 5.14, "sarbloh-experimentation", "Flash-Next + speculative decoding on SGLang, prefix-cache prompts, 10 games at once"),
    ("2026-10-08", 5.65, "sarbloh-experimentation", "one tool: moves sent from code"),
    ("2026-10-11", 6.06, "sarbloh-experimentation", "drain-only context, xhigh effort"),
    ("2026-10-15", 0.35, "sarbloh-experimentation", "regression (harness change)"),
    ("2026-10-16", 2.97, "sarbloh-experimentation", "regression (harness and backbone trials)"),
    ("2026-10-21", 10.98, "sarbloh-experimentation", "code graph, level memory, one-turn cycle, guardrails"),
    ("2026-10-22", 1.96, "sarbloh-experimentation", "over-budget run (10 h)"),
    ("2026-10-26", 16.53, "sarbloh-experimentation", "memory by level, Duck-style drain, briefing map back on"),
]

# experiments/E120_glm/hypothesis.md, the owner's Gemini experiment of 2026-09-29 (ls20, level 1).
GEMINI = [("raw grid", 250, "200-300, game lost"), ("+ image", 18, "12-24"),
          ("+ board explained", 12, "12"), ("+ reason with code", 6, "6")]

# experiments/E115_perception_lab/verdict.md: rubric /28 on two dev screenshots, text tokens per frame.
LAB = [("E0 old eye", 3, 3500), ("X1 inventory", 3, 180), ("X2 surprise", 4, 270), ("X4 look-alikes", 6, 140),
       ("X6 blueprint", 8, 100), ("X5 foresight", 12, 140), ("X3 routines", 15, 240), ("X7 roles", 20, 380),
       ("X8 fusion", 25, 370)]
HOLDOUT = {"screenshot to grid": (23, 23), "YOU (keyboard games)": (6, 14), "gauge (moves left)": (10, 20),
           "click handles that respond": (7, 29), "keyboard foresight": (1, 16)}


def per_game() -> dict:
    ours = json.loads(RUN.read_text(encoding="utf-8"))["per_game"]
    b = json.loads(TAAF.read_text(encoding="utf-8"))
    taaf: dict[str, int] = {}
    for run in b["game_runs"]:
        taaf[run["game_id"]] = max(taaf.get(run["game_id"], 0), int(run["levels_completed"]))
    return {g.split("-")[0]: {"ours": v["levels"], "of": v["of"], "ours_score": v["score"],
                              "taaf_best_of_4": taaf.get(g, 0)} for g, v in sorted(ours.items())}


if __name__ == "__main__":
    s = json.loads(RUN.read_text(encoding="utf-8"))
    keep = {k: s[k] for k in ("mean_score", "levels_completed", "levels_total", "actions", "turns", "tool_calls",
                               "act_calls", "curator_runs", "skills_written", "images_sent", "wall_s", "llm")}
    out = {"submissions": SUBMISSIONS, "gemini": GEMINI, "lab": LAB, "holdout": HOLDOUT, "best_run": keep,
           "per_game": per_game()}
    (REPO / "Writing/data/paper_data.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    pg = out["per_game"]
    print(sum(v["taaf_best_of_4"] for v in pg.values()), sum(v["ours"] for v in pg.values()))
    for g, v in pg.items():
        print(g, v)
