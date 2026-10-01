"""THE single source of truth for scoring. Nothing else may reimplement RHAE.

Why this module exists
----------------------
Every harness in this repo used to carry its own copy of the RHAE formula, read
off the ARC-AGI-3 paper. All of them computed a number that the competition does
NOT compute, and the difference was worth ~400x. The fix is to stop transcribing
the formula and instead call the scorer the competition ships
(`arc_agi.scorecard.EnvironmentScoreCalculator`), so our number cannot drift from
theirs by construction.

Two numbers, always reported together
-------------------------------------
`official` -- what the leaderboard computes, 0..100.

    Driven by `Card.actions_by_level`, a list appended to by
    `Card.set_levels_completed` whenever `levels_completed` CHANGES -- including
    when it DROPS. `_calculate_score` then reads the i-th entry as "level i+1,
    completed", ignoring the level number that was recorded. So an agent that
    banks level 1, triggers `full_reset()` (which sets `_score = 0`), re-banks
    level 1, and repeats, is credited with a completed level per cycle -- each
    one cheap enough to hit the 115% per-level cap.

    Measured on Test1 (the 0.31 build, 25 public games): lp85 reached level 1
    and never went further, yet scores 50.46% off 6 wipe events. Honest RHAE for
    the same trajectory is 0.0012.

`honest` -- what the agent can actually do, 0..1.

    Monotonic level accounting: a level counts once, the first time it is
    reached, and the actions charged to it are all actions since the previous
    first-reach (deaths and resets included, exactly as the competition charges
    them). Same weights, same cap, same denominator -- only the double-counting
    is removed. This is the number to steer capability on.

`wipes` / `inflation` make the difference legible: if `inflation` is large, the
official score is being carried by reset churn rather than by solving anything.

How full_reset is triggered (arcengine/base_game.py:305)
--------------------------------------------------------
    if ONLY_RESET_LEVELS == "true" and state != WIN:  level_reset()
    elif self._action_count == 0 or state == WIN:     full_reset()   # wipes score
    else:                                             level_reset()

`full_reset()` sets `_action_count = 0`, so a SECOND RESET with no action in
between wipes every banked level. `ONLY_RESET_LEVELS` is read by `arc_agi.api`
but is not set anywhere in this repo, so local runs use wiping semantics.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Optional

# The official calculator. If it is missing we fall back to a transcription that
# `test_metrics.py` pins against the real one whenever the real one is present,
# so the fallback can never silently drift.
try:
    from arc_agi.scorecard import EnvironmentScoreCalculator as _OfficialCalc
except Exception:                                          # pragma: no cover
    _OfficialCalc = None


class _FallbackCalc:
    """Faithful transcription of arc_agi.scorecard.EnvironmentScoreCalculator.

    Kept byte-for-byte equivalent in behaviour, including the parts that look
    like bugs -- this module reports what the competition scores, not what it
    ought to score.
    """

    def __init__(self, **_kw):
        self.level_indices, self.level_scores = [], []
        self.level_actions, self.level_baseline_actions = [], []
        self.levels_completed, self.actions = 0, 0

    def add_level(self, level_index, completed, actions_taken, baseline_actions,
                  game_id=None):
        self.actions += actions_taken
        if completed:
            self.levels_completed += 1
            score = (min(((baseline_actions / actions_taken) ** 2) * 100, 115.0)
                     if actions_taken > 0 else 0.0)
        else:
            score = 0.0
        self.level_indices.append(level_index)
        self.level_scores.append(score)
        self.level_actions.append(actions_taken)
        self.level_baseline_actions.append(baseline_actions)

    def to_score(self):
        if not self.level_scores:
            return _Score(0.0, 0, self.actions, [], [], [])
        total_score = total_weights = max_weights = 0.0
        for i, s in enumerate(self.level_scores):
            w = self.level_indices[i]
            total_score += s * w
            total_weights += w
            if s > 0:
                max_weights += w
        score = min(total_score / total_weights, max_weights / total_weights * 100)
        return _Score(score, self.levels_completed, self.actions,
                      self.level_scores, self.level_actions,
                      self.level_baseline_actions)


class _Score:
    __slots__ = ("score", "levels_completed", "actions", "level_scores",
                 "level_actions", "level_baseline_actions")

    def __init__(self, score, levels_completed, actions, level_scores,
                 level_actions, level_baseline_actions):
        self.score = score
        self.levels_completed = levels_completed
        self.actions = actions
        self.level_scores = level_scores
        self.level_actions = level_actions
        self.level_baseline_actions = level_baseline_actions


def _calc(**kw):
    return (_OfficialCalc(**kw) if _OfficialCalc is not None else _FallbackCalc(**kw))


# --------------------------------------------------------------------------
# event recording -- mirrors Card.set_levels_completed EXACTLY
# --------------------------------------------------------------------------
class LevelEvents:
    """Records (levels_completed, cumulative_actions) on every CHANGE.

    Feed it the frame's `levels_completed` after every action, together with the
    action counter. Appending on a DROP as well as a rise is not an oversight --
    it is what `Card.set_levels_completed` does, and reproducing it is the whole
    point of this class.
    """

    def __init__(self):
        self.events: list[tuple[int, int]] = []
        self._cur = 0
        self.first_reach: dict[int, int] = {}     # level -> cumulative actions
        self._peak = 0

    def observe(self, levels_completed: int, action_counter: int) -> bool:
        """Returns True if this observation was a level EVENT."""
        lv = int(levels_completed or 0)
        if lv == self._cur:
            return False
        self.events.append((lv, int(action_counter)))
        self._cur = lv
        # Monotonic side-ledger for the honest score: a level is banked once.
        while lv > self._peak:
            self._peak += 1
            self.first_reach.setdefault(self._peak, int(action_counter))
        return True

    @property
    def wipes(self) -> int:
        """Level-change events that went DOWN -- each one is free score."""
        return sum(1 for i, (lv, _) in enumerate(self.events)
                   if i > 0 and lv < self.events[i - 1][0])

    @property
    def real_levels(self) -> int:
        return self._peak

    def to_json(self) -> dict:
        return {"events": self.events, "wipes": self.wipes,
                "real_levels": self.real_levels,
                "first_reach": dict(self.first_reach)}


# --------------------------------------------------------------------------
# the two scores
# --------------------------------------------------------------------------
def official_score(events: Iterable[tuple[int, int]], total_actions: int,
                   baselines: list[int]) -> _Score:
    """Reproduce arc_agi.scorecard._calculate_score for one environment run.

    `events` is Card.actions_by_level: the i-th entry is charged as level i+1
    and treated as COMPLETED regardless of the level number it carries.
    """
    events = list(events)
    calc = _calc(resets=0, guid="local", state=None)
    prev = 0
    for i, baseline in enumerate(baselines):
        if i < len(events):
            _lvl, at = events[i]
            acts, done = at - prev, True
            prev = at
        else:
            acts, done = total_actions - prev, False
            prev = total_actions
        calc.add_level(level_index=i + 1, completed=done,
                       actions_taken=acts, baseline_actions=baseline)
    return calc.to_score()


def honest_score(first_reach: dict[int, int], total_actions: int,
                 baselines: list[int]) -> _Score:
    """Same formula, same weights, same cap -- but a level counts ONCE.

    `first_reach` maps level number (1-indexed) to the cumulative action count at
    which that level was first banked. Actions charged to level k are all actions
    since level k-1 was first banked, so deaths and resets in between are paid
    for, exactly as the competition charges them.
    """
    calc = _calc(resets=0, guid="local", state=None)
    prev = 0
    for i, baseline in enumerate(baselines):
        at = first_reach.get(i + 1)
        if at is not None:
            acts, done = at - prev, True
            prev = at
        else:
            acts, done = max(0, total_actions - prev), False
            prev = total_actions
        calc.add_level(level_index=i + 1, completed=done,
                       actions_taken=acts, baseline_actions=baseline)
    return calc.to_score()


def score_run(ev: LevelEvents, total_actions: int, baselines: list[int],
              win_levels: Optional[int] = None) -> dict:
    """Both numbers plus the diagnostics that explain the difference.

    `win_levels` is accepted only to WARN on disagreement: the official scorer
    iterates `range(len(baseline_actions))`, so the baseline list -- not the
    frame's win_levels -- sets the denominator. cn04 ships win_levels=5 with 6
    baselines, and using 5 would silently inflate every cn04 score by 21/15.
    """
    baselines = list(baselines or [])
    if not baselines:
        return {"official": 0.0, "honest": 0.0, "honest_rhae": 0.0,
                "real_levels": ev.real_levels, "credited_levels": 0,
                "wipes": ev.wipes, "inflation": 0.0, "levels_denominator": 0,
                "official_level_scores": [], "honest_level_actions": [],
                "warn": "no baselines: the competition scores this run 0"}
    o = official_score(ev.events, total_actions, baselines)
    h = honest_score(ev.first_reach, total_actions, baselines)
    warn = None
    if win_levels and win_levels != len(baselines):
        warn = (f"win_levels={win_levels} but {len(baselines)} baselines; "
                f"denominator follows the baselines")
    return {
        # BOTH in leaderboard units, 0..100. Reporting one as a percent and the
        # other as a fraction is how the 0.31-vs-0.0001 confusion started, so
        # this module never mixes scales. `honest_rhae` is the same number in the
        # 0..1 form the older logs and memories use.
        "official": o.score,
        "honest": h.score,
        "honest_rhae": h.score / 100.0,
        "real_levels": ev.real_levels,
        "credited_levels": o.levels_completed,    # what the scorer thinks
        "wipes": ev.wipes,
        "inflation": (o.score / h.score) if h.score > 0 else (
            float("inf") if o.score > 0 else 0.0),
        "levels_denominator": len(baselines),
        "official_level_scores": [round(x, 2) for x in o.level_scores],
        "honest_level_actions": h.level_actions,
        # Published so no caller ever has to write `min(1.15, human/ai)**2`
        # again to show a per-level number. Same units as the rest: 0..100.
        "honest_level_scores": [round(x, 2) for x in h.level_scores],
        "warn": warn,
    }


# --------------------------------------------------------------------------
def level_score(baseline_actions: int, actions_taken: int) -> float:
    """One level's score, 0..100, from the official calculator.

    Exists so a caller can say "this level cost X actions, so it scores Y"
    without writing `min(1.15, human/ai)**2` down again. Every place that wrote
    it down was eventually wrong.
    """
    if not actions_taken or actions_taken <= 0:
        return 0.0
    calc = _calc(resets=0, guid="local", state=None)
    calc.add_level(level_index=1, completed=True,
                   actions_taken=int(actions_taken),
                   baseline_actions=int(baseline_actions))
    return calc.to_score().level_scores[0]


def load_baselines(games_index_path: str) -> dict:
    """id -> {'baseline_steps': [...], 'win_levels': n}."""
    with open(games_index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {g["id"]: {"baseline_steps": list(g.get("baseline_steps") or []),
                      "win_levels": int(g.get("win_levels") or 0)}
            for g in data}


DEFAULT_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "real_games", "games_index.json")
