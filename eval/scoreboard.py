"""The competition's OWN scorecard objects, driving the local harness.

Why this file exists
--------------------
`eval/official_score.py` RECONSTRUCTS what the leaderboard does from the frame
stream we observe. Measured 2026-08-07 on one scripted stream, that
reconstruction reported 7.2662 where the real `Scorecard` reported 0.84028 --
**8.6x over** -- because it credits a `full_reset()` score wipe as a fresh
completed level, where the real card starts a NEW PLAY and takes the MAX over
plays. Every `official` number this repo has printed for a run that resets is
therefore too high, the 0.31 champion figure included.

This module removes the reconstruction from the loop. It hands the run to the
same chain `arc_agi` runs when a scorecard is closed on the leaderboard:

    ScorecardManager.new_scorecard()                 one card per worker cell
    LocalEnvironmentWrapper(..., scorecard_manager=mgr, scorecard_id=card_id)
        -> EnvironmentWrapper._set_last_response()   on EVERY frame
        -> ScorecardManager.update_scorecard(guid, frame, frame.full_reset)
        -> Scorecard.new_play / reset / take_action / set_levels_completed
    EnvironmentScorecard.from_scorecard(card, [EnvironmentInfo, ...])
        -> EnvironmentScoreCalculator.add_level(...) -> .to_score()

Plays, resets, per-level action splits, the 115 cap, the level-index weighting
and the max-over-plays / mean-over-environments aggregation are all THEIR code.
Nothing here decides anything about the score; that is the entire point.

Kaggle_test.py keeps its own self-contained copy of these two objects -- there is
no `eval/` directory on Kaggle, and importing from one was one of the two bugs
that made that file unrunnable there. The copies are pinned to each other by
`tests/test_scorecard.py` case 6 (currently delta 0.00e+00), NOT by an import.

One card per worker cell, not one per run
-----------------------------------------
bench.py runs (arm, game, seed) cells as separate PROCESSES, so a single card
cannot span them. Each cell builds its own card holding exactly one environment;
the per-game number is unaffected (a game's score is computed from its own runs)
and bench aggregates across games itself, which is what `card.score` does anyway
(mean over environments).
"""
from __future__ import annotations

import os

try:
    from arc_agi.models import EnvironmentInfo as _EnvInfo
    from arc_agi.scorecard import EnvironmentScorecard, ScorecardManager
    HAVE_SCORECARD = True
    SCORECARD_IMPORT_ERROR = None
except Exception as _e:                                     # pragma: no cover
    _EnvInfo = EnvironmentScorecard = ScorecardManager = None
    HAVE_SCORECARD = False
    SCORECARD_IMPORT_ERROR = _e


def class_name_from_file(path):
    import importlib.util
    from arcengine import ARCBaseGame
    spec = importlib.util.spec_from_file_location(
        "_probe_" + os.path.splitext(os.path.basename(path))[0], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, obj in vars(mod).items():
        if isinstance(obj, type) and issubclass(obj, ARCBaseGame) and obj is not ARCBaseGame:
            return name
    raise RuntimeError(f"No ARCBaseGame subclass in {path}")


def make_env_info(game_file, gid, meta):
    """The competition's EnvironmentInfo for one game -> (info, source).

    `metadata.json` sits beside every shipped game and is what
    `Arcade._scan_for_environments()` loads; its `baseline_actions` is the list
    the official scorer divides by, and its game_id carries the version suffix
    that keys the scorecard. Prefer it. Fall back to games_index.json only when
    the game shipped without metadata.

    Baseline disagreement is not cosmetic: cn04 ships 21 baselines in one source
    and 15 in the other, and a length mismatch makes the official calculator
    return 0.0 with "Human baseline actions size mismatch" -- a silent zero that
    reads as agent failure. Callers must surface `source` and compare.
    """
    d = os.path.dirname(os.path.abspath(game_file))
    mpath = os.path.join(d, "metadata.json")
    if os.path.isfile(mpath):
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                info = _EnvInfo.model_validate_json(f.read())
            info.local_dir = d              # the file's own local_dir is stale
            if not info.baseline_actions:
                info.baseline_actions = list((meta or {}).get("baseline_steps") or [])
            return info, "metadata.json"
        except Exception as e:
            print(f"    [scorecard] unreadable {mpath}: {e}", flush=True)
    info = _EnvInfo(game_id=gid, class_name=class_name_from_file(game_file),
                    baseline_actions=list((meta or {}).get("baseline_steps") or []))
    info.local_dir = d
    return info, "games_index.json"


class OfficialScorecard:
    """The competition's scorecard, driven by the competition's own objects."""

    API_KEY = "local"

    def __init__(self, competition_mode=True):
        self.ok = HAVE_SCORECARD
        self.infos = []
        self.mgr = None
        self.card_id = None
        self.competition_mode = bool(competition_mode)
        if not self.ok:
            return
        self.mgr = ScorecardManager()
        # competition_mode is the flag api.py reads before refusing a
        # score-wiping RESET; set it so the card matches the run we play.
        self.card_id = self.mgr.new_scorecard(source_url=None, tags=None,
                                              api_key=self.API_KEY, opaque=None,
                                              competition_mode=self.competition_mode)

    # -- wiring ------------------------------------------------------------
    def register(self, info):
        if self.ok and not any(i.game_id == info.game_id for i in self.infos):
            self.infos.append(info)

    def wrapper_kwargs(self):
        """The three arguments that put a LocalEnvironmentWrapper on this card."""
        if not self.ok:
            return {"scorecard_id": "local"}
        return {"scorecard_id": self.card_id, "scorecard_manager": self.mgr}

    def raw(self):
        return self.mgr.get_scorecard(self.card_id, self.API_KEY) if self.ok else None

    # -- read-out ----------------------------------------------------------
    def compute(self):
        """-> EnvironmentScorecard (the object close_scorecard() returns)."""
        sc = self.raw()
        if sc is None:
            return None
        try:
            return EnvironmentScorecard.from_scorecard(sc, self.infos)
        except Exception as e:
            print(f"  [scorecard] from_scorecard failed: {type(e).__name__}: {e}",
                  flush=True)
            return None

    def game(self, gid, card=None):
        """The official numbers for one game, as a plain dict (or None).

        `gid` is matched loosely: metadata.json game ids carry a version suffix
        (`lp85-abc123`) that the bench's directory-derived id does not.
        """
        card = card or self.compute()
        if card is None:
            return None
        env = card.find_environment(gid)
        if env is None:
            for e in card.environments:
                if str(e.id).split("-")[0] == str(gid).split("-")[0]:
                    env = e
                    break
        if env is None:
            return None
        best = max(env.runs, key=lambda r: r.score) if env.runs else None
        return {
            "score": env.score,                 # MAX over plays -- the game's score
            "levels_completed": env.levels_completed,
            "actions": env.actions,
            "resets": env.resets,
            "plays": len(env.runs),
            "level_count": env.level_count,
            "level_scores": list(best.level_scores or []) if best else [],
            "level_actions": list(best.level_actions or []) if best else [],
            "baselines": list(best.level_baseline_actions or []) if best else [],
            "message": best.message if best else None,
            "per_play": [round(r.score, 4) for r in env.runs],
        }

    @staticmethod
    def _field(obj, name):
        """pydantic's computed_field usually materialises as a property, but a
        plain method survives on some versions -- accept either."""
        v = getattr(obj, name, None)
        return v() if callable(v) else v

    def overall(self, card=None):
        card = card or self.compute()
        if card is None:
            return None
        return {"score": card.score,            # mean over environments
                "environments": self._field(card, "total_environments"),
                "levels_completed": self._field(card, "total_levels_completed"),
                "levels": self._field(card, "total_levels"),
                "actions": self._field(card, "total_actions"),
                "per_game": {e.id: round(e.score, 4) for e in card.environments}}


def reset_would_wipe(wrapper) -> bool:
    """True if a RESET sent right now is the score-wiping one.

    arcengine/base_game.py:305 calls full_reset() -- which zeroes the score --
    when the engine's `_action_count` is 0. arc_agi/api.py:316-334 is where the
    hosted API refuses exactly that request: the action is spent, the world does
    not change, the score survives. Competition mode reproduces the refusal;
    this predicate is the condition it tests.
    """
    return getattr(getattr(wrapper, "_game", None), "_action_count", -1) == 0
