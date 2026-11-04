"""A parameterised world model of the ARC-AGI-3 game ls20, with a planner.

Everything the game does is described by an `Ls20Params` object. The agent fills it in from what it sees on the
frame (where the walls, goals and changers are, how big the step budget is) and can correct any field when the real
game disagrees. `step(params, state, action)` predicts the next state without touching the game, and
`plan(params)` searches for the shortest action list that wins the level.

Coordinates are frame pixels (x right, y down). The player is a `cell` x `cell` box (5 x 5 in ls20) and moves one
box per action. Something "is in" a box when its top-left pixel lies inside that box.

Actions: 1 up, 2 down, 3 left, 4 right.

Rules, in the order the game applies them on each move:
  1. Moving changers advance one cell along their track.
  2. The player tries to move one box. A wall blocks it. A goal blocks it unless the key matches that goal.
     Anything the player moves onto fires: a shape, colour or rotation changer turns the key once; a refill puts
     the budget back to full and is used up.
  3. A blocked move still costs budget (bumping a goal with the wrong key does not).
  4. If the player now touches a pusher, it is thrown until the next wall or goal.
  5. Standing on a goal with a matching key completes that goal. All goals complete = level won.
  6. If the budget ran out, a life is lost and the level resets (budget, key, goals, refills).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

ACTIONS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
ACTION_NAMES = {1: "up", 2: "down", 3: "left", 4: "right"}
COLORS = [12, 9, 14, 8]  # the key's colour cycle, as frame palette values
ROTATIONS = [0, 90, 180, 270]  # the key's rotation cycle, in degrees
N_SHAPES = 6
CHANGER_KINDS = ("shape", "color", "rotation")


@dataclass(frozen=True)
class Key:
    """The key the player carries: shape index, colour value and rotation in degrees."""

    shape: int
    color: int
    rotation: int

    def turned(self, kind: str) -> "Key":
        if kind == "shape":
            return replace(self, shape=(self.shape + 1) % N_SHAPES)
        if kind == "color":
            return replace(self, color=COLORS[(COLORS.index(self.color) + 1) % len(COLORS)])
        if kind == "rotation":
            return replace(self, rotation=ROTATIONS[(ROTATIONS.index(self.rotation) + 1) % len(ROTATIONS)])
        raise ValueError(f"unknown changer kind {kind!r}")


@dataclass(frozen=True)
class Goal:
    x: int
    y: int
    key: Key  # the key the player must carry to enter and complete it


@dataclass(frozen=True)
class Pusher:
    """A spring: touching its box throws the player (dx, dy) until the cell before a wall or goal."""

    x: int
    y: int
    w: int
    h: int
    dx: int
    dy: int


@dataclass(frozen=True)
class MovingChanger:
    """A changer that walks along a track one cell per move, preferring straight, then left/right, then back."""

    x: int
    y: int
    kind: str
    track: frozenset  # every (x, y) pixel that is on the track


@dataclass
class Ls20Params:
    """Everything the model needs to know about one level. Every field can be edited."""

    player: tuple[int, int]
    start_key: Key
    goals: list[Goal]
    walls: set[tuple[int, int]] = field(default_factory=set)
    changers: dict[tuple[int, int], str] = field(default_factory=dict)  # position -> "shape"|"color"|"rotation"
    refills: set[tuple[int, int]] = field(default_factory=set)
    pushers: list[Pusher] = field(default_factory=list)
    moving_changers: list[MovingChanger] = field(default_factory=list)
    budget: int = 42  # bars on the HUD when the level starts
    cost_per_move: int = 2  # bars one move uses; 21 moves at 42/2, 42 moves at 42/1
    lives: int = 3
    cell: int = 5
    # In the first level, a move that makes the key match an open goal is free and ends the turn early.
    free_match_move: bool = False

    @property
    def moves_per_budget(self) -> int:
        return self.budget // self.cost_per_move


@dataclass(frozen=True)
class State:
    x: int
    y: int
    key: Key
    budget: int
    lives: int
    done: frozenset = frozenset()  # indices of completed goals
    used_refills: frozenset = frozenset()
    movers: tuple = ()  # (x, y, direction) for each moving changer
    won: bool = False
    lost: bool = False


def initial_state(p: Ls20Params) -> State:
    return State(
        x=p.player[0],
        y=p.player[1],
        key=p.start_key,
        budget=p.budget,
        lives=p.lives,
        movers=tuple((m.x, m.y, 0) for m in p.moving_changers),
    )


def _in_box(pt: tuple[int, int], x: int, y: int, size: int) -> bool:
    return x <= pt[0] < x + size and y <= pt[1] < y + size


_DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # moving-changer directions: down, right, up, left


def _advance(m: MovingChanger, x: int, y: int, d: int, cell: int) -> tuple[int, int, int]:
    for nd in (d, (d - 1) % 4, (d + 1) % 4, (d + 2) % 4):
        nx, ny = x + _DIRS[nd][0] * cell, y + _DIRS[nd][1] * cell
        if (nx, ny) in m.track:
            return nx, ny, nd
    return x, y, d


def _reset_level(p: Ls20Params, s: State, lives: int) -> State:
    return replace(initial_state(p), lives=lives)


def step(p: Ls20Params, s: State, action: int) -> State:
    """Predict the state after one action (1 up, 2 down, 3 left, 4 right)."""
    if s.won or s.lost:
        return s
    dx, dy = ACTIONS[action]
    c = p.cell

    old_movers = s.movers
    movers = tuple(_advance(m, x, y, d, c) for m, (x, y, d) in zip(p.moving_changers, s.movers))

    tx, ty = s.x + dx * c, s.y + dy * c
    key, budget = s.key, s.budget
    blocked = refilled = wrong_goal = False
    used = set(s.used_refills)
    open_goals = [i for i in range(len(p.goals)) if i not in s.done]

    # Everything whose top-left pixel lies in the target box fires, in no particular order.
    if any(_in_box(w, tx, ty, c) for w in p.walls):
        blocked = True
    for i in open_goals:
        g = p.goals[i]
        if _in_box((g.x, g.y), tx, ty, c) and key != g.key:
            blocked = wrong_goal = True
    for r in p.refills:
        if r not in used and _in_box(r, tx, ty, c):
            refilled = True
            used.add(r)
    fired = [kind for pos, kind in p.changers.items() if _in_box(pos, tx, ty, c)]
    fired += [m.kind for m, (x, y, _) in zip(p.moving_changers, movers) if _in_box((x, y), tx, ty, c)]
    for kind in fired:
        key = key.turned(kind)

    if blocked:
        movers = old_movers  # changers step back when the player could not move
        nx, ny = s.x, s.y
    else:
        nx, ny = tx, ty

    if wrong_goal or (
        p.free_match_move and fired and any(key == p.goals[i].key for i in open_goals)
    ):
        return replace(s, x=nx, y=ny, key=key, used_refills=frozenset(used), movers=movers)

    if refilled:
        budget, out = p.budget, False
    else:
        budget -= p.cost_per_move
        out = budget < 0

    if not out:
        for pu in p.pushers:
            if pu.x < nx + c and nx < pu.x + pu.w and pu.y < ny + c and ny < pu.y + pu.h:
                k = _push_distance(p, pu)
                if k > 0:
                    nx, ny = nx + pu.dx * pu.w * k, ny + pu.dy * pu.h * k
                    for kind in [kd for pos, kd in p.changers.items() if _in_box(pos, nx, ny, c)]:
                        key = key.turned(kind)
                    for r in p.refills:
                        if r not in used and _in_box(r, nx, ny, c):
                            used.add(r)
                            budget = p.budget
                    # The game does not check for a win at the end of a throw.
                    return replace(s, x=nx, y=ny, key=key, budget=budget, used_refills=frozenset(used), movers=movers)

    done = set(s.done)
    for i in open_goals:
        if (nx, ny) == (p.goals[i].x, p.goals[i].y) and key == p.goals[i].key:
            done.add(i)
    if len(done) == len(p.goals):
        return replace(s, x=nx, y=ny, key=key, budget=budget, done=frozenset(done), won=True)

    if out:
        if s.lives - 1 <= 0:
            return replace(s, lives=0, lost=True)
        return _reset_level(p, s, s.lives - 1)

    return State(nx, ny, key, budget, s.lives, frozenset(done), frozenset(used), movers)


def _push_distance(p: Ls20Params, pu: Pusher) -> int:
    stops = set(p.walls) | {(g.x, g.y) for g in p.goals}
    cx, cy = pu.x + pu.dx, pu.y + pu.dy
    for k in range(1, 12):
        if (cx + pu.dx * pu.w * k, cy + pu.dy * pu.h * k) in stops:
            return k - 1
    return 0


def rollout(p: Ls20Params, actions: Iterable[int], s: Optional[State] = None) -> list[State]:
    """Every state along an action list, starting with the initial one."""
    s = s or initial_state(p)
    out = [s]
    for a in actions:
        s = step(p, s, a)
        out.append(s)
    return out


def plan(p: Ls20Params, s: Optional[State] = None, max_nodes: int = 2_000_000) -> Optional[list[int]]:
    """Shortest action list that wins the level without losing a life, or None if none was found."""
    s = s or initial_state(p)

    def sig(t: State):
        return (t.x, t.y, t.key, t.done, t.used_refills, t.movers)

    best = {sig(s): s.budget}
    queue = deque([(s, [])])
    while queue and len(best) < max_nodes:
        cur, path = queue.popleft()
        for a in (1, 2, 3, 4):
            nxt = step(p, cur, a)
            if nxt.won:
                return path + [a]
            if nxt.lost or nxt.lives < cur.lives:
                continue
            k = sig(nxt)
            if best.get(k, -1) >= nxt.budget:
                continue
            best[k] = nxt.budget
            queue.append((nxt, path + [a]))
    return None


def describe(actions: Iterable[int]) -> str:
    return " ".join(ACTION_NAMES[a] for a in actions)
