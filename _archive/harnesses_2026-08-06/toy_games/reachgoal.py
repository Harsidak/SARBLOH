"""ReachGoal: canonical goal-directed ARC-AGI-3-style puzzle.

An avatar (color 2) moves on a grid via ACTION1-4 (up/down/left/right). Walls
(color 8) block movement. Reaching the goal cell (color 4) completes the level.
Multiple levels with different layouts. This is the simplest test of the whole
loop the real games demand: learn the movement world-model, INFER the goal
(reach the color-4 cell), and PLAN a short path to it.

Grid convention matches arcengine actions: ACTION1=up, 2=down, 3=left, 4=right.
"""
from arcengine import ARCBaseGame, Camera, Level, Sprite
from arcengine.enums import GameAction, InteractionMode

AVATAR = 2
GOAL = 4
WALL = 8
BG = 0

# Each level: (grid_w, grid_h, avatar_xy, goal_xy, [wall_xy, ...])
LEVELS = [
    (8, 8, (1, 1), (6, 6), [(3, 3), (3, 4), (4, 3)]),
    (10, 10, (0, 0), (9, 9), [(5, y) for y in range(0, 8)]),          # wall forces a detour
    (10, 10, (9, 0), (0, 9), [(4, y) for y in range(2, 10)] + [(7, y) for y in range(0, 7)]),
]


def _sprite(color, x, y, name, blocking=True):
    kw = {}
    if not blocking:
        kw["interaction"] = InteractionMode.INTANGIBLE
        kw["collidable"] = False
    return Sprite(pixels=[[color]], x=x, y=y, name=name, **kw)


class ReachGoal(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = []
        for i, (w, h, (ax, ay), (gx, gy), walls) in enumerate(LEVELS):
            sprites = [
                _sprite(GOAL, gx, gy, "goal", blocking=False),
                _sprite(AVATAR, ax, ay, "avatar", blocking=True),
            ]
            for j, (wx, wy) in enumerate(walls):
                sprites.append(_sprite(WALL, wx, wy, f"wall_{j}", blocking=True))
            levels.append(Level(sprites=sprites, grid_size=(w, h), name=f"L{i}"))
        super().__init__(
            game_id="reachgoal",
            levels=levels,
            camera=Camera(background=BG),
            win_score=len(levels),
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )

    def step(self) -> None:
        aid = self._action.id
        deltas = {
            GameAction.ACTION1: (0, -1),  # up
            GameAction.ACTION2: (0, 1),   # down
            GameAction.ACTION3: (-1, 0),  # left
            GameAction.ACTION4: (1, 0),   # right
        }
        if aid in deltas:
            dx, dy = deltas[aid]
            avatar = self.current_level.get_sprites_by_name("avatar")[0]
            # keep inside the grid
            w, h = self.current_level.grid_size
            nx, ny = avatar.x + dx, avatar.y + dy
            if 0 <= nx < w and 0 <= ny < h:
                self.try_move("avatar", dx, dy)   # blocked by walls automatically
            goal = self.current_level.get_sprites_by_name("goal")[0]
            if avatar.x == goal.x and avatar.y == goal.y:
                self.next_level()
        self.complete_action()
