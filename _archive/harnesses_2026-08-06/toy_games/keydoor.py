"""KeyDoor: a stress test with DECOY objects and a different avatar color.

Avatar (color 3) must reach the door (color 6). The board also contains decoy
objects (color 7) that are NOT the goal and NOT walls (walkable) plus walls
(color 8). Reaching a decoy does nothing; only the door completes the level.
This checks that goal inference doesn't just chase "any distinctive object" and
that obstacle learning separates walls from walkable decoys.
"""
from arcengine import ARCBaseGame, Camera, Level, Sprite
from arcengine.enums import GameAction, InteractionMode

AVATAR = 3
DOOR = 6
DECOY = 7
WALL = 8
BG = 0

# (grid, avatar_xy, door_xy, [decoys], [walls])
LEVELS = [
    (12, 12, (1, 1), (10, 10), [(2, 9), (9, 2)], [(6, y) for y in range(0, 9)]),
    (12, 12, (11, 0), (0, 11), [(5, 5), (6, 6)],
     [(4, y) for y in range(3, 12)] + [(8, y) for y in range(0, 9)]),
]


def _sprite(color, x, y, name, blocking=True):
    kw = {}
    if not blocking:
        kw["interaction"] = InteractionMode.INTANGIBLE
        kw["collidable"] = False
    return Sprite(pixels=[[color]], x=x, y=y, name=name, **kw)


class KeyDoor(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = []
        for i, (w, h, (ax, ay), (dx, dy), decoys, walls) in enumerate(LEVELS):
            sprites = [_sprite(DOOR, dx, dy, "door", blocking=False)]
            for j, (ex, ey) in enumerate(decoys):
                sprites.append(_sprite(DECOY, ex, ey, f"decoy_{j}", blocking=False))
            for j, (wx, wy) in enumerate(walls):
                sprites.append(_sprite(WALL, wx, wy, f"wall_{j}", blocking=True))
            sprites.append(_sprite(AVATAR, ax, ay, "avatar", blocking=True))
            levels.append(Level(sprites=sprites, grid_size=(w, h), name=f"L{i}"))
        super().__init__(game_id="keydoor", levels=levels, camera=Camera(background=BG),
                         win_score=len(levels), available_actions=[1, 2, 3, 4], seed=seed)

    def step(self) -> None:
        deltas = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
                  GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
        aid = self._action.id
        if aid in deltas:
            dx, dy = deltas[aid]
            avatar = self.current_level.get_sprites_by_name("avatar")[0]
            w, h = self.current_level.grid_size
            nx, ny = avatar.x + dx, avatar.y + dy
            if 0 <= nx < w and 0 <= ny < h:
                self.try_move("avatar", dx, dy)
            door = self.current_level.get_sprites_by_name("door")[0]
            if avatar.x == door.x and avatar.y == door.y:
                self.next_level()
        self.complete_action()
