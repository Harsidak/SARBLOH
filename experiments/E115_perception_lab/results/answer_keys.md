# Answer keys (written and engine-verified BEFORE any experiment was run)

## ls20, level 1 (screenshot 1)
- K1 CONTROL: the 5x5 two-tone block (orange top, blue bottom). ACTION1/2/3/4 = up/down/left/right,
  moves one 5-cell tile. Verified: ACTION1 (45,34)->(40,34); ACTION2 blocked (bottom is wall).
- K2 GOAL: reach the black box at the top (the lock) while your key matches the glyph shown in it.
- K3 LINK: bottom-left HUD icon = your current key (drawn 2x). Lock glyph = key rotated. They must match.
- K4 MODIFIER: the white/grey plus rotates your key when you step on it (verified: icon changed on step 6).
- K5 HUD: yellow bar = moves left (one column per action, even blocked moves); 3 red squares = lives.
- K6 PLAN: LEFT x3, UP x3 (onto plus), UP, RIGHT x3, UP x3 = 13 moves -> level complete (human baseline 22).
- K7 TERRAIN: light-grey area = walkable floor; dark area = wall/void.

## r11l, level 1 (screenshot 2)
- K1 CONTROL: by CLICKING (ACTION6 only). The two hollow diamonds are handles; white = selected,
  grey = not selected. Click a handle to select it; click an empty spot to move the selected handle there.
- K2 BODY: the purple disc (magenta centre) always sits at the MIDPOINT of its two handles; light-grey
  dotted lines are tethers drawn from each handle to the disc.
- K3 GOAL: get the purple disc onto the purple dotted ring.
- K4 HAZARD: grey cave rock = walls. If the disc touches rock while moving it bounces back (5 hits = loss).
- K5 HUD: the white column at the far left = moves left (60 max, drains from the top).
- K6 PLAN: 3 clicks: move white handle to (21,31); click grey handle; move it to (21,47) -> disc lands on
  ring, level complete (human baseline 22).
- K7 TERRAIN: black = open space; grey = rock.

## Grading rubric (per experiment, per game)
For each K item: 2 = stated or shown correctly and unmistakably, 1 = partly / only implied,
0 = absent, -1 = states something wrong that would mislead.  Max 14 per game, 28 total.
Plus INTUITION 1-5 (my judgment: after reading only this output, how fast and how confidently would a
player 'get' the game) and COST (image pixels + text tokens).
Caveat: I know both games, so I grade only what the output itself states/shows, not what I can infer.
