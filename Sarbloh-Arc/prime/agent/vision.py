"""E008 perception, image channel: the newest game frame as a PNG in a user message.

Standard library only (zlib + struct): no Pillow on the offline Kaggle path. The colours are the ARC-AGI-3 palette
(index -> RGB), the same values as Tufa's ``ARC_COLOR_MAP``; the table is a fact of the game, the code is ours.
Nearest-neighbour upscaling: every grid cell becomes an ``upscale`` x ``upscale`` block, so 64x64 -> 256x256 at x4.

    render_png(grid, upscale=4) -> bytes
    data_url(grid, upscale=4)   -> "data:image/png;base64,..."
    image_message(grid, text, step) -> {"role": "user", "content": [text part, image part], "_image_step": step}

Only the newest image stays in the context (``prime.agent.context`` replaces older ones with a text stub).
"""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any

PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102),   # 0 white, 1 light grey, 2 grey, 3 dark grey
    (51, 51, 51), (0, 0, 0), (229, 58, 163), (255, 123, 204),             # 4 charcoal, 5 black, 6 magenta, 7 pink
    (249, 60, 49), (30, 147, 255), (136, 216, 241), (255, 220, 0),        # 8 red, 9 blue, 10 sky blue, 11 yellow
    (255, 133, 27), (146, 18, 49), (79, 204, 48), (163, 86, 214),         # 12 orange, 13 dark red, 14 green, 15 purple
)


def _rows(grid: Any) -> list[list[int]]:
    grid = getattr(grid, "grid", grid)
    return grid.tolist() if hasattr(grid, "tolist") else [list(r) for r in grid]


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def render_png(grid: Any, upscale: int = 4) -> bytes:
    """An 8-bit RGB PNG of the grid, each cell an ``upscale``-pixel square."""
    rows = _rows(grid)
    if not rows or not rows[0]:
        raise ValueError("cannot render an empty grid")
    k = max(1, int(upscale))
    h, w = len(rows), max(len(r) for r in rows)
    colours = [bytes(c) for c in PALETTE]
    raw = bytearray()
    for r in rows:
        line = b"".join(colours[int(v) & 15] * k for v in r) + colours[0] * k * (w - len(r))
        for _ in range(k):
            raw += b"\x00" + line      # filter type 0 (none) per scanline
    header = struct.pack(">IIBBBBB", w * k, h * k, 8, 2, 0, 0, 0)  # 8-bit, colour type 2 (RGB)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + _chunk(b"IEND", b""))


def data_url(grid: Any, upscale: int = 4) -> str:
    return "data:image/png;base64," + base64.b64encode(render_png(grid, upscale)).decode("ascii")


def image_message(grid: Any, text: str, step: int | None, upscale: int = 4) -> dict[str, Any]:
    """A user message: the observation text, then the image of ``grid``. ``_image_step`` marks it for the context
    builder, which keeps only the newest image."""
    return {"role": "user", "_kind": "observation", "_image_step": step,
            "content": [{"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": data_url(grid, upscale)}}]}


def decode_png_size(png: bytes) -> tuple[int, int]:
    """(width, height) from a PNG header: for tests and logs."""
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return struct.unpack(">II", png[16:24])
