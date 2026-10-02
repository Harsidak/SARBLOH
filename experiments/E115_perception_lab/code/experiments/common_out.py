"""Small helpers shared by experiment renderers: location words, text panels, image stacking."""
from __future__ import annotations

from PIL import Image, ImageDraw

from scene import N, NAMES, Obj, Scene, font


def zone(r: float, c: float) -> str:
    v = "top" if r < N / 3 else ("bottom" if r >= 2 * N / 3 else "middle")
    h = "left" if c < N / 3 else ("right" if c >= 2 * N / 3 else "centre")
    return "centre" if (v, h) == ("middle", "centre") else f"{v}-{h}" if v != "middle" else f"middle-{h}"


def is_hud(o: Obj, sc: Scene) -> bool:
    if o.kind == "bar":
        return True
    if o.panel is not None:
        p = sc.panels[o.panel]
        if p.r0 <= 2 or p.c0 <= 2 or p.r1 >= N - 3 or p.c1 >= N - 3:
            return True
    return o.r0 <= 1 or o.c0 <= 1 or o.r1 >= N - 2 or o.c1 >= N - 2


def text_block(lines: list[str], width: int, size: int = 15, pad: int = 10, bold_first: bool = True) -> Image.Image:
    f, fb = font(size), font(size + 1, bold=True)
    # wrap
    wrapped = []
    for i, ln in enumerate(lines):
        words, cur = ln.split(" "), ""
        ff = fb if (i == 0 and bold_first) else f
        for w in words:
            t = (cur + " " + w).strip()
            if ff.getlength(t) > width - 2 * pad and cur:
                wrapped.append((cur, i == 0 and bold_first)); cur = "   " + w
            else:
                cur = t
        wrapped.append((cur, i == 0 and bold_first))
    lh = size + 6
    img = Image.new("RGB", (width, pad * 2 + lh * len(wrapped)), (250, 250, 247))
    d = ImageDraw.Draw(img)
    for k, (t, b) in enumerate(wrapped):
        d.text((pad, pad + k * lh), t, fill=(20, 20, 20), font=fb if b else f)
    return img


def hstack(imgs: list[Image.Image], gap: int = 12, bg=(250, 250, 247)) -> Image.Image:
    W = sum(i.width for i in imgs) + gap * (len(imgs) - 1); H = max(i.height for i in imgs)
    out = Image.new("RGB", (W, H), bg); x = 0
    for i in imgs:
        out.paste(i, (x, 0)); x += i.width + gap
    return out


def vstack(imgs: list[Image.Image], gap: int = 12, bg=(250, 250, 247)) -> Image.Image:
    W = max(i.width for i in imgs); H = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (W, H), bg); y = 0
    for i in imgs:
        out.paste(i, (0, y)); y += i.height + gap
    return out


def titled(img: Image.Image, title: str, size: int = 16) -> Image.Image:
    f = font(size, bold=True)
    out = Image.new("RGB", (img.width, img.height + size + 12), (250, 250, 247))
    ImageDraw.Draw(out).text((4, 4), title, fill=(20, 20, 20), font=f)
    out.paste(img, (0, size + 12))
    return out


def box(d: ImageDraw.ImageDraw, o: Obj, cell: int, color, width: int = 2, pad: int = 1):
    d.rectangle([o.c0 * cell - pad, o.r0 * cell - pad, (o.c1 + 1) * cell + pad - 1, (o.r1 + 1) * cell + pad - 1],
                outline=color, width=width)


def colour_list(o: Obj) -> str:
    return "/".join(NAMES[c] for c, _ in o.colors.most_common())
