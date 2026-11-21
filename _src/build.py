# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml>=6", "jinja2>=3.1", "pillow>=10", "opencv-python-headless>=4.9,<5"]
# ///
"""Build index.html from content.yaml and _src/template.html.

Run from anywhere with:  uv run jpacilo.github.io/_src/build.py
(or `make build` / `make site` from the workspace root).
"""

import hashlib
import html
import re
from datetime import date
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup
from PIL import Image, ImageOps

SITE = Path(__file__).resolve().parent.parent
CONTENT = SITE / "content.yaml"
TEMPLATE_DIR = SITE / "_src"
OUTPUT = SITE / "index.html"
PHOTO_OUT = "img/photo.jpg"   # generated from content.yaml's `photo`; the "photo" link opens it
PHOTO_SIZE = 600
RASTER = {".jpg", ".jpeg", ".png", ".webp"}

LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
BOLD = re.compile(r"\*\*(.+?)\*\*")
APOSTROPHE = re.compile(r"(?<=\w)'(?=\w)")
ENTITY = re.compile(r"&amp;(nbsp|ndash|mdash|middot|rsquo|lsquo|#\d+);")


def md(text) -> Markup:
    """Tiny inline formatter: **bold**, [text](url), curly apostrophes, &nbsp;.

    Everything else is HTML-escaped, so a stray < or & in the YAML can't break the page.
    """
    if text is None:
        return Markup("")
    s = html.escape(str(text).strip(), quote=False)
    s = ENTITY.sub(r"&\1;", s)                     # let &nbsp; and friends through
    s = APOSTROPHE.sub("’", s)
    s = LINK.sub(lambda m: f'<a href="{html.escape(m.group(2))}">{m.group(1)}</a>', s)
    s = BOLD.sub(r"<b>\1</b>", s)
    return Markup(s)


def find_face(im: Image.Image) -> tuple[float, float, float] | None:
    """Return (centre x, centre y, size) of the most likely face, in pixels, or None.

    Two OpenCV face detectors propose candidates; each is scored by detector confidence
    times the share of skin-toned pixels in its middle, which weeds out windows,
    leaves and bike parts that merely look face-shaped.
    """
    import cv2
    import numpy as np

    W, H = im.size
    scale = min(1.0, 1600 / max(W, H))   # detect on a smaller copy; it's faster and enough
    small = np.array(im.resize((round(W * scale), round(H * scale))))
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
    ycc = cv2.cvtColor(small, cv2.COLOR_RGB2YCrCb)

    best = None
    for name in ("haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_default.xml"):
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        rects, _, weights = detector.detectMultiScale3(
            gray, scaleFactor=1.05, minNeighbors=5, minSize=(16, 16), outputRejectLevels=True
        )
        for (x, y, w, h), weight in zip(rects, np.ravel(weights) if len(rects) else []):
            middle = ycc[y + h // 5 : y + 4 * h // 5, x + w // 5 : x + 4 * w // 5]
            cr, cb = middle[..., 1], middle[..., 2]
            skin = float(((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).mean())
            score = float(weight) * skin
            if skin >= 0.4 and (best is None or score > best[0]):
                best = (score, (x + w / 2) / scale, (y + h / 2) / scale, h / scale)
    return best[1:] if best else None


def smart_square(im: Image.Image, zoom: float, face_y: float) -> tuple[Image.Image, str]:
    """Square crop framed on the face: `zoom` face-heights wide, face `face_y` from the top."""
    W, H = im.size
    face = find_face(im)
    if not face:
        side = min(W, H)
        box = ((W - side) / 2, (H - side) * 0.4, side)
        note = "no face found, used a centred crop (set photo_focus to steer it)"
    else:
        cx, cy, size = face
        side = min(size * zoom, W, H)
        left = min(max(cx - side / 2, 0), W - side)
        top = min(max(cy - face_y * side, 0), H - side)
        box = (left, top, side)
        note = f"face found, framed at {zoom}x its height"
    left, top, side = box
    return im.crop((round(left), round(top), round(left + side), round(top + side))), note


def prepare_photo(src: str | None, focus: str | None, zoom, face_y=None) -> str | None:
    """Turn any JPG/PNG/WebP into a 600px square at img/photo.jpg, metadata stripped.

    focus "auto" finds the face and frames it (`zoom` face-heights wide; bigger shows
    more of the scene). Otherwise focus is "X% Y%" like CSS object-position: the
    largest square that fits, slid to that point. SVGs pass through as-is.
    """
    if not src or Path(src).suffix.lower() not in RASTER or src == PHOTO_OUT:
        return src
    path = SITE / src
    if not path.exists():
        raise SystemExit(
            f"photo not found: {src}\n"
            f"Paths are relative to {SITE.name}/, so a photo in the workspace img folder is ../img/<name>."
        )
    if path.resolve().is_relative_to(SITE):
        print(
            f"WARNING: the original photo ({src}) is inside {SITE.name}/, so it would be published "
            "with its metadata (phone and camera photos often include GPS location). "
            "Move it to the workspace img folder and use photo: ../img/<name>."
        )
    focus = str(focus or "50% 50%").strip()
    zoom = float(zoom or 4.5)
    face_y = float(str(face_y or "40%").strip().rstrip("%")) / 100

    # Skip the work when nothing changed: the output remembers what it was made from.
    stat = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{focus}|{zoom}|{face_y}|{PHOTO_SIZE}|v3".encode()).hexdigest()[:16]
    out = SITE / PHOTO_OUT
    if out.exists():
        with Image.open(out) as done:
            if done.info.get("comment", b"") == key.encode():
                return PHOTO_OUT

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")   # phone photos: honour rotation
        if focus.lower() == "auto":
            im, note = smart_square(im, zoom, face_y)
            print(f"Photo: {note}")
        else:
            try:
                fx, fy = (float(v.strip().rstrip("%")) / 100 for v in focus.split())
            except ValueError:
                raise SystemExit(f'photo_focus should be auto or look like "50% 30%", got {focus!r}')
            im = ImageOps.fit(im, (min(im.size),) * 2, centering=(fx, fy))
        im = im.resize((PHOTO_SIZE, PHOTO_SIZE), Image.LANCZOS)
        # Only an opaque build key is stored; no EXIF, so no GPS or camera data.
        im.save(out, "JPEG", quality=85, optimize=True, progressive=True, comment=key)
    return PHOTO_OUT


def paragraphs(text) -> list[str]:
    """Split a YAML text block into paragraphs.

    In a `>-` block, a blank line between lines becomes a line break, so each line of
    the folded result is one paragraph.
    """
    if not text:
        return []
    return [p.strip() for p in str(text).split("\n") if p.strip()]


def main() -> int:
    try:
        data = yaml.safe_load(CONTENT.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise SystemExit(
            f"content.yaml has a formatting error{where}: {getattr(e, 'problem', e)}.\n"
            "Check the indentation there: keys of the same item must line up, using spaces, not tabs."
        )
    data["site"] = data.get("site") or {}
    data["site"].setdefault("title", data.get("name", ""))
    data["site"]["description"] = data["site"].get("description") or ""
    data["links"] = data.get("links") or []
    data["news"] = data.get("news") or []
    data["photo"] = prepare_photo(
        data.get("photo"), data.get("photo_focus"), data.get("photo_zoom"), data.get("photo_face_y")
    )

    # Optional fields may be left out; missing keys render as empty rather than erroring.
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=True,
        keep_trailing_newline=True,
    )
    env.filters["md"] = md
    env.filters["paragraphs"] = paragraphs
    page = env.get_template("template.html").render(
        **data,
        updated=date.today().strftime("%B %Y"),
        # style.css?v=<hash>: the URL changes whenever the CSS does, so browsers (and
        # GitHub Pages' cache) never serve an old copy.
        css_version=hashlib.sha1((SITE / "style.css").read_bytes()).hexdigest()[:8],
    )
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"Built {OUTPUT.relative_to(SITE.parent)} from {CONTENT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
