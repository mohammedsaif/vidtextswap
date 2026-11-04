"""Measure how the original text looks and redraw the replacement the same way."""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from .fonts import load_font

_MEASURE = ImageDraw.Draw(Image.new("RGBA", (1, 1)))


def _ring(frame, X0, Y0, X1, Y1, r, exclude=()):
    """Pixels in a thin ring just outside a region (= local background), leaving
    out anything inside `exclude` boxes (other text lines)."""
    H, W = frame.shape[:2]
    a0, b0, a1, b1 = max(0, X0 - r), max(0, Y0 - r), min(W, X1 + r), min(H, Y1 + r)
    sel = np.ones((b1 - b0, a1 - a0), bool)
    sel[max(0, Y0 - b0):max(0, Y1 - b0), max(0, X0 - a0):max(0, X1 - a0)] = False
    full = sel.sum()
    for ex0, ey0, ex1, ey1 in exclude:
        sel[max(0, int(ey0) - b0):max(0, int(ey1) - b0), max(0, int(ex0) - a0):max(0, int(ex1) - a0)] = False
    if sel.sum() < 0.25 * full:     # nearly all excluded: better a noisy ring than none
        sel[:] = True
        sel[max(0, Y0 - b0):max(0, Y1 - b0), max(0, X0 - a0):max(0, X1 - a0)] = False
    ring = frame[b0:b1, a0:a1][sel].astype(np.float32)
    return ring if len(ring) else frame[Y0:Y1, X0:X1].reshape(-1, 3).astype(np.float32)


class Ink:
    """Decides which pixels are text.

    Calm background (one colour, light texture): text = clearly different from
    the background colour. Busy background (tiles, photos, several colours):
    text = the colours that fill the text box but are rare around it."""

    def __init__(self, region, ring):
        self.bg = np.median(ring, axis=0)
        # Background variation from the calmer 75% of the ring, so text from a
        # neighbouring line that the ring happens to touch doesn't raise the bar
        self.thr = max(35.0, float(np.percentile(np.linalg.norm(ring - self.bg, axis=1), 75)) + 15)
        # High-contrast text (e.g. white with a black outline) gets a stricter cut-off,
        # so darker/lighter patches of a textured background aren't taken for text
        contrast = float(np.percentile(np.linalg.norm(region.reshape(-1, 3).astype(np.float32) - self.bg, axis=1), 98))
        self.thr = max(self.thr, 0.3 * contrast)
        self.colours = None
        mad = float(np.median(np.abs(ring - self.bg), axis=0).mean())
        if mad > 9:                                     # busy background
            px = region.reshape(-1, 3).astype(np.float32)
            if len(px) > 6000:
                px = px[np.random.default_rng(0).choice(len(px), 6000, replace=False)]
            k = 6
            if len(px) > 4 * k:
                _, labels, centres = cv2.kmeans(px, k, None,
                                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1),
                                                2, cv2.KMEANS_PP_CENTERS)
                labels = labels.ravel()
                keep = []
                for i, c in enumerate(centres):
                    in_ring = float(np.mean(np.linalg.norm(ring - c, axis=1) < 40))
                    in_box = float(np.mean(labels == i))
                    if in_ring < 0.08 and in_box > 0.03:
                        keep.append(c)
                if keep:
                    self.colours = np.array(keep, np.float32)

    def __call__(self, pixels):
        p = pixels.astype(np.float32)
        if self.colours is None:
            return np.linalg.norm(p - self.bg, axis=-1) > self.thr
        d = np.min(np.stack([np.linalg.norm(p - c, axis=-1) for c in self.colours]), axis=0)
        return d < 45


def _text_mask(region, ring, ink=None):
    ink = ink or Ink(region, ring)
    return ink(region), ink.bg


def _render(text, font, fill, sw, stroke):
    """Tightly cropped RGBA image of the text."""
    l, t, r, b = _MEASURE.textbbox((0, 0), text, font=font, stroke_width=sw)
    img = Image.new("RGBA", (max(1, r - l), max(1, b - t)), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((-l, -t), text, font=font, fill=fill, stroke_width=sw, stroke_fill=stroke)
    return img, t


def _ink_band(text_mask, y_centre):
    """Contiguous rows with ink around the text's centre line (ignores other lines)."""
    has = text_mask.mean(axis=1) > 0.02
    c = int(np.clip(y_centre, 0, len(has) - 1))
    if not has[c]:
        on = np.where(has)[0]
        if not len(on):
            return None
        c = on[np.argmin(np.abs(on - c))]
    top = bot = c
    while top > 0 and has[top - 1]:
        top -= 1
    while bot < len(has) - 1 and has[bot + 1]:
        bot += 1
    return (top, bot) if bot - top >= 2 else None


def _thickness(fill_mask):
    """Average stroke thickness of glyphs in pixels. Uses the distance transform
    (for a stroke of width w the mean distance-to-edge is about w/4), which is
    not thrown off by jagged, compressed edges the way a perimeter would be."""
    m = fill_mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    if int(m.sum()) < 20:
        return None
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return 4 * float(dt[m > 0].mean())


def _ink_runs(frame, ink, y0, y1, x0, x1, ink_h):
    """Words on one text line, from pixels: list of (start, end, solid) absolute
    columns. Letters closer than ~0.18 x text height are joined into one word.
    `solid` marks blocks that are mostly filled - a box edge or background change,
    not text."""
    H, W = frame.shape[:2]
    x0, x1, y0, y1 = max(0, int(x0)), min(W, int(x1)), max(0, int(y0)), min(H, int(y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return []
    dist = ink(frame[y0:y1, x0:x1])
    on = dist.mean(axis=0) > 0.03
    gap_min = max(2, int(0.18 * ink_h))
    runs, start, gap = [], None, 0
    for i, v in enumerate(on):
        if v:
            if start is None:
                start = i
            gap, end = 0, i
        elif start is not None:
            gap += 1
            if gap >= gap_min:
                runs.append((start, end + 1))
                start, gap = None, 0
    if start is not None:
        runs.append((start, end + 1))
    return [(x0 + a, x0 + e, dist[:, a:e].mean() > 0.85) for a, e in runs]


def _colours(region, text_mask, bg):
    """Fill colour, and outline colour if the text has a border (else None)."""
    px = region[text_mask].astype(np.float32)
    fill = np.array([0, 0, 0], np.float32) if bg.mean() > 128 else np.array([255] * 3, np.float32)
    if len(px) <= 30:
        return fill, None, 0.0
    fill = np.median(px, axis=0)
    _, labels, centers = cv2.kmeans(px, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1),
                                    3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    if np.linalg.norm(centers[0] - centers[1]) <= 80 or min(np.bincount(labels, minlength=2)) <= 0.08 * len(labels):
        return fill, None, 0.0
    lab = np.full(text_mask.shape, -1, np.int32)
    lab[text_mask] = labels
    near_bg = cv2.dilate((~text_mask).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    o = int(np.argmax([np.mean(near_bg[lab == k]) for k in (0, 1)]))   # outline touches bg
    stroke, fill = centers[o], centers[1 - o]
    # Anti-aliased edges are a blend of fill and background; a real outline is not.
    seg = fill - bg
    t = float(np.clip(np.dot(stroke - bg, seg) / max(1e-6, np.dot(seg, seg)), 0, 1))
    if np.linalg.norm(stroke - (bg + t * seg)) > 45 and np.linalg.norm(stroke - bg) > 60:
        return fill, stroke, float(np.mean(labels == o))
    return fill, None, 0.0


def _fit_font(path, sample, ink_h, has_stroke, outline_ratio):
    """Size + outline width so `sample` renders as tall as the original ink."""
    probe = load_font(path, 100)
    _, t, _, b = probe.getbbox(sample)
    h100 = max(1, b - t)
    best, best_err = (load_font(path, 100 * ink_h / h100), 0), None
    if has_stroke:
        for sw in range(1, max(2, int(ink_h * 0.25)) + 1):
            f = load_font(path, 100 * (ink_h - 2 * sw) / h100)
            img, _ = _render(sample, f, (255, 255, 255, 255), sw, (0, 0, 0, 255))
            a = np.array(img)
            ink = a[..., 3] > 128
            if ink.any():
                err = abs(np.mean(a[..., 0][ink] < 128) - outline_ratio)
                if best_err is None or err < best_err:
                    best, best_err = (f, sw), err
    font, sw = best
    for _ in range(3):   # exact height, outline included
        _, t, _, b = _MEASURE.textbbox((0, 0), sample, font=font, stroke_width=sw)
        if b - t <= 2 * sw or abs((b - t) - ink_h) < 1:
            break
        font = load_font(path, font.size * (ink_h - 2 * sw) / (b - t - 2 * sw))
    return font, sw


def _is_centred(line, others, left_margin, right_margin, ink_h, W):
    """Is this text line centre-aligned? Other lines in the frame vote first
    (same centre vs same left edge); otherwise compare the margins to the
    surrounding box or frame edges."""
    tol = 0.015 * W
    c = (line[0] + line[2]) / 2
    centre_votes = sum(abs((a + b) / 2 - c) < tol for a, b in others)
    left_votes = sum(abs(a - line[0]) < tol for a, b in others)
    if centre_votes != left_votes:
        return centre_votes > left_votes
    return abs(left_margin - right_margin) < max(0.5 * ink_h, 0.03 * W)


class _NoReflow(Exception):
    pass


def analyse(frame, hit, new_text, fonts, reflow=True):
    """Plan the edit for one occurrence: what to erase and what to draw.
    Returns a dict consumed by apply()."""
    if reflow:
        try:
            return _analyse(frame, hit, new_text, fonts, True)
        except _NoReflow:
            pass
    return _analyse(frame, hit, new_text, fonts, False)


def _analyse(frame, hit, new_text, fonts, reflow):
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = hit["box"]
    bh = y1 - y0
    pad = max(4, int(bh * 0.2))
    X0, Y0, X1, Y1 = max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(W, int(x1 + pad)), min(H, int(y1 + pad))
    r = max(2, pad // 2)
    others = hit.get("other_boxes", ())
    ring = _ring(frame, X0, Y0, X1, Y1, r, others)
    region = frame[Y0:Y1, X0:X1]
    ink = Ink(region, ring)
    mask, bg = ink(region), ink.bg

    # Real extent of the old text's ink
    oc0, oc1 = max(0, int(x0 - X0 - pad / 2)), min(mask.shape[1], int(x1 - X0 + pad / 2) + 1)
    band = _ink_band(mask[:, oc0:oc1], (y0 + y1) / 2 - Y0)
    if band:
        ink_top, ink_h = Y0 + band[0], band[1] - band[0] + 1
        cols = np.where(mask[band[0]:band[1] + 1, oc0:oc1].any(axis=0))[0]
        ink_l, ink_r = X0 + oc0 + cols[0], X0 + oc0 + cols[-1] + 1
        density = float(mask[band[0]:band[1] + 1, ink_l - X0:ink_r - X0].mean())
    else:
        ink_top, ink_h, ink_l, ink_r, density = y0 + bh * 0.15, bh * 0.7, x0, x1, None
    # OCR may glue words ("WhatsApp:+971..."), so snap edges to real gaps in the pixels
    runs = _ink_runs(frame, ink, ink_top, ink_top + ink_h, x0 - 1.5 * bh, x1 + 1.5 * bh, ink_h)
    tol = 0.6 * ink_h
    starts = [a for a, _, solid in runs if not solid and abs(a - ink_l) < tol]
    ends = [e for _, e, solid in runs if not solid and abs(e - ink_r) < tol]
    if starts:
        ink_l = min(starts, key=lambda v: abs(v - ink_l))
    if ends:
        ink_r = min(ends, key=lambda v: abs(v - ink_r))
    ink_w = max(1, ink_r - ink_l)

    fill, stroke, outline_ratio = _colours(region[:, oc0:oc1], mask[:, oc0:oc1], bg)
    sample = hit["old_text"].strip() or new_text

    # Letter thickness of the original fill (outline excluded) - tells bold from regular
    thick = None
    if band:
        rows = slice(band[0], band[1] + 1)
        cols_ = slice(max(0, ink_l - X0), max(1, ink_r - X0))
        fm = mask[rows, cols_]
        if stroke is not None:
            px = region[rows, cols_].astype(np.float32)
            fm = fm & (np.linalg.norm(px - fill, axis=2) < np.linalg.norm(px - stroke, axis=2))
        t = _thickness(fm)
        thick = t / ink_h if t else None

    # Pick the font (bold/regular/user font) whose shape matches the original best
    best = None
    for path in fonts:
        font, sw = _fit_font(path, sample, ink_h, stroke is not None, outline_ratio)
        img, _ = _render(sample, font, (255, 255, 255, 255), sw, (0, 0, 0, 255))
        a_ = np.array(img)
        width_ratio = ink_w / img.width
        score = 0.15 * abs(np.log(width_ratio))
        t = _thickness((a_[..., 3] > 128) & (a_[..., 0] > 128))
        if thick and t:
            score += abs(np.log(thick / (t / img.height)))
        elif density is not None:
            score += abs(float((a_[..., 3] > 128).mean()) - density)
        if best is None or score < best[0]:
            best = (score, path, font, sw, width_ratio)
    _, path, font, sw, width_ratio = best

    to_rgba = lambda c: tuple(int(v) for v in np.clip(c, 0, 255)[::-1]) + (255,)
    stroke_rgba = to_rgba(stroke) if stroke is not None else None
    _, sample_top = _render(sample, font, (0, 0, 0, 255), sw, stroke_rgba)

    # Width: copy the original's condensed/expanded look
    hx = float(np.clip(width_ratio, 0.6, 1.15))
    tight, new_top = _render(new_text, font, to_rgba(fill), sw, stroke_rgba)
    new_w = tight.width * hx

    # Reflow: slide the rest of the line so there is no gap / overlap
    moves, nx = [], (ink_l + ink_r) / 2 - new_w / 2 if hit.get("alone") else ink_l
    line = hit.get("line_box")
    if reflow and line is not None and not hit.get("alone") and abs(new_w - ink_w) > 0.15 * ink_h:
        delta = new_w - ink_w
        # Real extent of the line: words reachable from the old text with normal spacing
        lr = _ink_runs(frame, ink, ink_top, ink_top + ink_h, 0, W, ink_h)
        lx0, lx1 = ink_l, ink_r
        left_wall, right_wall = 0, W
        for a_, e_, solid in sorted(lr, key=lambda t: -t[0]):    # walk left
            if e_ <= lx0:
                if solid or lx0 - e_ >= 1.5 * ink_h:
                    left_wall = e_ if solid else left_wall
                    if solid:
                        break
                    continue
                lx0 = a_
        for a_, e_, solid in sorted(lr):                          # walk right
            if a_ >= lx1:
                if solid or a_ - lx1 >= 1.5 * ink_h:
                    right_wall = a_ if solid else right_wall
                    if solid:
                        break
                    continue
                lx1 = e_
        centred = _is_centred(line, hit.get("other_lines", []), lx0 - left_wall, right_wall - lx1, ink_h, W)
        d_pre, d_new, d_post = (-delta / 2, -delta / 2, delta / 2) if centred else (0, 0, delta)
        if ink_l - lx0 > 0.3 * ink_h and d_pre:
            moves.append((lx0 - 3, ink_l - 3, d_pre))
        if lx1 - ink_r > 0.3 * ink_h and d_post:
            moves.append((ink_r + 3, lx1 + 3, d_post))
        lo = min([ink_l + d_new] + [c0 + d for c0, _, d in moves])
        hi = max([ink_l + d_new + new_w] + [c1 + d for _, c1, d in moves])
        if lo >= 2 and hi <= W - 2:
            nx = ink_l + d_new
        else:
            moves = []          # no room in the frame: squeeze instead

    if not moves:
        max_w = ink_w * 1.04 if hit.get("followed") else min(ink_w * 1.6, W - 2 * pad)
        for _ in range(20):
            if tight.width * hx <= max_w:
                break
            need = max_w / tight.width
            if need >= 0.6:
                hx = need
                break
            hx = 0.6
            font = load_font(path, font.size * 0.93)   # still too long: smaller text
            tight, new_top = _render(new_text, font, to_rgba(fill), sw, stroke_rgba)
            _, sample_top = _render(sample, font, (0, 0, 0, 255), sw, stroke_rgba)
        if hit.get("alone"):
            nx = (ink_l + ink_r) / 2 - tight.width * hx / 2
    if abs(hx - 1) > 0.01:
        tight = tight.resize((max(1, round(tight.width * hx)), tight.height), Image.LANCZOS)
    ny = ink_top - sample_top + new_top   # same baseline as the original

    # Rows of the whole line (for moving neighbouring words)
    ly0, ly1 = int(ink_top - 0.35 * ink_h), int(ink_top + 1.35 * ink_h)
    xs = [nx, nx + tight.width, X0, X1] + [v for c0, c1, d in moves for v in (c0, c1, c0 + d, c1 + d)]
    X0n, X1n = max(0, int(min(xs) - pad)), min(W, int(max(xs) + pad))
    Y0n = max(0, int(min(Y0, ny - pad, ly0 if moves else Y0)))
    Y1n = min(H, int(max(Y1, ny + tight.height + pad, ly1 if moves else Y1)))
    ring = _ring(frame, X0n, Y0n, X1n, Y1n, r, others)
    mask = ink(frame[Y0n:Y1n, X0n:X1n])
    bg = np.median(ring, axis=0)

    # Erase only the old text's own pixels: its columns, its rows
    erase = np.zeros(mask.shape, np.uint8)
    ex0, ex1 = int(ink_l - X0n - 2), int(ink_r - X0n + 3)
    ey0, ey1 = int(ink_top - Y0n - 3), int(ink_top + ink_h - Y0n + 3)
    erase[max(0, ey0):max(0, ey1), max(0, ex0):max(0, ex1)] = mask[max(0, ey0):max(0, ey1), max(0, ex0):max(0, ex1)]
    erase = cv2.dilate(erase * 255, np.ones((3, 3), np.uint8), iterations=2 + int(sw > 0))
    keep = np.zeros_like(erase)
    keep[max(0, ey0):max(0, ey1), max(0, ex0 - 3):max(0, ex1 + 3)] = 1
    erase *= keep

    # Neighbouring words: lift their pixels, erase them, put them back shifted
    layer_np = np.zeros((Y1n - Y0n, X1n - X0n, 4), np.float32)
    for c0, c1, d in moves:
        r0, r1 = max(0, ly0 - Y0n), min(mask.shape[0], ly1 - Y0n)
        a0, a1 = max(0, int(c0) - X0n), min(mask.shape[1], int(c1) - X0n)
        if r1 <= r0 or a1 <= a0:
            continue
        m = cv2.dilate(mask[r0:r1, a0:a1].astype(np.uint8), np.ones((3, 3), np.uint8))
        if m.mean() > 0.6:
            raise _NoReflow
        erase[r0:r1, a0:a1] = np.maximum(erase[r0:r1, a0:a1],
                                         cv2.dilate(m * 255, np.ones((3, 3), np.uint8), iterations=1 + int(sw > 0)))
        soft = cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0)
        src = cv2.cvtColor(frame[Y0n + r0:Y0n + r1, X0n + a0:X0n + a1], cv2.COLOR_BGR2RGB).astype(np.float32)
        s = int(round(d))
        b0, b1 = max(0, a0 + s), min(layer_np.shape[1], a1 + s)
        if b1 <= b0:
            continue
        src, soft = src[:, b0 - (a0 + s):b1 - (a0 + s)], soft[:, b0 - (a0 + s):b1 - (a0 + s)]
        sel = soft > layer_np[r0:r1, b0:b1, 3] / 255.0
        layer_np[r0:r1, b0:b1, :3][sel] = src[sel]
        layer_np[r0:r1, b0:b1, 3][sel] = soft[sel] * 255

    layer = Image.fromarray(layer_np.astype(np.uint8), "RGBA")
    dx, dy = int(round(nx - X0n)), int(round(ny - Y0n))
    tight = tight.crop((max(0, -dx), max(0, -dy), tight.width, tight.height))
    dx, dy = max(0, dx), max(0, dy)
    tight = tight.crop((0, 0, min(tight.width, layer.width - dx), min(tight.height, layer.height - dy)))
    layer.alpha_composite(tight, (dx, dy))
    a = np.array(layer).astype(np.float32)

    desc = f"fill {to_rgba(fill)[:3]}"
    if stroke is not None:
        desc += f", outline {stroke_rgba[:3]} {sw}px"
    if path:
        desc += f", {path.replace(chr(92), '/').rsplit('/', 1)[-1]}"
    if abs(hx - 1) > 0.01:
        desc += f", width {hx:.0%}"
    if moves:
        desc += ", line reflowed"
    return {"X0": X0n, "Y0": Y0n, "X1": X1n, "Y1": Y1n, "erase": erase,
            "rgb": cv2.cvtColor(a[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32),
            "alpha": a[..., 3:4] / 255.0,
            "flat_bg": bg if float(ring.std(axis=0).mean()) < 10 else None,
            "desc": desc}


def _repair(frame, plan):
    """Clean background for the plan's region of this frame (old text removed).

    plan["inpaint"]:
      "fast"  - OpenCV on every frame
      "ai"    - LaMa; reused while the background stays still, re-run when it moves
      "auto"  - LaMa while the background is still (one run per still shot),
                OpenCV on frames where it moves (motion hides small flaws)
    """
    from . import inpaint as ip
    X0, Y0, X1, Y1 = plan["X0"], plan["Y0"], plan["X1"], plan["Y1"]
    crop, erase = frame[Y0:Y1, X0:X1], plan["erase"]
    method = plan.get("inpaint", "fast")
    if method == "fast" or not ip.lama_available(required=method == "ai"):
        return ip.inpaint(crop, erase, "fast")

    keep = erase == 0
    cache = plan.get("_cache")
    if cache is not None:
        diff = np.abs(crop.astype(np.int16) - cache["ref"].astype(np.int16)).mean(axis=2)[keep]
        if diff.size and float(diff.mean()) < 4.0:          # background hasn't moved: reuse
            clean = crop.copy()
            clean[~keep] = cache["clean"][~keep]
            return clean
        if method == "auto":
            return ip.inpaint(crop, erase, "fast")
    full = np.zeros(frame.shape[:2], np.uint8)
    full[Y0:Y1, X0:X1] = erase
    clean = ip.inpaint(frame, full, "ai")[Y0:Y1, X0:X1]    # whole frame = more context
    plan["_cache"] = {"ref": crop.copy(), "clean": clean.copy()}
    return clean


def apply(frame, plan):
    """Erase the old text from THIS frame (moving backgrounds stay natural) and
    draw the new text. Modifies `frame` in place and returns it."""
    X0, Y0, X1, Y1 = plan["X0"], plan["Y0"], plan["X1"], plan["Y1"]
    crop = frame[Y0:Y1, X0:X1]
    if plan["flat_bg"] is not None:
        clean = crop.copy()
        clean[plan["erase"] > 0] = plan["flat_bg"].astype(np.uint8)
    elif plan["erase"].any():
        clean = _repair(frame, plan)
    else:
        clean = crop
    frame[Y0:Y1, X0:X1] = (clean.astype(np.float32) * (1 - plan["alpha"])
                           + plan["rgb"] * plan["alpha"]).astype(np.uint8)
    return frame
