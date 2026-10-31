"""OCR the frame and locate every occurrence of every rule's text."""
import re
from difflib import SequenceMatcher

import cv2
import numpy as np

from .fonts import candidate_fonts, load_font
from .rules import loose_char

_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1"})
_NUMBERISH = re.compile(r"[\d+()\-\s.OoDIl|]{6,}")
_COUNTRY_PREFIX = re.compile(r"(\+\s*\d{1,3}|00\s*\d{1,3})[\s\-.()]*$")


def default_ocr():
    """RapidOCR engine as a callable: img -> ([(points, text, score), ...], _).
    Uses `rapidocr` 3.x (all current Pythons); falls back to the older
    `rapidocr_onnxruntime` package if that is what is installed."""
    try:
        import logging

        from rapidocr import RapidOCR
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        try:
            engine = RapidOCR(params={"Global.log_level": "warning"})
        except Exception:     # older 2.x/3.x without that setting
            engine = RapidOCR()

        def run(img):
            r = engine(img)
            if r is None or getattr(r, "boxes", None) is None:
                return [], None
            return [(b.tolist(), t, s) for b, t, s in zip(r.boxes, r.txts, r.scores)], None
        return run
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()


def _joined(a, b):
    """True if a and b look like parts of one word. OCR often drops spaces
    ("ENDS31DEC"), so a letter/digit change still counts as a word break."""
    return (a.isalpha() and b.isalpha()) or (a.isdigit() and b.isdigit())


class Finder:
    """Find rule text in images. `ocr` can be injected (tests); default is RapidOCR."""

    def __init__(self, rules, fuzz=0.85, ocr=None, max_side=1280):
        self.ocr = ocr or default_ocr()
        self.set_rules(rules)
        self.fuzz = fuzz
        self.max_side = max_side
        self._measure = load_font(candidate_fonts()[0], 40)

    def set_rules(self, rules):
        # Longest first, so "Al Quoz Branch" wins over "Al Quoz" on the same text
        self.rules = sorted(rules, key=lambda r: len(r.targets[0]), reverse=True)

    # ------------------------------------------------------------------ OCR
    def ocr_boxes(self, img_bgr):
        h, w = img_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1 else img_bgr
        result, _ = self.ocr(small)
        boxes = []
        for pts, text, _score in result or []:
            pts = np.array(pts, dtype=float) / scale
            (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
            boxes.append({"box": (x0, y0, x1, y1), "text": str(text)})
        return boxes

    @staticmethod
    def _lines(boxes):
        """Group OCR boxes into text lines, left to right."""
        lines = []
        for b in sorted(boxes, key=lambda b: (b["box"][1] + b["box"][3]) / 2):
            cy, h = (b["box"][1] + b["box"][3]) / 2, b["box"][3] - b["box"][1]
            for line in lines:
                lcy = np.mean([(x["box"][1] + x["box"][3]) / 2 for x in line])
                lh = np.mean([x["box"][3] - x["box"][1] for x in line])
                if abs(cy - lcy) < 0.5 * max(h, lh):
                    line.append(b)
                    break
            else:
                lines.append([b])
        return [sorted(l, key=lambda b: b["box"][0]) for l in lines]

    def _char_x(self, box, text, idx):
        full = self._measure.getlength(text) or 1
        return box[0] + (box[2] - box[0]) * self._measure.getlength(text[:idx]) / full

    # ------------------------------------------------------------------ matching
    @staticmethod
    def _phone_spans(text, rule):
        digits, pos = [], []
        for m in _NUMBERISH.finditer(text):
            run = m.group()
            if sum(c.isdigit() for c in run) < 4:
                continue
            for k, ch in enumerate(run.translate(_DIGIT_FIX)):
                if ch.isdigit():
                    digits.append(ch)
                    pos.append(m.start() + k)
        dstr, spans = "".join(digits), []
        for target in rule.targets:
            start = 0
            while (i := dstr.find(target, start)) >= 0:
                c0, c1 = pos[i], pos[i + len(target) - 1]
                if m := _COUNTRY_PREFIX.search(text[:c0]):
                    c0 = m.start()
                elif c0 > 0 and text[c0 - 1] == "+":
                    c0 -= 1
                spans.append((c0, c1, 1.0))
                start = i + len(target)
            if spans:
                break
        return spans

    def _text_spans(self, text, rule):
        target = rule.targets[0]
        norm, npos = [], []
        for i, ch in enumerate(text):
            if ch.isalnum():
                norm.append(loose_char(ch))
                npos.append(i)
        nstr, L = "".join(norm), len(target)

        found = [(m.start(), L, 1.0) for m in re.finditer(re.escape(target), nstr)]
        if not found and L >= 5 and self.fuzz < 1:
            cands = []
            for i in range(len(nstr)):
                for L2 in (L - 1, L, L + 1):
                    if i + L2 <= len(nstr):
                        r = SequenceMatcher(None, target, nstr[i:i + L2]).ratio()
                        if r >= self.fuzz:
                            cands.append((r, i, L2))
            taken = set()
            for r, i, L2 in sorted(cands, reverse=True):
                if not taken & set(range(i, i + L2)):
                    taken |= set(range(i, i + L2))
                    found.append((i, L2, r))

        spans = []
        lead = not rule.find[0].isalnum()
        trail = not rule.find[-1].isalnum()
        for i, L2, score in found:
            c0, c1 = npos[i], npos[i + L2 - 1]
            while trail and c1 + 1 < len(text) and not text[c1 + 1].isalnum() and not text[c1 + 1].isspace():
                c1 += 1
            while lead and c0 > 0 and not text[c0 - 1].isalnum() and not text[c0 - 1].isspace():
                c0 -= 1
            before = text[c0 - 1] if c0 > 0 else " "
            after = text[c1 + 1] if c1 + 1 < len(text) else " "
            if L < 6 and (_joined(before, text[c0]) or _joined(text[c1], after)):
                continue  # short word inside a longer word - not a real match
            spans.append((c0, c1, score))
        return spans

    def find(self, img_bgr):
        """Every occurrence of every rule. Each hit: box, rule, old_text, score,
        alone (only thing on its line), followed (more text after it on the line)."""
        hits = []
        lines = self._lines(self.ocr_boxes(img_bgr))
        line_boxes = [(min(b["box"][0] for b in l), min(b["box"][1] for b in l),
                       max(b["box"][2] for b in l), max(b["box"][3] for b in l)) for l in lines]
        line_spans = [(b[0], b[2]) for b in line_boxes]
        for li, line in enumerate(lines):
            chars = []  # (char, box index, index inside box text)
            for bi, b in enumerate(line):
                if chars:
                    chars.append((" ", -1, -1))
                chars += [(ch, bi, ci) for ci, ch in enumerate(b["text"])]
            text = "".join(c[0] for c in chars)
            claimed = set()
            for rule in self.rules:
                spans = self._phone_spans(text, rule) if rule.kind == "phone" else self._text_spans(text, rule)
                for c0, c1, score in spans:
                    rng = set(range(c0, c1 + 1))
                    if rng & claimed:
                        continue
                    claimed |= rng
                    _, b0, k0 = chars[c0]
                    _, b1, k1 = chars[c1]
                    if b0 < 0 or b1 < 0:
                        continue
                    used = {bi for _, bi, _ in chars[c0:c1 + 1] if bi >= 0}
                    hits.append({
                        "box": (self._char_x(line[b0]["box"], line[b0]["text"], k0),
                                min(line[bi]["box"][1] for bi in used),
                                self._char_x(line[b1]["box"], line[b1]["text"], k1 + 1),
                                max(line[bi]["box"][3] for bi in used)),
                        "line_box": (min(b["box"][0] for b in line), min(b["box"][1] for b in line),
                                     max(b["box"][2] for b in line), max(b["box"][3] for b in line)),
                        "other_lines": line_spans[:li] + line_spans[li + 1:],
                        "other_boxes": line_boxes[:li] + line_boxes[li + 1:],
                        "rule": rule,
                        "old_text": text[c0:c1 + 1],
                        "score": score,
                        "alone": not re.search(r"\w", text[:c0] + text[c1 + 1:]),
                        "followed": bool(re.search(r"\w", text[c1 + 1:])),
                    })
        return hits
