"""Replacement rules: what to find and what to put in its place."""
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

# OCR mixes these up; matching treats each group as the same character.
_LOOSE = str.maketrans({"0": "o", "1": "l", "i": "l", "|": "l", "!": "l"})


def only_digits(s):
    return re.sub(r"\D", "", s)


def loose_char(ch):
    return ch.casefold().translate(_LOOSE)


def normalise(text):
    """Letters/digits only, case-folded, OCR look-alikes merged."""
    return "".join(loose_char(c) for c in text if c.isalnum())


def looks_like_phone(text):
    return bool(re.fullmatch(r"[\d\s+\-().]+", text.strip())) and len(only_digits(text)) >= 6


@dataclass
class Rule:
    find: str
    replace: str
    kind: str = "auto"          # "phone", "text" or "auto"
    font: str | None = None     # optional .ttf just for this rule
    targets: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.find, self.replace = self.find.strip(), self.replace.strip()
        if not self.find:
            raise ValueError("'find' text is empty")
        if self.kind not in ("auto", "phone", "text"):
            raise ValueError(f"Unknown rule type '{self.kind}' (use phone, text or auto)")
        if self.kind == "auto":
            self.kind = "phone" if looks_like_phone(self.find) else "text"
        if self.kind == "phone":
            d = only_digits(self.find)
            if len(d) < 6:
                raise ValueError(f"Phone number '{self.find}' looks too short")
            t = {d}
            if d.startswith("0") and len(d) >= 9:
                t.add(d[1:])        # 058... is also written +971 58... / 0097158...
            if d.startswith("00") and len(d) > 10:
                t.add(d[2:])
            if len(d) > 10:
                t.add(d[-9:])       # without the country code
            self.targets = sorted(t, key=len, reverse=True)
        else:
            n = normalise(self.find)
            if len(n) < 2:
                raise ValueError(f"'{self.find}' is too short to search for")
            self.targets = [n]

    def replacement_for(self, found_text, keep_case=True):
        """Text to draw. If the original was ALL CAPS, the replacement follows."""
        if keep_case and self.kind == "text":
            letters = [c for c in found_text if c.isalpha()]
            if letters and all(c.isupper() for c in letters):
                return self.replace.upper()
        return self.replace

    def label(self):
        return f"'{self.find}' -> '{self.replace}'"


def load_rules_csv(path):
    """CSV with columns: find, replace, [type], [font]. Header row required."""
    rules = []
    with open(Path(path), newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            if not row.get("find"):
                continue
            if "replace" not in row:
                raise ValueError(f"{path}: needs 'find' and 'replace' columns")
            try:
                rules.append(Rule(row["find"], row["replace"], row.get("type") or "auto",
                                  row.get("font") or None))
            except ValueError as e:
                raise ValueError(f"{path} line {i}: {e}") from None
    if not rules:
        raise ValueError(f"{path}: no rules found")
    return rules
