"""Matching logic with a fake OCR engine - fast, no model needed."""
import numpy as np

from vidtextswap.finder import Finder
from vidtextswap.rules import Rule


class FakeOCR:
    """Returns fixed lines of text; each line is one box 20px tall."""

    def __init__(self, *lines):
        self.lines = lines

    def __call__(self, img):
        out = []
        for i, text in enumerate(self.lines):
            y = 10 + i * 40
            w = 12 * len(text)
            out.append(([[10, y], [10 + w, y], [10 + w, y + 20], [10, y + 20]], text, 0.99))
        return out, None


IMG = np.zeros((400, 800, 3), np.uint8)


def found(rules, *lines, fuzz=0.85):
    return [(h["rule"].find, h["old_text"]) for h in Finder(rules, fuzz=fuzz, ocr=FakeOCR(*lines)).find(IMG)]


def test_phone_in_many_formats():
    r = Rule("050 123 4567", "050 765 4321")
    assert found([r], "0501234567") == [("050 123 4567", "0501234567")]
    assert found([r], "Call 050-123-4567 now") == [("050 123 4567", "050-123-4567")]
    assert found([r], "WhatsApp:+971501234567") == [("050 123 4567", "+971501234567")]
    assert found([r], "+971 50 123 4567") == [("050 123 4567", "+971 50 123 4567")]
    assert found([r], "O5Ol234567") == [("050 123 4567", "O5Ol234567")]   # OCR read 0 as O, 1 as l
    assert found([r], "050 123 4568") == []


def test_text_case_insensitive_and_inside_a_sentence():
    r = Rule("Al Quoz", "JLT")
    assert found([r], "Visit our AL QUOZ showroom") == [("Al Quoz", "AL QUOZ")]


def test_text_survives_ocr_glitches():
    r = Rule("Al Quoz showroom", "JLT showroom")
    assert found([r], "Visit our Al Ouoz showroom") == [("Al Quoz showroom", "Al Ouoz showroom")]
    assert found([r], "Visit our Al Ouoz showroom", fuzz=1.0) == []


def test_glued_words_still_match():
    r = Rule("31 Dec", "15 Jan")
    assert found([r], "OFFERENDS31DEC") == [("31 Dec", "31DEC")]


def test_short_text_not_matched_inside_longer_words():
    r = Rule("JLT", "Marina")
    assert found([r], "JLTX tower") == []
    assert found([r], "JLT tower") == [("JLT", "JLT")]


def test_every_occurrence_and_every_rule():
    rules = [Rule("Al Quoz", "JLT"), Rule("050 123 4567", "050 765 4321")]
    got = found(rules, "Al Quoz and Al Quoz", "Call 0501234567")
    assert got == [("Al Quoz", "Al Quoz"), ("Al Quoz", "Al Quoz"), ("050 123 4567", "0501234567")]


def test_longer_rule_wins_on_overlap():
    rules = [Rule("Al Quoz", "JLT"), Rule("Al Quoz Branch", "Marina Branch")]
    assert found(rules, "Al Quoz Branch") == [("Al Quoz Branch", "Al Quoz Branch")]


def test_alignment_flags():
    f = Finder([Rule("Al Quoz", "JLT")], ocr=FakeOCR("Al Quoz", "in Al Quoz today"))
    a, b = f.find(IMG)
    assert a["alone"] and not a["followed"]
    assert not b["alone"] and b["followed"]
