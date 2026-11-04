import pytest

from vidtextswap.rules import Rule, load_rules_csv, normalise


def test_auto_detects_phone_and_text():
    assert Rule("050 123 4567", "050 765 4321").kind == "phone"
    assert Rule("+971 (50) 123-4567", "x").kind == "phone"
    assert Rule("Al Quoz", "JLT").kind == "text"
    assert Rule("31 Dec", "15 Jan").kind == "text"   # too few digits for a phone number


def test_phone_targets_cover_local_and_international_forms():
    r = Rule("050 123 4567", "050 765 4321")
    assert "0501234567" in r.targets
    assert "501234567" in r.targets          # as written after +971


def test_text_normalisation_ignores_case_spacing_and_ocr_lookalikes():
    assert normalise("Al  Quoz!") == normalise("al quoz")
    assert normalise("0ffer") == normalise("Offer")
    assert normalise("I1l") == "lll"


def test_keeps_all_caps_style_unless_disabled():
    r = Rule("31 Dec", "15 Jan")
    assert r.replacement_for("31 DEC") == "15 JAN"
    assert r.replacement_for("31 Dec") == "15 Jan"
    assert r.replacement_for("31 DEC", keep_case=False) == "15 Jan"


def test_rejects_bad_rules():
    with pytest.raises(ValueError):
        Rule("", "x")
    with pytest.raises(ValueError):
        Rule("12 34", "x", kind="phone")
    with pytest.raises(ValueError):
        Rule("abc", "x", kind="weird")


def test_load_rules_csv(tmp_path):
    p = tmp_path / "rules.csv"
    p.write_text("find,replace,type\n050 123 4567,050 765 4321,phone\nAl Quoz,JLT,\n\n", encoding="utf-8")
    rules = load_rules_csv(p)
    assert [(r.find, r.replace, r.kind) for r in rules] == [
        ("050 123 4567", "050 765 4321", "phone"), ("Al Quoz", "JLT", "text")]


def test_load_rules_csv_errors(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("find\nx\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_rules_csv(p)
