"""Plain-language instructions and scan, with a fake Claude client (no API key needed)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from vidtextswap import assistant
from vidtextswap.assistant import find_items, propose_rules

TEXTS = {
    "Now open in Al Quoz": {Path("a.jpg")},
    "Call us: 050 123 4567 today": {Path("a.jpg"), Path("b.mp4")},
    "WhatsApp: +971501234567": {Path("c.jpg")},
    "OFFER ENDS 31 DEC": {Path("b.mp4")},
    "www.oldshop.ae | AED 1,499 | 20% off": {Path("a.jpg")},
}


class FakeClient:
    """Mimics anthropic.Anthropic().messages.create(...) returning a tool_use block."""

    def __init__(self, tool_input):
        self.tool_input = tool_input
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=self.tool_input)])


def test_find_items_groups_what_people_usually_update():
    items = find_items(TEXTS)
    assert "050 123 4567" in items["Phone numbers"]
    assert items["Phone numbers"]["050 123 4567"] == 2          # in two files
    assert "www.oldshop.ae" in items["Websites"]
    assert "31 DEC" in items["Dates"]
    assert any("1,499" in p for p in items["Prices"])
    assert "20% off" in items["Offers"]


def test_propose_rules_keeps_valid_and_rejects_invented_text():
    client = FakeClient({
        "rules": [
            {"find": "050 123 4567", "replace": "050 765 4321", "kind": "phone"},
            {"find": "0501234567", "replace": "050 765 4321", "kind": "phone"},   # same number again
            {"find": "Al Quoz", "replace": "JLT", "kind": "text"},
            {"find": "Downtown", "replace": "Marina", "kind": "text"},           # not on screen
            {"find": "31 DEC", "replace": "", "kind": "text"},                    # empty replacement
        ],
        "notes": "There is only one date, so I changed nothing else.",
    })
    rules, notes, rejected = propose_rules("new number 050 765 4321, branch is now JLT", TEXTS, client=client)
    assert [(r.find, r.replace, r.kind) for r in rules] == [
        ("050 123 4567", "050 765 4321", "phone"), ("Al Quoz", "JLT", "text")]
    assert {f for f, _ in rejected} == {"Downtown", "31 DEC"}
    assert "only one date" in notes

    sent = client.calls[0]
    assert sent["tool_choice"] == {"type": "tool", "name": "set_replacement_rules"}
    user = sent["messages"][0]["content"]
    assert "new number 050 765 4321" in user and "Call us: 050 123 4567 today" in user


def test_phone_rule_validated_against_any_format_on_screen():
    client = FakeClient({"rules": [{"find": "+971 50 123 4567", "replace": "050 765 4321", "kind": "phone"}]})
    rules, _, rejected = propose_rules("update phone", {"WhatsApp: 0501234567": {Path("x.jpg")}}, client=client)
    assert len(rules) == 1 and not rejected


def test_no_tool_answer_is_an_error():
    client = FakeClient(None)
    client.messages = SimpleNamespace(create=lambda **kw: SimpleNamespace(content=[SimpleNamespace(type="text")]))
    with pytest.raises(RuntimeError):
        propose_rules("x", TEXTS, client=client)


def test_missing_key_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    pytest.importorskip("anthropic")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        assistant.get_client("claude")
