"""The Python API used by the web app."""
from pathlib import Path

from PIL import Image, ImageDraw

from vidtextswap import Rule
from vidtextswap.api import process_files, scan_files
from vidtextswap.fonts import candidate_fonts, load_font

ROOT = Path(__file__).resolve().parents[1]


def poster(path, text="Call 050 123 4567"):
    img = Image.new("RGB", (900, 500), (40, 60, 90))
    ImageDraw.Draw(img).text((60, 200), text, font=load_font(candidate_fonts()[0], 64), fill="white")
    img.save(path, quality=95)


def test_process_files_writes_only_changed_files(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    poster(a)
    poster(b, "Nothing to change here")
    res = process_files([a, b, tmp_path / "notes.txt"], [Rule("050 123 4567", "050 765 4321")],
                        tmp_path / "out", inpaint="fast")
    assert [r.status for r in res] == ["updated", "no match"]          # .txt ignored
    assert Path(res[0].output).exists() and Path(res[0].preview).exists()
    assert not (tmp_path / "out" / "b.jpg").exists()


def test_same_name_twice_does_not_overwrite(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    poster(tmp_path / "x" / "p.jpg")
    poster(tmp_path / "y" / "p.jpg")
    res = process_files([tmp_path / "x" / "p.jpg", tmp_path / "y" / "p.jpg"],
                        [Rule("050 123 4567", "050 765 4321")], tmp_path / "out", inpaint="fast")
    assert len({r.output for r in res}) == 2


def test_scan_files_finds_phone(tmp_path):
    poster(tmp_path / "a.jpg")
    items, texts = scan_files([tmp_path / "a.jpg"])
    assert "050 123 4567" in items["Phone numbers"]


def test_space_examples_contain_only_fake_data():
    ex = ROOT / "space" / "examples"
    if ex.exists():
        items, _ = scan_files(sorted(ex.iterdir()))
        from vidtextswap.rules import only_digits
        assert {only_digits(n) for n in items.get("Phone numbers", {})} <= {"0501234567"}
