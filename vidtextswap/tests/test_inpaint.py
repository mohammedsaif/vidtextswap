"""Background repair: still-shot caching logic (fake LaMa) and the real model if available."""
import os

import numpy as np
import pytest

from vidtextswap import inpaint, style


def make_plan(method):
    erase = np.zeros((40, 80), np.uint8)
    erase[15:25, 20:60] = 255
    return {"X0": 10, "Y0": 10, "X1": 90, "Y1": 50, "erase": erase, "flat_bg": None,
            "rgb": np.zeros((40, 80, 3), np.float32), "alpha": np.zeros((40, 80, 1), np.float32),
            "inpaint": method, "desc": ""}


@pytest.fixture
def fake_lama(monkeypatch):
    calls = []

    def lama(bgr, mask):
        calls.append(1)
        out = bgr.copy()
        out[mask > 0] = 7
        return out
    monkeypatch.setattr(inpaint, "get_lama", lambda log=print: lama)
    monkeypatch.setattr(inpaint, "lama_available", lambda required=False: True)
    return calls


def frame(shift=0):
    rng = np.random.default_rng(0)
    f = rng.integers(0, 255, (100, 140, 3), dtype=np.uint8)
    return np.roll(f, shift, axis=1)


def test_still_background_runs_ai_once(fake_lama):
    plan = make_plan("auto")
    for _ in range(5):
        out = style.apply(frame(), plan)
    assert len(fake_lama) == 1
    assert (out[25:35, 30:70] == 7).all()          # AI result reused on every frame


def test_moving_background_auto_uses_fast_ai_reruns(fake_lama):
    plan = make_plan("auto")
    style.apply(frame(0), plan)
    style.apply(frame(9), plan)                    # camera moved
    assert len(fake_lama) == 1                     # auto: fast method for moving frames

    plan = make_plan("ai")
    style.apply(frame(0), plan)
    style.apply(frame(9), plan)
    assert len(fake_lama) == 3                     # ai: re-run when the background moves


def test_fast_never_calls_ai(fake_lama):
    plan = make_plan("fast")
    style.apply(frame(), plan)
    assert not fake_lama


def _model_ready():
    try:
        p = os.environ.get("VIDTEXTSWAP_LAMA_MODEL") or str(inpaint.cache_dir() / "lama.onnx")
        return os.path.exists(p) or os.environ.get("VIDTEXTSWAP_TEST_AI") == "1"
    except OSError:
        return False


@pytest.mark.skipif(not _model_ready(), reason="LaMa model not downloaded (set VIDTEXTSWAP_TEST_AI=1)")
def test_real_lama_restores_a_pattern():
    # stripes with a hole: LaMa should continue the stripes, beating OpenCV
    img = np.zeros((256, 256, 3), np.uint8)
    img[:, ::16] = 255
    img[:, 1::16] = 255
    mask = np.zeros((256, 256), np.uint8)
    mask[100:150, 60:200] = 255
    holed = img.copy()
    holed[mask > 0] = 128
    ai = inpaint.get_lama()(holed, mask)
    fast = inpaint.inpaint(holed, mask, "fast")
    err = lambda r: np.abs(r.astype(int) - img)[mask > 0].mean()
    assert err(ai) < err(fast)
