"""Background repair: OpenCV (fast) or LaMa (AI, better on busy backgrounds).

LaMa (Suvorov et al., "Resolution-robust Large Mask Inpainting with Fourier
Convolutions", https://github.com/advimman/lama) runs locally through ONNX
Runtime. The ~90 MB model is downloaded once on first use and cached.
"""
import hashlib
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# (url, sha256 or None). First is OpenCV Zoo's copy (Apache-2.0), second the original export.
MODEL_SOURCES = [
    ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/inpainting_lama/"
     "inpainting_lama_2025jan.onnx",
     "7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2"),
    ("https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx", None),
]
SIZE = 512


def cache_dir():
    base = os.environ.get("VIDTEXTSWAP_CACHE") or (
        Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "vidtextswap")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(log=print):
    """Path to the LaMa model, downloading it on first use."""
    custom = os.environ.get("VIDTEXTSWAP_LAMA_MODEL")
    if custom:
        return Path(custom)
    target = cache_dir() / "lama.onnx"
    if target.exists() and target.stat().st_size > 10_000_000:
        return target
    errors = []
    for url, sha in MODEL_SOURCES:
        tmp = target.with_suffix(".part")
        try:
            log("Downloading the AI background-repair model (~90 MB, one time only)...")
            urllib.request.urlretrieve(url, tmp)
            if sha and _sha256(tmp) != sha:
                raise ValueError("checksum mismatch")
            tmp.replace(target)
            return target
        except Exception as e:  # try the next source
            errors.append(f"{url}: {e}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError("Could not download the LaMa model.\n  " + "\n  ".join(errors) +
                       "\nDownload it manually and set VIDTEXTSWAP_LAMA_MODEL=<path to .onnx>.")


class Lama:
    def __init__(self, model_path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])

    def __call__(self, bgr, mask):
        """Inpaint `mask` (uint8, >0 = fill) in a BGR image. Returns BGR."""
        h, w = mask.shape
        ys, xs = np.where(mask > 0)
        if not len(ys):
            return bgr.copy()
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        # Square crop with generous context around the hole, so LaMa sees the pattern
        m = max(32, int(0.6 * max(y1 - y0, 24)))
        side = min(max(x1 - x0, y1 - y0) + 2 * m, max(h, w))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        X0 = int(np.clip(cx - side // 2, 0, max(0, w - side)))
        Y0 = int(np.clip(cy - side // 2, 0, max(0, h - side)))
        X1, Y1 = min(w, X0 + side), min(h, Y0 + side)
        crop, mc = bgr[Y0:Y1, X0:X1], mask[Y0:Y1, X0:X1]
        rgb = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        mm = (cv2.resize(mc, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
        mm = cv2.dilate(mm, np.ones((3, 3), np.uint8))
        out = self.sess.run(None, {"image": (rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None],
                                   "mask": mm[None, None]})[0][0]
        out = np.clip(out.transpose(1, 2, 0), 0, 255).astype(np.uint8)
        out = cv2.cvtColor(cv2.resize(out, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)
        result = bgr.copy()
        sub = result[Y0:Y1, X0:X1]
        k = mc > 0
        sub[k] = out[k]
        return result


_LAMA = None
_LAMA_ERROR = None


def get_lama(log=print):
    global _LAMA, _LAMA_ERROR
    if _LAMA is None and _LAMA_ERROR is None:
        try:
            _LAMA = Lama(ensure_model(log))
        except Exception as e:
            _LAMA_ERROR = e
    if _LAMA is None:
        raise RuntimeError(f"AI background repair unavailable: {_LAMA_ERROR}")
    return _LAMA


def lama_available(required=False):
    """True if LaMa can run. In auto mode a failure falls back to OpenCV (with one
    warning); when the user asked for AI explicitly, the error is raised."""
    try:
        get_lama()
        return True
    except RuntimeError as e:
        if required:
            raise
        global _WARNED
        if not _WARNED:
            print(f"      warning: {e} - using the fast method instead", flush=True)
            _WARNED = True
        return False


_WARNED = False


def inpaint(crop, mask, method):
    """method: 'ai' (LaMa) or 'fast' (OpenCV Telea)."""
    if not mask.any():
        return crop
    if method == "ai":
        return get_lama()(crop, mask)
    return cv2.inpaint(crop, mask, 5, cv2.INPAINT_TELEA)
