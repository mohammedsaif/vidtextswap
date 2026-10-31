"""Font discovery and loading."""
from pathlib import Path

from PIL import ImageFont

# (bold, regular) pairs, most Arial-like first. The style matcher picks whichever
# weight looks closest to the original text.
_FONT_PAIRS = [
    # Windows
    ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"),
    ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/segoeui.ttf"),
    # macOS
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"),
    ("/Library/Fonts/Arial Bold.ttf", "/Library/Fonts/Arial.ttf"),
    # Linux
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]


def candidate_fonts(user_font=None):
    """Fonts the style matcher may choose from. A user-supplied font is used alone."""
    if user_font:
        if not Path(user_font).exists():
            raise FileNotFoundError(f"Font not found: {user_font}")
        return [str(user_font)]
    for pair in _FONT_PAIRS:
        found = [p for p in pair if Path(p).exists()]
        if found:
            return found
    return [None]  # Pillow's built-in font


def load_font(path, size):
    size = max(6, int(round(size)))
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()
