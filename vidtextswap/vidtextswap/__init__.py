"""VidTextSwap - find text in videos and images and replace it in the same style."""
__version__ = "0.2.0"

__all__ = ["Rule", "load_rules_csv", "__version__"]

from .rules import Rule, load_rules_csv  # noqa: E402
