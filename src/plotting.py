# Headless matplotlib setup with CJK font fallback (Linux has no Microsoft JhengHei).
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

CJK_CANDIDATES = [
    "Noto Sans CJK TC",
    "Noto Sans TC",
    "Microsoft JhengHei",
    "SimHei",
    "WenQuanYi Zen Hei",
    "AR PL UMing TW",
]

_configured = False
HAS_CJK = False


def setup_fonts() -> None:
    global _configured, HAS_CJK
    if _configured:
        return
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = [name]
            HAS_CJK = True
            break
    else:
        print("[WARN] No CJK font found; figures fall back to English labels.")
    plt.rcParams["axes.unicode_minus"] = False
    _configured = True


setup_fonts()


def label(zh: str, en: str) -> str:
    """Chinese figure text where a CJK font exists, English otherwise (tofu boxes
    are worse than English)."""
    return zh if HAS_CJK else en
