from __future__ import annotations

import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bruteforce_canvas.gradio_ui import CSS, _build_theme, build_demo


demo = build_demo(mode=os.getenv("BC_GRADIO_MODE"))


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(css=CSS, theme=_build_theme())
