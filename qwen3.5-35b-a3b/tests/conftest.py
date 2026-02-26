"""Register the oddly-named package so tests can import it as ``qwen3_5_35b_a3b``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PKG_ALIAS = "qwen3_5_35b_a3b"
_PKG_DIR = Path(__file__).resolve().parent.parent / "qwen3.5-35b-a3b"

if _PKG_ALIAS not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        _PKG_ALIAS,
        _PKG_DIR / "__init__.py",
        submodule_search_locations=[str(_PKG_DIR)],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[_PKG_ALIAS] = pkg
    spec.loader.exec_module(pkg)
