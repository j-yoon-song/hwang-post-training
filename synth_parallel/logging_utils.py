from __future__ import annotations

import logging
from pathlib import Path

from .io_utils import ensure_dir


def _set_hf_verbosity(level: int) -> None:
    # Best-effort: keep compatibility across datasets/huggingface_hub versions.
    try:
        from datasets.utils import logging as ds_logging  # type: ignore

        if level <= logging.DEBUG:
            ds_logging.set_verbosity_debug()
        elif level <= logging.INFO:
            ds_logging.set_verbosity_info()
        elif level <= logging.WARNING:
            ds_logging.set_verbosity_warning()
        else:
            ds_logging.set_verbosity_error()
        try:
            ds_logging.enable_progress_bar()
        except Exception:  # pylint: disable=broad-except
            pass
    except Exception:  # pylint: disable=broad-except
        pass

    try:
        from huggingface_hub.utils import logging as hf_logging  # type: ignore

        if level <= logging.DEBUG:
            hf_logging.set_verbosity_debug()
        elif level <= logging.INFO:
            hf_logging.set_verbosity_info()
        elif level <= logging.WARNING:
            hf_logging.set_verbosity_warning()
        else:
            hf_logging.set_verbosity_error()
        try:
            hf_logging.enable_progress_bars()
        except Exception:  # pylint: disable=broad-except
            pass
    except Exception:  # pylint: disable=broad-except
        pass


def setup_logging(out_dir: str | Path, log_level: str = "INFO") -> Path:
    out = ensure_dir(out_dir)
    log_path = out / "logs.txt"

    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _set_hf_verbosity(level)
    return log_path
