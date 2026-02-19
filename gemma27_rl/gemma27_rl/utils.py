from __future__ import annotations

import logging
import os
from pathlib import Path
import random

try:
    import torch
except Exception:  # pragma: no cover - optional during lightweight tests
    torch = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def resolve_device(requested: str | None) -> str:
    if requested and requested.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        logger.warning("CUDA requested (%s) but unavailable; falling back to cpu.", requested)
        return "cpu"
    if requested:
        return requested
    return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"


def resolve_torch_dtype(dtype_name: str | None):
    if torch is None:
        return None
    if not dtype_name:
        return None
    key = dtype_name.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def configure_huggingface_cache(cache_dir: str | None, token: str | None = None) -> str | None:
    if not cache_dir:
        return None

    root = Path(cache_dir).expanduser().resolve()
    hub = root / "hub"
    transformers = root / "transformers"
    datasets = root / "datasets"

    hub.mkdir(parents=True, exist_ok=True)
    transformers.mkdir(parents=True, exist_ok=True)
    datasets.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers)
    os.environ["HF_DATASETS_CACHE"] = str(datasets)
    # Disable xet backend for more stable large-model downloads on flaky networks.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    logger.info(
        "Configured Hugging Face cache: HF_HOME=%s TRANSFORMERS_CACHE=%s HF_DATASETS_CACHE=%s token=%s",
        root,
        transformers,
        datasets,
        "set" if token else "unset",
    )
    return str(root)


def resolve_huggingface_token(explicit_token: str | None, token_env_name: str | None = "HF_TOKEN") -> str | None:
    if explicit_token and explicit_token.strip():
        return explicit_token.strip()

    candidate_envs: list[str] = []
    if token_env_name and token_env_name.strip():
        candidate_envs.append(token_env_name.strip())
    candidate_envs.extend(["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])

    seen: set[str] = set()
    for env_name in candidate_envs:
        if env_name in seen:
            continue
        seen.add(env_name)
        value = os.environ.get(env_name)
        if value and value.strip():
            logger.info("Using Hugging Face token from env var: %s", env_name)
            return value.strip()
    return None


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))
