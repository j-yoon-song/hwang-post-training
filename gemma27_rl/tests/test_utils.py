from __future__ import annotations

from gemma27_rl.utils import resolve_huggingface_token


def test_resolve_hf_token_from_explicit_and_env(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert resolve_huggingface_token("explicit-token", "HF_TOKEN") == "explicit-token"


def test_resolve_hf_token_from_env_name(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("MY_HF_TOKEN", "custom-token")
    assert resolve_huggingface_token(None, "MY_HF_TOKEN") == "custom-token"
