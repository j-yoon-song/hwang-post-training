from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .config import FiltersConfig, SourceAtomicityConfig
from .prompts import (
    build_judge_messages,
    build_round_trip_backtranslate_messages,
    build_round_trip_judge_messages,
    parse_judge_json,
)
from .teacher import TeacherClient
from .utils import jaccard_overlap


_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HTML_RE = re.compile(r"(?is)<\s*[a-z!/][^>]*>")
_BULLET_LINE_RE = re.compile(r"(?m)^\s*(?:[-*•]|[0-9]+[.)])\s+\S+")
_SENTENCE_SPLIT_RE = re.compile(r"\n+|(?<=[.!?。！？])\s+")


@dataclass
class FilterDecision:
    passed: bool
    reason_code: str
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def apply_rule_based_filters(source_text: str, target_text: str, cfg: FiltersConfig) -> FilterDecision:
    if cfg.source_atomicity.enabled:
        source_decision = _apply_source_atomicity_filter(source_text, cfg.source_atomicity)
        if not source_decision.passed:
            return source_decision

    cleaned = target_text.strip()
    if not cleaned:
        return FilterDecision(False, "empty")

    if len(cleaned) < cfg.min_chars:
        return FilterDecision(False, "too_short")

    if len(cleaned) > cfg.max_chars:
        return FilterDecision(False, "too_long")

    lowered = cleaned.lower()
    for token in cfg.blocked_substrings:
        if token.lower() in lowered:
            return FilterDecision(False, "blocked_substring", notes=token)

    ratio = len(cleaned) / max(1, len(source_text.strip()))
    if ratio < cfg.length_ratio_min:
        return FilterDecision(False, "ratio_too_small", notes=f"ratio={ratio:.3f}")
    if ratio > cfg.length_ratio_max:
        return FilterDecision(False, "ratio_too_large", notes=f"ratio={ratio:.3f}")

    overlap = jaccard_overlap(source_text, cleaned)
    if overlap > cfg.max_copy_overlap:
        return FilterDecision(False, "copy_overlap", notes=f"overlap={overlap:.3f}")

    return FilterDecision(True, "pass")


def _apply_source_atomicity_filter(source_text: str, cfg: SourceAtomicityConfig) -> FilterDecision:
    text = source_text.strip()
    if not text:
        return FilterDecision(False, "source_empty")

    if len(text) < cfg.min_chars:
        return FilterDecision(False, "source_too_short", notes=f"chars={len(text)}")
    if len(text) > cfg.max_chars:
        return FilterDecision(False, "source_too_long", notes=f"chars={len(text)}")

    newline_count = text.count("\n")
    if newline_count > cfg.max_newlines:
        return FilterDecision(False, "source_multiline", notes=f"newlines={newline_count}")

    sentence_count = _approx_sentence_count(text)
    if sentence_count > cfg.max_sentences:
        return FilterDecision(
            False,
            "source_non_atomic",
            notes=f"sentences={sentence_count}",
        )

    word_count = len(text.split())
    if word_count > cfg.max_words:
        return FilterDecision(False, "source_too_many_words", notes=f"words={word_count}")

    lowered = text.lower()
    for token in cfg.blocked_substrings:
        token_norm = str(token).strip().lower()
        if token_norm and token_norm in lowered:
            return FilterDecision(False, "source_blocked_substring", notes=token_norm)

    if cfg.reject_urls and _URL_RE.search(text):
        return FilterDecision(False, "source_contains_url")
    if cfg.reject_emails and _EMAIL_RE.search(text):
        return FilterDecision(False, "source_contains_email")
    if cfg.reject_html and _HTML_RE.search(text):
        return FilterDecision(False, "source_contains_html")
    if cfg.reject_bullets and _BULLET_LINE_RE.search(text):
        return FilterDecision(False, "source_list_like")

    ratios = _text_char_ratios(text)
    letter_ratio = ratios["letter_ratio"]
    digit_ratio = ratios["digit_ratio"]
    punct_symbol_ratio = ratios["punct_symbol_ratio"]

    if letter_ratio < cfg.min_letter_ratio:
        return FilterDecision(False, "source_low_letter_ratio", notes=f"letter_ratio={letter_ratio:.3f}")
    if digit_ratio > cfg.max_digit_ratio:
        return FilterDecision(False, "source_high_digit_ratio", notes=f"digit_ratio={digit_ratio:.3f}")
    if punct_symbol_ratio > cfg.max_punct_symbol_ratio:
        return FilterDecision(
            False,
            "source_high_punct_symbol_ratio",
            notes=f"punct_symbol_ratio={punct_symbol_ratio:.3f}",
        )

    return FilterDecision(True, "pass")


def _approx_sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    chunks = [part.strip() for part in _SENTENCE_SPLIT_RE.split(stripped) if part.strip()]
    return max(1, len(chunks))


def _text_char_ratios(text: str) -> dict[str, float]:
    visible_chars = [ch for ch in text if not ch.isspace()]
    if not visible_chars:
        return {"letter_ratio": 0.0, "digit_ratio": 0.0, "punct_symbol_ratio": 0.0}

    visible_count = len(visible_chars)
    letter_count = sum(1 for ch in visible_chars if ch.isalpha())
    digit_count = sum(1 for ch in visible_chars if ch.isdigit())
    punct_symbol_count = 0
    for ch in visible_chars:
        cat = unicodedata.category(ch)
        if cat and cat[0] in {"P", "S"}:
            punct_symbol_count += 1

    return {
        "letter_ratio": letter_count / visible_count,
        "digit_ratio": digit_count / visible_count,
        "punct_symbol_ratio": punct_symbol_count / visible_count,
    }


def apply_llm_judge_filter(
    teacher: TeacherClient,
    source_lang: str,
    target_lang: str,
    source_text: str,
    target_text: str,
    cfg: FiltersConfig,
) -> FilterDecision:
    if cfg.llm_judge.round_trip.enabled:
        return _apply_round_trip_judge_filter(
            teacher=teacher,
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=source_text,
            target_text=target_text,
            cfg=cfg,
        )

    messages = build_judge_messages(source_lang, target_lang, source_text, target_text)
    generation_cfg = teacher.cfg.generation
    temperature = (
        float(generation_cfg.sampling_temperature)
        if generation_cfg.sampling_temperature is not None
        else float(cfg.llm_judge.temperature)
    )
    top_p = (
        float(generation_cfg.sampling_top_p)
        if generation_cfg.sampling_top_p is not None
        else float(generation_cfg.top_p)
    )
    presence_penalty = generation_cfg.sampling_presence_penalty
    sampling_extra_body: dict[str, Any] = {}
    if generation_cfg.sampling_top_k is not None:
        sampling_extra_body["top_k"] = int(generation_cfg.sampling_top_k)
    if generation_cfg.sampling_min_p is not None:
        sampling_extra_body["min_p"] = float(generation_cfg.sampling_min_p)
    if generation_cfg.sampling_repetition_penalty is not None:
        sampling_extra_body["repetition_penalty"] = float(generation_cfg.sampling_repetition_penalty)

    try:
        content = teacher.complete(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=cfg.llm_judge.max_tokens,
            model=cfg.llm_judge.model,
            presence_penalty=presence_penalty,
            extra_body=sampling_extra_body or None,
        )
    except Exception as exc:  # pylint: disable=broad-except
        if cfg.llm_judge.fail_policy == "permissive":
            return FilterDecision(True, "judge_failed_permissive", notes=str(exc))
        return FilterDecision(False, "judge_failed", notes=str(exc))

    parsed = parse_judge_json(content)
    if parsed is None:
        if cfg.llm_judge.fail_policy == "permissive":
            return FilterDecision(True, "judge_parse_failed_permissive")
        return FilterDecision(False, "judge_parse_failed")

    passed = bool(parsed.get("pass", False))
    reason = str(parsed.get("reason_code", "judge_reject" if not passed else "pass"))
    notes = str(parsed.get("notes", ""))
    return FilterDecision(passed=passed, reason_code=reason, notes=notes)


def apply_round_trip_semantic_filter(
    teacher: TeacherClient,
    source_lang: str,
    target_lang: str,
    source_text: str,
    target_text: str,
    cfg: FiltersConfig,
) -> FilterDecision:
    return _apply_round_trip_judge_filter(
        teacher=teacher,
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        target_text=target_text,
        cfg=cfg,
    )


def _sampling_extra_body(generation_cfg: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if generation_cfg.sampling_top_k is not None:
        payload["top_k"] = int(generation_cfg.sampling_top_k)
    if generation_cfg.sampling_min_p is not None:
        payload["min_p"] = float(generation_cfg.sampling_min_p)
    if generation_cfg.sampling_repetition_penalty is not None:
        payload["repetition_penalty"] = float(generation_cfg.sampling_repetition_penalty)
    return payload


def _apply_round_trip_judge_filter(
    teacher: TeacherClient,
    source_lang: str,
    target_lang: str,
    source_text: str,
    target_text: str,
    cfg: FiltersConfig,
) -> FilterDecision:
    generation_cfg = teacher.cfg.generation
    rt_cfg = cfg.llm_judge.round_trip
    model_name = rt_cfg.model or cfg.llm_judge.model
    presence_penalty = generation_cfg.sampling_presence_penalty
    sampling_extra_body = _sampling_extra_body(generation_cfg)

    backtranslate_messages = build_round_trip_backtranslate_messages(
        source_lang=source_lang,
        target_lang=target_lang,
        candidate_translation=target_text,
    )
    try:
        backtranslated_text = teacher.complete(
            messages=backtranslate_messages,
            temperature=float(rt_cfg.back_translation_temperature),
            top_p=float(rt_cfg.back_translation_top_p),
            max_tokens=int(rt_cfg.back_translation_max_tokens),
            model=model_name,
            presence_penalty=presence_penalty,
            extra_body=sampling_extra_body or None,
        ).strip()
    except Exception as exc:  # pylint: disable=broad-except
        if cfg.llm_judge.fail_policy == "permissive":
            return FilterDecision(
                True,
                "round_trip_backtranslate_failed_permissive",
                notes=str(exc),
                metadata={"round_trip_error": str(exc)},
            )
        return FilterDecision(
            False,
            "round_trip_backtranslate_failed",
            notes=str(exc),
            metadata={"round_trip_error": str(exc)},
        )

    if not backtranslated_text:
        if cfg.llm_judge.fail_policy == "permissive":
            return FilterDecision(
                True,
                "round_trip_empty_backtranslation_permissive",
                metadata={"back_translation_text": ""},
            )
        return FilterDecision(False, "round_trip_empty_backtranslation")

    judge_messages = build_round_trip_judge_messages(
        source_lang=source_lang,
        source_text=source_text,
        back_translated_text=backtranslated_text,
    )
    try:
        judge_content = teacher.complete(
            messages=judge_messages,
            temperature=float(rt_cfg.judge_temperature),
            top_p=float(rt_cfg.judge_top_p),
            max_tokens=int(rt_cfg.judge_max_tokens),
            model=model_name,
            presence_penalty=presence_penalty,
            extra_body=sampling_extra_body or None,
        )
    except Exception as exc:  # pylint: disable=broad-except
        if cfg.llm_judge.fail_policy == "permissive":
            return FilterDecision(
                True,
                "round_trip_judge_failed_permissive",
                notes=str(exc),
                metadata={"back_translation_text": backtranslated_text, "round_trip_error": str(exc)},
            )
        return FilterDecision(
            False,
            "round_trip_judge_failed",
            notes=str(exc),
            metadata={"back_translation_text": backtranslated_text, "round_trip_error": str(exc)},
        )

    parsed = parse_judge_json(judge_content)
    if parsed is None:
        if cfg.llm_judge.fail_policy == "permissive":
            return FilterDecision(
                True,
                "round_trip_parse_failed_permissive",
                metadata={"back_translation_text": backtranslated_text},
            )
        return FilterDecision(
            False,
            "round_trip_parse_failed",
            metadata={"back_translation_text": backtranslated_text},
        )

    similarity: float | None = None
    raw_similarity = parsed.get("semantic_similarity", parsed.get("similarity"))
    if raw_similarity is not None:
        try:
            similarity = float(raw_similarity)
        except (TypeError, ValueError):
            similarity = None

    passed = bool(parsed.get("pass", False))
    reason = str(parsed.get("reason_code", "round_trip_reject" if not passed else "pass"))
    notes = str(parsed.get("notes", ""))

    if similarity is not None and similarity < float(rt_cfg.min_semantic_similarity):
        passed = False
        reason = "round_trip_similarity_low"
        if notes:
            notes = (
                f"{notes}; semantic_similarity={similarity:.3f} "
                f"(threshold={float(rt_cfg.min_semantic_similarity):.3f})"
            )
        else:
            notes = (
                f"semantic_similarity={similarity:.3f} "
                f"(threshold={float(rt_cfg.min_semantic_similarity):.3f})"
            )

    metadata: dict[str, Any] = {"back_translation_text": backtranslated_text}
    if similarity is not None:
        metadata["semantic_similarity"] = similarity
    return FilterDecision(passed=passed, reason_code=reason, notes=notes, metadata=metadata)
