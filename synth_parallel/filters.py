from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import FiltersConfig
from .prompts import (
    build_judge_messages,
    build_round_trip_backtranslate_messages,
    build_round_trip_judge_messages,
    parse_judge_json,
)
from .teacher import TeacherClient
from .utils import jaccard_overlap


@dataclass
class FilterDecision:
    passed: bool
    reason_code: str
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def apply_rule_based_filters(source_text: str, target_text: str, cfg: FiltersConfig) -> FilterDecision:
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
