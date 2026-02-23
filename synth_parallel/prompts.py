from __future__ import annotations

import json
import re
from typing import Any


def build_translation_messages(
    source_lang: str,
    target_lang: str,
    src_lang_code: str,
    tgt_lang_code: str,
    text: str,
) -> list[dict[str, str]]:
    system = "You are a professional translator."
    user = (
        f"Translate the following text from {source_lang} ({src_lang_code}) "
        f"to {target_lang} ({tgt_lang_code}).\n"
        "Preserve meaning and nuance. Use natural grammar in the target language.\n"
        "Output only the translation with no commentary.\n\n"
        f"Text:\n{text}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_judge_messages(
    source_lang: str,
    target_lang: str,
    source_text: str,
    candidate_translation: str,
) -> list[dict[str, str]]:
    system = (
        "You are a strict translation quality gate for dataset curation. "
        "Be conservative: if uncertain, reject. "
        "Return JSON only with no markdown and no extra text."
    )
    user = (
        "Task: decide whether the candidate translation is safe to keep.\n\n"
        "Check all criteria:\n"
        "1) Language: output must be in target language (not source language or mixed heavily).\n"
        "2) Fidelity: preserve meaning, facts, entities, numbers, dates, units, negation, and relations.\n"
        "3) Completeness: no major omission, truncation, or hallucinated additions.\n"
        "4) Format: translation text only (no 'Here is the translation', role tags, code fences, or explanations).\n"
        "5) Critical errors dominate: a single severe error means reject.\n\n"
        "Use one reason_code from this set:\n"
        "pass, wrong_language, severe_meaning_error, omission, addition_or_hallucination, "
        "format_violation, too_literal_or_ungrammatical, unsafe_or_corrupted.\n\n"
        "Return exactly one JSON object with this schema:\n"
        '{"pass": true|false, "reason_code": "<one_code>", "notes": "<short>", "semantic_similarity": 0.0-1.0}\n'
        "Rules:\n"
        "- Keep notes short and concrete (max 25 words).\n"
        "- semantic_similarity is your estimate of meaning match (0.0 to 1.0).\n"
        "- Output JSON only.\n\n"
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n"
        f"Source text:\n{source_text}\n\n"
        f"Candidate translation:\n{candidate_translation}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_round_trip_backtranslate_messages(
    source_lang: str,
    target_lang: str,
    candidate_translation: str,
) -> list[dict[str, str]]:
    system = (
        "You are a professional translator doing back-translation for QA. "
        "Preserve all information exactly; do not summarize, explain, or clean up content."
    )
    user = (
        f"Translate the following text from {target_lang} back to {source_lang}.\n"
        "Requirements:\n"
        "- Keep named entities, numbers, dates, units, negation, and modality faithful.\n"
        "- Keep sentence boundaries and discourse relations when possible.\n"
        "- Do not add commentary, labels, or quotation wrappers.\n"
        "- Output only the back-translation text.\n\n"
        f"Text:\n{candidate_translation}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_round_trip_judge_messages(
    source_lang: str,
    source_text: str,
    back_translated_text: str,
) -> list[dict[str, str]]:
    system = (
        "You are a strict semantic equivalence judge for back-translation QA. "
        "Be conservative and return JSON only."
    )
    user = (
        "Compare the original source text and the back-translated text.\n"
        "Focus on semantic equivalence, not stylistic differences.\n\n"
        "Critical checks:\n"
        "1) Facts/entities/numbers/dates/units preserved.\n"
        "2) Negation, modality, and causal/temporal relations preserved.\n"
        "3) No major omission, addition, contradiction, or role reversal.\n"
        "4) If uncertain, reject.\n\n"
        "Use one reason_code from:\n"
        "pass, equivalent, minor_drift, omission, addition_or_hallucination, contradiction, "
        "entity_or_number_error, unclear_or_corrupted.\n\n"
        "Return exactly one JSON object:\n"
        '{"pass": true|false, "semantic_similarity": 0.0-1.0, "reason_code": "<one_code>", "notes": "<short>"}\n'
        "Scoring guide for semantic_similarity:\n"
        "- 0.95-1.00: near-perfect equivalence.\n"
        "- 0.85-0.94: small non-critical drift.\n"
        "- 0.70-0.84: noticeable meaning drift.\n"
        "- 0.00-0.69: severe mismatch.\n"
        "Output JSON only.\n\n"
        f"Language: {source_lang}\n"
        f"Original source text:\n{source_text}\n\n"
        f"Back-translated text:\n{back_translated_text}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_judge_json(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "pass" in payload:
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict) and "pass" in payload:
            return payload
    except json.JSONDecodeError:
        return None
    return None
