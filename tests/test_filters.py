from types import SimpleNamespace

from synth_parallel.config import FiltersConfig
from synth_parallel.filters import apply_llm_judge_filter, apply_rule_based_filters


def test_rule_based_pass():
    cfg = FiltersConfig(min_chars=2, max_chars=200, max_copy_overlap=0.95)
    decision = apply_rule_based_filters("hello world", "안녕하세요 세계", cfg)
    assert decision.passed


def test_rule_based_reject_blocked_token():
    cfg = FiltersConfig(min_chars=2, max_chars=200)
    decision = apply_rule_based_filters("hello", "Here is the translation: 안녕하세요", cfg)
    assert not decision.passed
    assert decision.reason_code == "blocked_substring"


def test_rule_based_reject_length_ratio():
    cfg = FiltersConfig(min_chars=1, max_chars=200, length_ratio_min=0.5, length_ratio_max=1.5)
    decision = apply_rule_based_filters("hello world", "x", cfg)
    assert not decision.passed
    assert decision.reason_code == "ratio_too_small"


def test_llm_judge_uses_sampling_hyperparameters():
    class _FakeTeacher:
        def __init__(self):
            self.cfg = SimpleNamespace(
                generation=SimpleNamespace(
                    top_p=1.0,
                    sampling_temperature=0.7,
                    sampling_top_p=0.8,
                    sampling_top_k=20,
                    sampling_min_p=0.0,
                    sampling_presence_penalty=1.5,
                    sampling_repetition_penalty=1.0,
                )
            )
            self.last_kwargs = None

        def complete(self, **kwargs):
            self.last_kwargs = kwargs
            return '{"pass": true, "reason_code": "pass", "notes": "ok"}'

    teacher = _FakeTeacher()
    cfg = FiltersConfig()
    decision = apply_llm_judge_filter(
        teacher=teacher,  # type: ignore[arg-type]
        source_lang="English",
        target_lang="Korean",
        source_text="hello",
        target_text="안녕하세요",
        cfg=cfg,
    )

    assert decision.passed
    assert teacher.last_kwargs is not None
    assert teacher.last_kwargs["temperature"] == 0.7
    assert teacher.last_kwargs["top_p"] == 0.8
    assert teacher.last_kwargs["presence_penalty"] == 1.5
    assert teacher.last_kwargs["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }


def test_llm_judge_round_trip_passes_with_high_similarity():
    class _FakeTeacher:
        def __init__(self):
            self.cfg = SimpleNamespace(
                generation=SimpleNamespace(
                    top_p=1.0,
                    sampling_temperature=None,
                    sampling_top_p=None,
                    sampling_top_k=None,
                    sampling_min_p=None,
                    sampling_presence_penalty=None,
                    sampling_repetition_penalty=None,
                )
            )
            self.calls = []
            self.responses = [
                "hello world",
                '{"pass": true, "semantic_similarity": 0.93, "reason_code": "pass", "notes": ""}',
            ]

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return self.responses.pop(0)

    teacher = _FakeTeacher()
    cfg = FiltersConfig()
    cfg.llm_judge.round_trip.enabled = True
    cfg.llm_judge.round_trip.min_semantic_similarity = 0.8
    decision = apply_llm_judge_filter(
        teacher=teacher,  # type: ignore[arg-type]
        source_lang="English",
        target_lang="Korean",
        source_text="hello world",
        target_text="안녕하세요 세계",
        cfg=cfg,
    )

    assert decision.passed
    assert decision.reason_code == "pass"
    assert decision.metadata["back_translation_text"] == "hello world"
    assert decision.metadata["semantic_similarity"] == 0.93
    assert len(teacher.calls) == 2
    assert teacher.calls[0]["max_tokens"] == cfg.llm_judge.round_trip.back_translation_max_tokens
    assert teacher.calls[1]["max_tokens"] == cfg.llm_judge.round_trip.judge_max_tokens


def test_llm_judge_round_trip_rejects_low_similarity():
    class _FakeTeacher:
        def __init__(self):
            self.cfg = SimpleNamespace(
                generation=SimpleNamespace(
                    top_p=1.0,
                    sampling_temperature=None,
                    sampling_top_p=None,
                    sampling_top_k=None,
                    sampling_min_p=None,
                    sampling_presence_penalty=None,
                    sampling_repetition_penalty=None,
                )
            )
            self.responses = [
                "hello world",
                '{"pass": true, "semantic_similarity": 0.41, "reason_code": "pass", "notes": "partial"}',
            ]

        def complete(self, **kwargs):
            return self.responses.pop(0)

    teacher = _FakeTeacher()
    cfg = FiltersConfig()
    cfg.llm_judge.round_trip.enabled = True
    cfg.llm_judge.round_trip.min_semantic_similarity = 0.8
    decision = apply_llm_judge_filter(
        teacher=teacher,  # type: ignore[arg-type]
        source_lang="English",
        target_lang="Korean",
        source_text="hello world",
        target_text="안녕하세요 세계",
        cfg=cfg,
    )

    assert not decision.passed
    assert decision.reason_code == "round_trip_similarity_low"
