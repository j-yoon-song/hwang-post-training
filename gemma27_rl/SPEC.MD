# SPEC.MD — TranslateGemma(2601.09012) 방식 RL (MetricX-QE + XCOMET-XL)

## 0) 목표 / 범위

### 목표
- 이미 SFT 완료된 번역 모델(Policy)을 **TranslateGemma(2601.09012)** 리포트에 나온 RL 방식으로 추가 학습한다.
- 우선은 **소규모 토이 실험/검증**이 목적이며, reward는 다음 2개만 사용한다.
  - **MetricX-QE**
  - **Unbabel/XCOMET-XL** (sentence score + error spans 기반 token-level signal)

### Non-goals (이번 스펙에서 제외)
- AutoMQM/Gemma-AutoMQM, chrF, Naturalness autorater, Generalist RM 등 다른 reward 추가
- 대규모 분산 학습/복잡한 데이터 파이프라인/대규모 리더보드 재현
- SFT 데이터 생성/정제(이미 SFT는 끝났다는 전제)

---

## 1) 용어 / 기본 단위

- **prompt**: 번역 지시 + source text가 들어있는 입력.
- **completion**: 모델이 생성한 번역 출력(학습 대상).
- **token-level advantage**: completion 토큰별로 다른 advantage 값을 갖는 형태(스팬 기반 신호를 토큰에 매핑).
- **sequence-level reward**: 문장 단위(샘플 단위) 하나의 reward 스칼라.
- **reward-to-go broadcast**: sequence-level reward를 completion의 모든 토큰에 동일하게 복제하여 per-token reward/advantage로 만드는 것.
- **batch-normalization (of advantages)**: 배치 내 모든 유효 토큰에 대해 advantage를 평균 0, 표준편차 1로 정규화.

---

## 2) 입력 데이터(토이/확장 가능)

### 최소 요구 필드
각 샘플은 최소 다음 필드를 가져야 한다(형식은 JSONL/Parquet/HF dataset 등 자유. Codex가 프로젝트에 맞게 매핑).
- `id: str|int` (옵션이지만 권장)
- `src_text: str` (source 문장)
- `src_lang: str` (예: "English")
- `tgt_lang: str` (예: "Korean")

### 선택 필드
- `src_lang_code: str` (예: "en-US")
- `tgt_lang_code: str` (예: "ko-KR")
- `ref_text: str` (참고 번역; **reward에는 사용하지 않음**, 디버깅/리포팅용)

### 데이터 로더 요구 기능
- `load_examples(...) -> List[Example]`:
  - small split(dev/test) 지원
  - 샘플 수 `--limit` 같은 옵션으로 쉽게 줄일 수 있어야 함(토이 실험)

---

## 3) 프롬프트 구성(논문 방식)

### 기본 프롬프트 템플릿
- TranslateGemma 리포트의 “preferred prompt”를 기본으로 제공한다.
- `format_translation_prompt(example) -> str` 는 아래 템플릿을 채운 문자열을 반환해야 한다.

템플릿(개행/문구는 동일 유지 권장):
You are a professional {source_lang} ({src_lang_code}) to {target_lang} ({tgt_lang_code}) translator. Your goal is to accurately convey the meaning and nuances of the original {source_lang} text while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities. Produce only the {target_lang} translation, without any additional explanations or commentary. Please translate the following {source_lang} text into {target_lang}:\n\n{text}
- `src_lang_code/tgt_lang_code`가 없으면 빈 문자열 또는 `src_lang/tgt_lang`로 대체 가능.
- completion은 “번역문만” 나오도록 유도하지만, 실제 출력이 어길 수 있으니 후처리 훅을 제공:
  - `postprocess_translation(raw_text: str) -> str` (기본: strip만 수행; 과도한 정규화는 금지)

---

## 4) Rollout(생성) 수집

### 요구 기능: generation + logprob 저장
- `generate_rollouts(examples, policy_model, tokenizer, gen_cfg) -> List[Rollout]`

각 `Rollout`은 최소 다음을 포함:
- `example_id`
- `prompt_text`
- `prompt_input_ids`
- `completion_text` (후처리 이후 텍스트)
- `completion_token_ids`
- `old_logprobs` (completion 각 토큰의 log π_old(a_t|s_t))
- `ref_logprobs` (optional, KL용. reference policy의 logprob)
- `token_char_offsets` (completion 토큰별 (char_start, char_end) 오프셋)

### 토큰 오프셋 계산 요구
- `compute_token_char_offsets(tokenizer, completion_token_ids, decode_cfg) -> List[(start,end)]`
  - 가능한 한 **정확히 completion_text와 동일한 문자열을 재구성**하면서 토큰별 char range를 기록해야 함.
  - 권장 구현: `tokenizer.decode([tok_id], clean_up_tokenization_spaces=False, skip_special_tokens=False)`를 토큰 단위로 누적하여 텍스트/오프셋 생성.
  - 재구성 문자열이 completion_text와 불일치하면:
    - (가능하면) fast tokenizer의 `return_offsets_mapping=True` 경로로 fallback
    - 둘 다 안 되면 best-effort(디버그 로그 필수)

---

## 5) Reward 모델 래퍼

reward 모델은 서로 다른 구현체이지만 공통 인터페이스를 갖도록 한다.

### 공통 인터페이스(권장)
- `score_batch(samples: List[SampleForScoring], cfg) -> RewardOutput`

여기서 `SampleForScoring` 최소 필드:
- `src: str`
- `mt: str`
- `ref: Optional[str]` (QE면 None 또는 "")

`RewardOutput` 최소 필드:
- `sequence_scores: List[float]` (샘플별 스칼라)
- `metadata: Any` (스팬 등 추가정보)

---

## 6) MetricX-QE reward

### 목적
- source + model translation(mt)을 입력으로 받아 **에러 스코어(낮을수록 좋음)**를 출력하는 learned metric.
- RL에서는 “높을수록 좋음” reward로 변환해 사용한다.

### 구현 요구
- `MetricXQEScorer`:
  - 모델 이름(예: metricx-24-hybrid-xxl/large 등)과 tokenizer를 로드해 배치 inference
  - 입력 문자열 포맷:
    - QE 모드: `"source: {src} candidate: {mt}"`
    - (참고: reference 기반 모드는 `"source: ... candidate: ... reference: ..."` 이지만 이번 스펙은 QE만)
  - 출력: `metricx_score` (float, 낮을수록 좋음; 통상 0~25 범위)

### Reward 변환(TranslateGemma 방식)
- `metricx_reward = metricx_offset - metricx_score`
  - 기본값: `metricx_offset = 5.0`
- 이 reward는 sequence-level reward로 사용한다.

### 효율 요구
- `batch_size`, `device`, `dtype` 옵션 제공
- 입력 길이 제한(예: max_input_length) 처리:
  - 너무 길면 truncate 또는 skip(정책 선택 가능, 로그 필수)

---

## 7) XCOMET-XL reward + error spans

### 목적
- sentence-level quality score + error span(설명 가능한 스팬)을 제공하는 metric.
- 이번 스펙에서는:
  1) sentence-level score를 선택적으로 sequence reward로 사용하고
  2) **error spans를 token-level advantage로 변환**하는 것이 핵심이다.

### 구현 요구
- `XCometXLScorer`:
  - `unbabel-comet` 라이브러리로 모델 로드/추론
  - 입력 데이터 포맷:
    - 기본: `{ "src": ..., "mt": ..., "ref": ... }`
    - QE 사용 시: `"ref"`를 **omit**하거나 `""`로 제공 가능해야 함
  - 출력:
    - `xcomet_score` (샘플별 float; 보통 0~1 범위로 취급)
    - `error_spans`: 스팬 리스트(최소 필드 요구)
      - `start: int` (mt 텍스트 기준 char start)
      - `end: int` (mt 텍스트 기준 char end)
      - `severity: str` (MINOR/MAJOR/CRITICAL 등)
      - (있으면) `confidence: float`

---

## 8) Error spans → token-level rewards(핵심)

### 목표
- xCOMET의 `error_spans`를 completion 토큰 인덱스에 매핑하여 `token_rewards[t]` 생성.
- 이후 sequence reward 기반 advantage와 **additive**로 합친다.

### 필수 함수
- `spans_to_token_rewards(mt_text, token_char_offsets, error_spans, severity_weights, overlap_policy) -> List[float]`
  - 반환 길이 = completion 토큰 수
  - 토큰 t의 char range가 스팬과 겹치면 패널티 부여

### overlap_policy
- `"any_overlap"`: 토큰 범위와 스팬이 1 char이라도 겹치면 패널티 적용 (기본 권장)
- `"majority_overlap"`: 겹침 비율이 일정 threshold 이상일 때만 적용 (옵션)

### severity_weights (기본값)
- 기본 가중치는 설정 가능해야 하며, 최소 다음 키를 지원:
  - `MINOR`
  - `MAJOR`
  - `CRITICAL`
- 권장 기본값(초기값):
  - `MINOR: -1.0`
  - `MAJOR: -5.0`
  - `CRITICAL: -10.0`
- `confidence`가 제공되면 옵션으로 `penalty *= confidence` 적용 가능하도록 한다(기본 off).

### 여러 스팬이 겹치는 경우
- 기본 정책: 누적 합(sum)
- 옵션: 최소값(min) / 최대 패널티(max) 등 선택 가능

---

## 9) Reward 결합 → per-token advantage 생성(TranslateGemma 방식)

### 9.1 sequence-level rewards
- 최소 구성:
  - `r_seq = w_metricx * metricx_reward`
- 옵션으로 xCOMET sentence score도 사용 가능:
  - `r_seq += w_xcomet_seq * (xcomet_score * xcomet_seq_scale)`
  - 기본은 `w_xcomet_seq = 0`(토이 실험에서는 우선 token-level만 쓰고 싶으면 0)

### 9.2 reward-to-go broadcast (sequence → token)
- completion 토큰 길이가 T일 때:
  - `A_seq[t] = r_seq` for all t in [0..T-1]
- (여기서 “reward-to-go broadcast”는 sequence reward를 모든 토큰에 균일하게 복제한다는 의미)

### 9.3 token-level signal 추가
- `A_tok[t] = token_rewards[t]`  (xCOMET spans 기반)
- 결합:
  - `A_raw[t] = A_seq[t] + A_tok[t]`  (additive)

### 9.4 batch-normalize advantages
- 배치 내 모든 샘플의 completion 토큰을 flatten하고, padding 제외한 유효 토큰 마스크에 대해:
  - `A = (A_raw - mean(A_raw)) / (std(A_raw) + eps)`
- eps 기본값: `1e-8`
- 반환: `A_norm` (shape = [batch, T_i])

---

## 10) RL 업데이트(정책 최적화)

### 요구사항(알고리즘은 구현체 선택 가능)
- 어떤 RL 알고리즘을 쓰든 **“토큰별 advantage”를 받아서 학습**할 수 있어야 한다.
- 최소 구현 목표는 다음 중 하나:
  1) PPO-style token-level objective (권장)
  2) REINFORCE-style token-level objective (간단)

### (권장) PPO-style 요구 기능
- `update_policy(rollouts, advantages, optimizer, ppo_cfg) -> TrainStats`

필수 연산:
- `new_logprobs` 재계산 (현재 policy로 prompt+completion teacher-forcing forward)
- ratio: `r_t = exp(new_logprobs_t - old_logprobs_t)`
- clipped objective:
  - `L = -mean( min(r_t*A_t, clip(r_t,1-eps,1+eps)*A_t) )`
- 옵션(강권):
  - KL regularization to reference policy:
    - `kl_t = (new_logprobs_t - ref_logprobs_t)`
    - `L += kl_coef * mean(kl_t)`
  - entropy bonus (optional)

### reference policy(KL) 요구
- reference policy는 일반적으로 “RL 시작 시점의 SFT 체크포인트”를 freeze한 모델.
- `ref_logprobs`는 rollout 시 저장하거나 update 시 재계산해도 됨(토이에서는 저장 권장).

---

## 11) 소규모 검증(토이 실험) 기능 요구

### 11.1 Metric-only 평가(학습 없이)
- `evaluate_on_dataset(examples, policy_model, ...) -> Report`
  - 모델 생성 → MetricX-QE score/reward 계산
  - XCOMET-XL score + spans 통계
  - 출력 리포트 최소 포함:
    - 평균 MetricX score(↓), 평균 MetricX reward(↑)
    - 평균 xCOMET score(↑)
    - span 개수 평균, severity별 카운트 분포
    - 평균 completion length

### 11.2 RL 토이 런
- `run_toy_rl(train_examples, eval_examples, cfg) -> artifacts`
  - 1~N 스텝만 돌아도 end-to-end가 확인되게 구성
  - N이 작아도(예: 5~50 updates) 지표가 로그로 남아야 함
  - 중간 체크포인트 저장/로드 가능

---

## 12) 관측/로그(최소)

각 update마다 다음을 기록(콘솔 + jsonl 등):
- rollout:
  - 평균 completion 길이
- reward:
  - metricx_score mean/std, metricx_reward mean/std
  - xcomet_score mean/std
  - token_rewards mean/std, 0이 아닌 토큰 비율
  - severity별 span 개수(mean)
- advantage:
  - A_raw mean/std, A_norm mean/std(정상적으로 mean≈0 std≈1인지)
- optimization:
  - policy loss
  - approx_kl (new-old), clip fraction(PPO면)
  - KL to reference mean(사용 시)

---

## 13) 테스트(필수)

### 단위 테스트
1) MetricX 입력 포맷:
   - QE 포맷 `"source: ... candidate: ..."` 생성이 정확한지
2) MetricX reward 변환:
   - `reward = 5.0 - score`가 적용되는지
3) token_char_offsets:
   - 토큰 누적 decode로 만든 텍스트가 completion_text와 일치(또는 허용 오차 정책)
4) spans_to_token_rewards:
   - 특정 substring span이 주어졌을 때 해당 토큰만 패널티가 들어가는지
5) advantage 결합/정규화:
   - A_norm의 mean≈0, std≈1 (마스크 고려)
6) PPO/REINFORCE update가 NaN 없이 1 step 업데이트되는지(smoke)

### 통합 테스트(권장)
- 샘플 8~32개로:
  - 생성 → reward 계산 → advantage → 1 update → 다시 평가
  - 전체 파이프라인이 1 커맨드로 실행되는지

---

## 14) 설정(Config) 요구

최소 아래 항목은 cfg로 제어 가능해야 함(형식은 자유):
- generation:
  - max_new_tokens, temperature, top_p/top_k, num_samples_per_prompt
- reward:
  - metricx_model_name, metricx_tokenizer_name, metricx_offset(=5.0), w_metricx
  - xcomet_model_name(=Unbabel/XCOMET-XL), w_xcomet_seq, xcomet_seq_scale
  - severity_weights(MINOR/MAJOR/CRITICAL), overlap_policy, use_confidence
- RL:
  - algorithm(ppo/reinforce), lr, batch_size, grad_accum, clip_eps, kl_coef, entropy_coef
- eval:
  - eval_every_n_updates, eval_limit
- misc:
  - seed, device, dtype, caching on/off

---

## 15) 성능/안정성 고려(최소)

- reward 모델은 매우 무거울 수 있으므로:
  - batching 필수
  - GPU 사용 옵션 필수
  - (가능하면) 동일 (src,mt) 쌍에 대해 score 캐싱 옵션 제공
- 길이 제한:
  - reward 모델 입력 길이 초과 시 정책(잘라내기/스킵/패널티) 명시적으로 선택 가능해야 함
### 근거/참고(이 SPEC 작성에 사용한 핵심 출처) * TranslateGemma RL에서 MetricX를 QE로 쓰고(빈 reference), MetricX reward를 5.0 - score로 뒤집어 사용했다는 점, 그리고 span 기반 token-level advantage를 sequence-level advantage에 **additive**로 더하고 **batch-normalize**했다는 점. ([arXiv][1]) * TranslateGemma RL의 “sequence-level reward를 reward-to-go로 간주하여 모든 토큰에 균일 broadcast”한다는 설명(그림 2 캡션). ([arXiv][1]) * TranslateGemma에서 사용한 번역 프롬프트 템플릿(“preferred prompt” figure). ([arXiv][1]) * MetricX-24는 hybrid이며, score가 [0,25]로 clip되고, QE inference를 지원한다는 설명. ([Hugging Face][2]) * MetricX-24 QE 입력 문자열 포맷이 "source: ... candidate: ..." 임을 보여주는 공식 predict 스크립트. ([GitHub][3]) * XCOMET-XL의 unbabel-comet>=2.2.0 요구, model.predict(...) 출력에 metadata.error_spans가 포함된다는 점(모델 카드). ([Hugging Face][4]) * xCOMET은 단일 모델로 여러 평가 모드(quality estimation 포함)를 지원한다는 점. ([ACL Anthology][5]) * xCOMET 논문에서 sentence-level score가 “보통 0~1 범위”로 설명되는 부분과, span severity 기반 penalty를 1/5/10으로 합산하는 공식(e(S)=cMIN+5cMAJ+10cCRIT). * (레퍼런스 없이) “refs를 제거할 수 있다”는 실사용 예시 노트(노트북/gist). ([Gist][6]) [1]: https://arxiv.org/pdf/2601.09012 "https://arxiv.org/pdf/2601.09012" [2]: https://huggingface.co/google/metricx-24-hybrid-large-v2p6/blame/50f9b84035febefbed202f783f00c6154a5b2871/README.md "https://huggingface.co/google/metricx-24-hybrid-large-v2p6/blame/50f9b84035febefbed202f783f00c6154a5b2871/README.md" [3]: https://raw.githubusercontent.com/google-research/metricx/main/metricx24/predict.py "https://raw.githubusercontent.com/google-research/metricx/main/metricx24/predict.py" [4]: https://huggingface.co/Unbabel/XCOMET-XL "https://huggingface.co/Unbabel/XCOMET-XL" [5]: https://aclanthology.org/2024.tacl-1.54.pdf "https://aclanthology.org/2024.tacl-1.54.pdf" [6]: https://gist.github.com/mtreviso/b618b499bc6de0414a3e11157e91cf02 "https://gist.github.com/mtreviso/b618b499bc6de0414a3e11157e91cf02"