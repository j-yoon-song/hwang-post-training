# Gemma 27B RL Post-Training (GRPO)

`gemma27b_sft`로 학습된 체크포인트를 시작점으로, `SPEC.MD` 요구사항에 맞춘
TranslateGemma 스타일 RL post-training 파이프라인입니다.

구현 포함 항목:
- rollout 수집 (completion + old/ref logprobs + token char offsets)
- MetricX-QE sequence reward (`5.0 - score`)
- XCOMET-XL sentence score + error spans
- error span -> token reward 매핑
- sequence reward broadcast + token reward additive + batch normalize
- GRPO/PPO-clip 스타일 업데이트 (value head 없음)
- metric-only eval 및 toy RL loop

## 1) 설치

```bash
cd gemma27_rl
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

## 2) 설정

기본 예시: `configs/train_toy.yaml`
WMT24pp 빠른 테스트 예시: `configs/train_wmt24pp_enko_toy.yaml`

핵심값:
- `model.policy_name_or_path`: SFT 결과 체크포인트 경로
- `model.policy_gpu_ids`: policy 모델에 할당할 GPU 인덱스 목록 (예: `[0,1,2]`)
- `model.reference_gpu_ids`: reference 모델에 할당할 GPU 인덱스 목록 (예: `[3,4,5]`)
- `data.train_file`: RL 학습용 JSONL/JSON/Parquet
- 또는 `data.hf_dataset_name` + `data.hf_dataset_config_name` + `data.hf_train_split`
- `generation.num_samples_per_prompt`: GRPO group 크기
- `reward.metricx.*`, `reward.xcomet.*`
- `rl.*` (clip, kl, batch, updates)
- `misc.huggingface_cache_dir`: HF 캐시 루트 (예: `/media/sdd3`)
- `misc.huggingface_token`: (권장 비활성) 직접 토큰 입력값
- `misc.huggingface_token_env`: 토큰을 읽을 환경변수 이름 (기본 `HF_TOKEN`)

GPU 배치(자동):
- 기본값(`misc.device: cuda`, `reward.metricx.device: cuda`, `reward.xcomet.device: cuda`)이면
  실행 시 자동으로 `policy -> cuda:0`, `metricx -> cuda:1`, `xcomet -> cuda:2` 순으로
  가능한 한 서로 다른 GPU를 배정합니다.
- GPU 개수가 부족하면 가능한 범위에서 배정하고 경고 로그를 출력합니다.
- `xcomet`은 Lightning `Trainer` 재생성을 피하고 모델을 메모리에 상주시켜,
  반복 스코어링 시 초기화 오버헤드를 줄입니다.

GPU 배치(명시적 8-GPU 분할):
- `model.policy_gpu_ids`/`model.reference_gpu_ids`를 설정하면 자동 배치보다 우선합니다.
- 이 경우 policy/reference는 지정한 GPU 목록에 `device_map=auto`로 로드됩니다.
- MetricX/XCOMET은 `reward.metricx.device`, `reward.xcomet.device`로 단일 GPU를 직접 지정하세요.
- 예시(3/3/1/1): `policy=[0,1,2]`, `reference=[3,4,5]`, `metricx=cuda:6`, `xcomet=cuda:7`.

토큰 사용 권장 방식:

```bash
export HF_TOKEN=...
python -m gemma27_rl.cli --config configs/train_toy.yaml
```

## 3) 실행

학습:

```bash
python -m gemma27_rl.cli --config configs/train_toy.yaml
```

평가만:

```bash
python -m gemma27_rl.cli --config configs/train_toy.yaml --eval-only
```

로그/체크포인트:
- `logging.output_dir/resolved_config.yaml`
- `logging.output_dir/train_log.jsonl`
- `logging.output_dir/train_rollouts.jsonl` (`logging.save_rollouts: true`일 때)
- `logging.output_dir/eval_outputs.jsonl` (`logging.save_eval_outputs: true`일 때)
- `logging.output_dir/checkpoint-*`
- `logging.output_dir/final`
