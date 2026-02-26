# Qwen3.5-35B-A3B MoE RL Post-Training (GSPO/GRPO)

`gemma27_rl`을 기반으로, MoE (Mixture of Experts) 모델에 최적화된
RL post-training 파이프라인입니다.

MoE 모델에 vanilla GRPO를 적용하면 expert-activation volatility로 학습이 불안정해지는
문제를 해결하기 위해 다음 기능을 추가했습니다:

- **GSPO** (Group Sequence Policy Optimization) — 시퀀스 수준 IS ratio로 MoE ratio 폭발 방지
- **Auxiliary load balancing loss** — `outputs.aux_loss`로 expert collapse 방지
- **Expert utilization monitoring** — router logits에서 utilization entropy/max/min 추적
- **Gradient checkpointing** — MoE 모델 메모리 최적화
- **분산학습** — DeepSpeed ZeRO-2/3, FSDP, DDP (opt-in, 비활성 시 zero overhead)

## 코드 파일 구조

```
qwen3.5-35b-a3b/
├── qwen3.5-35b-a3b/           # Python 패키지
│   ├── __init__.py             # Public API 내보내기
│   ├── cli.py                  # CLI 진입점 (--config, --eval-only, --local_rank)
│   ├── config.py               # 설정 dataclass (ModelConfig, RLConfig, DistributedConfig 등)
│   ├── types.py                # 핵심 타입 (Example, Rollout, TrainStats, DistributedContext)
│   ├── data.py                 # 데이터 로딩 (JSONL/JSON/Parquet/HF datasets)
│   ├── prompting.py            # 번역 프롬프트 템플릿 및 후처리
│   ├── rollout.py              # Rollout 생성 + logprob 계산 + token offset 매핑
│   ├── rewards.py              # MetricX-QE + xCOMET-XL 보상 스코어링
│   ├── metricx_model.py        # MT5 regression 모델 (MetricX용)
│   ├── advantage.py            # Advantage 계산 (token-level + sequence-level GSPO)
│   ├── grpo.py                 # Policy update (GSPO/GRPO/REINFORCE + aux loss + expert monitoring)
│   ├── eval.py                 # 평가 (rollout 생성 → 보상 → 통계)
│   ├── trainer.py              # 학습 루프 (모델 로딩, 분산학습 래핑, 체크포인트)
│   └── utils.py                # 유틸리티 (device, dtype, seed, HF cache, 분산 init/cleanup)
├── configs/
│   └── train_toy_gspo.yaml     # 예시 설정 (GSPO + MoE + 분산학습)
├── tests/                      # pytest 테스트
├── pyproject.toml
└── README.md
```

## 1) 설치

```bash
cd qwen3.5-35b-a3b
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
# 보상 모델 사용 시:
uv pip install -e ".[reward]"
# 테스트:
uv pip install -e ".[test]"
```

## 2) 설정

기본 예시: `configs/train_toy_gspo.yaml`

핵심값:
- `model.policy_name_or_path`: 정책 모델 경로 (기본: `Qwen/Qwen3.5-35B-A3B`)
- `model.gradient_checkpointing`: MoE 메모리 최적화 활성화
- `data.train_file`: 학습용 JSONL/JSON/Parquet
- `generation.num_samples_per_prompt`: GRPO/GSPO group 크기
- `rl.algorithm`: `gspo` | `grpo` | `reinforce`
- `rl.output_router_logits`: MoE router logits 출력 (aux loss/expert monitoring용)
- `rl.router_aux_loss_coef`: auxiliary load balancing loss 계수
- `rl.monitor_expert_utilization`: expert utilization entropy/max/min 모니터링
- `reward.metricx.*`, `reward.xcomet.*`: 보상 모델 설정
- `misc.huggingface_cache_dir`: HF 캐시 루트
- `misc.huggingface_token_env`: 토큰을 읽을 환경변수 (기본 `HF_TOKEN`)

GPU 배치(자동):
- `misc.device: cuda`, `reward.metricx.device: cuda`, `reward.xcomet.device: cuda`이면
  자동으로 서로 다른 GPU 배정
- `model.policy_gpu_ids`/`model.reference_gpu_ids`로 명시적 분할 가능

## 3) 실행 — Single Node

학습:

```bash
python -m qwen3.5-35b-a3b --config configs/train_toy_gspo.yaml
```

평가만:

```bash
python -m qwen3.5-35b-a3b --config configs/train_toy_gspo.yaml --eval-only
```

## 4) 실행 — 분산학습

`distributed.enabled: false` (기본값)일 때 분산학습 코드는 완전히 비활성화되어
single node 성능에 영향을 주지 않습니다.

### 활성화 방법

`configs/train_toy_gspo.yaml`에서:

```yaml
distributed:
  enabled: true
  backend: deepspeed        # deepspeed | fsdp | ddp
  zero_stage: 2             # 0, 1, 2, 3 (DeepSpeed only)
  offload_optimizer: false  # optimizer state → CPU
  offload_param: false      # parameters → CPU (ZeRO-3 only)
```

### DeepSpeed로 실행

```bash
deepspeed --num_gpus=4 -m qwen3.5-35b-a3b --config configs/train_toy_gspo.yaml
```

### torchrun으로 실행

```bash
torchrun --nproc_per_node=4 -m qwen3.5-35b-a3b --config configs/train_toy_gspo.yaml
```

### 분산학습 옵션

| 설정 | 설명 | 기본값 |
|------|------|--------|
| `distributed.enabled` | 분산학습 활성화 | `false` |
| `distributed.backend` | 백엔드 선택 | `deepspeed` |
| `distributed.zero_stage` | ZeRO 최적화 단계 (DeepSpeed) | `2` |
| `distributed.offload_optimizer` | optimizer state를 CPU로 offload | `false` |
| `distributed.offload_param` | parameter를 CPU로 offload (ZeRO-3) | `false` |
| `distributed.deepspeed_config_path` | 외부 DeepSpeed JSON config override | `null` |
| `distributed.fsdp_sharding_strategy` | FSDP 샤딩 전략 | `FULL_SHARD` |

### 분산학습 동작 방식

- 각 rank가 학습 데이터를 자동 분할 (shard)
- 로깅, 체크포인트 저장은 rank 0만 수행
- 보상 모델 (MetricX/xCOMET)은 각 rank가 독립 로드
- `DistributedContext`가 backward/step/zero_grad를 추상화하여 DeepSpeed/FSDP/DDP 차이를 숨김

## 5) 알고리즘 비교

| 알고리즘 | IS Ratio 수준 | MoE 호환성 | 설정 |
|----------|--------------|-----------|------|
| **GSPO** | 시퀀스 수준 | 최적 (ratio 안정) | `rl.algorithm: gspo` |
| **GRPO** | 토큰 수준 | 불안정 가능 (expert volatility) | `rl.algorithm: grpo` |
| **REINFORCE** | 없음 (직접 logprob) | 안정적 | `rl.algorithm: reinforce` |

GSPO 핵심:
```
GRPO: ratio_t = exp(new_lp_t - old_lp_t)           # 토큰별
GSPO: ratio_seq = exp(mean(new_lp) - mean(old_lp))  # 시퀀스별
```

## 6) 출력물

- `logging.output_dir/resolved_config.yaml` — 해석된 설정
- `logging.output_dir/train_log.jsonl` — 학습 로그 (loss, reward, MoE metrics)
- `logging.output_dir/train_rollouts.jsonl` — rollout 상세 (`save_rollouts: true`)
- `logging.output_dir/eval_outputs.jsonl` — 평가 출력 (`save_eval_outputs: true`)
- `logging.output_dir/checkpoint-*` — 주기적 체크포인트
- `logging.output_dir/best` — 최고 평가 점수 모델
- `logging.output_dir/final` — 최종 모델

## 7) 테스트

```bash
cd qwen3.5-35b-a3b
pytest tests/ -v
```
