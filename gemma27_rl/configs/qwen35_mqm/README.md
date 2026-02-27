# Qwen3.5-27B + MetricX-XXL + GEMBA-MQM (OpenAI-Compatible)

This folder contains isolated configs for RL experiments with:
- policy: `Qwen/Qwen3.5-27B-Instruct`
- sequence rewards: `MetricX-24-XXL` + `GEMBA-MQM`
- **xCOMET disabled**

MQM prompt/scoring behavior is aligned with:
`/home/seungyoonee/initial_translation/configs/metrics/gemba_mqm.yaml`
and the message/scoring logic in:
`/home/seungyoonee/initial_translation/evalmt/metrics/gemba_mqm_metric.py`

## Configs

- `train_wmt24pp_enko_qwen35_27b_mqm_dev4gpu.yaml`
  - environment check / coding run
  - policy on 3 GPUs (`[0,1,2]`), MetricX on 1 GPU (`cuda:3`)
  - reference model enabled on CPU (`reference_device: cpu`)

- `train_wmt24pp_enko_qwen35_27b_mqm_scale8gpu.yaml`
  - scale-up run
  - policy on 4 GPUs (`[0..3]`), reference on 3 GPUs (`[4..6]`), MetricX on 1 GPU (`cuda:7`)
  - larger sampling and training batch settings

## Run

```bash
export HF_TOKEN=...
export OPENAI_API_KEY=...
python -m gemma27_rl.cli --config configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_mqm_dev4gpu.yaml
```

or

```bash
python -m gemma27_rl.cli --config configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_mqm_scale8gpu.yaml
```
