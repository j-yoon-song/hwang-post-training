#!/usr/bin/env bash
set -euo pipefail

# Run from gemma27b_sft directory.
# Optional:
#   MODEL_DIR=/path/to/checkpoint ./scripts/sample_infer.sh
#   SRC_TEXT="..." ./scripts/sample_infer.sh
#   CONFIG_PATH=configs/train_8xh100_fsdp.yaml ./scripts/sample_infer.sh

MODEL_DIR="${MODEL_DIR:-}"
if [[ -z "${MODEL_DIR}" ]]; then
  MODEL_DIR="$(ls -dt ../outputs/gemma3-27b-it-sft-fsdp/checkpoint-* 2>/dev/null | head -1 || true)"
fi
if [[ -z "${MODEL_DIR}" ]]; then
  MODEL_DIR="../outputs/gemma3-27b-it-sft-fsdp"
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Model directory not found: ${MODEL_DIR}" >&2
  exit 1
fi

CONFIG_PATH="${CONFIG_PATH:-configs/train_8xh100_fsdp.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

SRC_TEXT="${SRC_TEXT:-The weather is lovely today. Let us go for a walk by the river.}"

echo "MODEL_DIR=${MODEL_DIR}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "SRC_TEXT=${SRC_TEXT}"

MODEL_DIR="${MODEL_DIR}" CONFIG_PATH="${CONFIG_PATH}" SRC_TEXT="${SRC_TEXT}" python - <<'PY'
import os
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = os.environ["MODEL_DIR"]
config_path = os.environ["CONFIG_PATH"]
src = os.environ["SRC_TEXT"]

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
model.eval()

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}

prompt_template = str(data_cfg.get("prompt_template", "{text}"))
src_lang_code = str(os.environ.get("SRC_LANG_CODE") or data_cfg.get("source_lang_code", "en")).strip()
tgt_lang_code = str(os.environ.get("TGT_LANG_CODE") or data_cfg.get("target_lang_code", "ko")).strip()
src_lang_name_cfg = str(data_cfg.get("source_lang_name", "auto")).strip()
tgt_lang_name_cfg = str(data_cfg.get("target_lang_name", "auto")).strip()
src_lang_name = str(os.environ.get("SRC_LANG_NAME") or src_lang_name_cfg).strip()
tgt_lang_name = str(os.environ.get("TGT_LANG_NAME") or tgt_lang_name_cfg).strip()

try:
    from gemma27b_sft.data import _normalize_code, _resolve_language_name  # pylint: disable=import-error
except Exception:  # pylint: disable=broad-except
    def _normalize_code(code: str) -> str:
        return code.strip().replace("_", "-").lower()

    def _resolve_language_name(name: str, code: str) -> str:
        if name and name.strip() and name.strip().lower() != "auto":
            return name.strip()
        return code

src_lang_code = _normalize_code(src_lang_code)
tgt_lang_code = _normalize_code(tgt_lang_code)
source_lang = _resolve_language_name(src_lang_name, src_lang_code)
target_lang = _resolve_language_name(tgt_lang_name, tgt_lang_code)

try:
    prompt = prompt_template.format(
        source_lang=source_lang,
        src_lang_code=src_lang_code,
        target_lang=target_lang,
        tgt_lang_code=tgt_lang_code,
        text=src,
    )
except KeyError as exc:
    missing = exc.args[0]
    raise ValueError(
        f"Unknown placeholder in prompt_template: {missing}. "
        "Allowed: source_lang, src_lang_code, target_lang, tgt_lang_code, text."
    ) from exc

with torch.inference_mode():
    if getattr(tokenizer, "chat_template", None):
        chat_inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(chat_inputs, torch.Tensor):
            input_ids = chat_inputs.to(model.device)
            model_inputs = {"input_ids": input_ids}
        else:
            # Some tokenizer versions return BatchEncoding here.
            chat_inputs = chat_inputs.to(model.device)
            if "input_ids" not in chat_inputs:
                raise ValueError("chat template output has no input_ids")
            model_inputs = dict(chat_inputs)
            input_ids = model_inputs["input_ids"]
        out = model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
        gen = out[0][input_ids.shape[1] :]
    else:
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=256, do_sample=False)
        gen = out[0][enc["input_ids"].shape[1] :]

print("\n=== Model Output ===")
print(tokenizer.decode(gen, skip_special_tokens=True).strip())
PY
