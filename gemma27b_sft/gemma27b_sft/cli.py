from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import inspect
import logging
import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Adafactor,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from .config import SFTConfig, compute_gradient_accumulation_steps, dump_config, load_config
from .data import build_datasets


logger = logging.getLogger(__name__)


class FixedAdafactorTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = Adafactor(
                params,
                lr=self.args.learning_rate,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            )
        return self.optimizer

    def _save_optimizer_and_scheduler(self, *args, **kwargs):
        try:
            return super()._save_optimizer_and_scheduler(*args, **kwargs)
        except AssertionError as exc:
            # FSDP + Adafactor can fail to gather optimizer state due to per-rank
            # RMS state divergence. Keep model checkpointing progressing.
            fsdp_enabled = bool(getattr(self.args, "fsdp", None))
            err = str(exc)
            known_mismatch = "different values for RMS" in err or (
                "different values for" in err and "RMS" in err
            )
            if fsdp_enabled and known_mismatch:
                logger.warning(
                    "Skipping optimizer/scheduler state save due to known FSDP+Adafactor "
                    "optimizer-state mismatch: %s",
                    err,
                )
                return None
            raise


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _freeze_embeddings(model: AutoModelForCausalLM, freeze_output_embeddings: bool) -> tuple[int, int]:
    frozen_params = 0
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        for param in input_embeddings.parameters():
            param.requires_grad = False
            frozen_params += param.numel()

    output_embeddings = model.get_output_embeddings()
    if freeze_output_embeddings and output_embeddings is not None and output_embeddings is not input_embeddings:
        for param in output_embeddings.parameters():
            param.requires_grad = False
            frozen_params += param.numel()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Parameter summary total=%s trainable=%s frozen=%s",
        total,
        trainable,
        frozen_params,
    )
    return trainable, frozen_params


def _is_flash_attn_available() -> bool:
    return torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None


def _resolve_attn_implementation(cfg: SFTConfig) -> str | None:
    requested = (cfg.model.attn_implementation or "auto").strip().lower()
    resolved: str | None
    if requested in {"", "auto"}:
        if _is_flash_attn_available():
            logger.info("Attention implementation resolved to flash_attention_2 (auto mode).")
            resolved = "flash_attention_2"
        else:
            logger.info("flash_attn unavailable; using sdpa attention implementation.")
            resolved = "sdpa"
    else:
        resolved = cfg.model.attn_implementation

    if (
        resolved == "flash_attention_2"
        and cfg.train.fsdp
        and cfg.train.fsdp_activation_checkpointing
    ):
        logger.warning(
            "Switching attention implementation to sdpa because "
            "flash_attention_2 + FSDP activation checkpointing can trigger "
            "torch.utils.checkpoint.CheckpointError (metadata mismatch) on current stacks."
        )
        resolved = "sdpa"
    return resolved


def _apply_runtime_compat_overrides(cfg: SFTConfig) -> None:
    model_name = cfg.model.name_or_path.lower()
    if "gemma-3" not in model_name or not cfg.train.fsdp:
        return

    requested_attn = (cfg.model.attn_implementation or "auto").strip().lower()
    if requested_attn == "eager":
        logger.warning(
            "attn_implementation=eager is unstable for Gemma 3 FSDP training "
            "(can trigger 1024/2047 attention-mask mismatch). Switching to auto."
        )
        cfg.model.attn_implementation = "auto"

    if cfg.train.fsdp_activation_checkpointing:
        logger.warning(
            "Disabling train.fsdp_activation_checkpointing for Gemma 3 FSDP run. "
            "Using HF gradient_checkpointing instead to avoid checkpoint/mask mismatch errors."
        )
        cfg.train.fsdp_activation_checkpointing = False
        if not cfg.train.gradient_checkpointing:
            cfg.train.gradient_checkpointing = True


def _is_gemma3_model_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return "gemma-3" in lowered or "gemma3" in lowered


def _ensure_gemma3_tokenizer_attrs(tokenizer) -> bool:
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    extra = dict(getattr(tokenizer, "extra_special_tokens", {}) or {})
    extra.update(dict(init_kwargs.get("extra_special_tokens") or {}))

    boi_token = (
        getattr(tokenizer, "boi_token", None)
        or extra.get("boi_token")
        or init_kwargs.get("boi_token")
        or "<start_of_image>"
    )
    eoi_token = (
        getattr(tokenizer, "eoi_token", None)
        or extra.get("eoi_token")
        or init_kwargs.get("eoi_token")
        or "<end_of_image>"
    )
    image_token = (
        getattr(tokenizer, "image_token", None)
        or extra.get("image_token")
        or init_kwargs.get("image_token")
        or "<image_soft_token>"
    )

    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if image_token_id is None or image_token_id == unk_id:
        return False

    setattr(tokenizer, "boi_token", boi_token)
    setattr(tokenizer, "eoi_token", eoi_token)
    setattr(tokenizer, "image_token", image_token)
    setattr(tokenizer, "image_token_id", int(image_token_id))

    extra.update(
        {
            "boi_token": boi_token,
            "eoi_token": eoi_token,
            "image_token": image_token,
        }
    )
    if isinstance(init_kwargs, dict):
        init_kwargs["boi_token"] = boi_token
        init_kwargs["eoi_token"] = eoi_token
        init_kwargs["image_token"] = image_token
        init_kwargs["extra_special_tokens"] = extra
        init_kwargs["processor_class"] = init_kwargs.get("processor_class") or "Gemma3Processor"

    return True


def _save_processor_artifacts(cfg: SFTConfig, tokenizer) -> None:
    if _is_gemma3_model_name(cfg.model.name_or_path):
        has_image_token_id = hasattr(tokenizer, "image_token_id")
        if not has_image_token_id:
            has_image_token_id = _ensure_gemma3_tokenizer_attrs(tokenizer)
        if not has_image_token_id:
            logger.warning(
                "Gemma3 tokenizer is missing image_token_id. "
                "Upgrade transformers (recommended >=4.50.0) in training/serving env."
            )

    try:
        processor = AutoProcessor.from_pretrained(
            cfg.model.name_or_path,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        processor.save_pretrained(cfg.train.output_dir)
        return
    except Exception as exc:
        if not _is_gemma3_model_name(cfg.model.name_or_path):
            raise RuntimeError(str(exc)) from exc
        logger.warning("AutoProcessor save failed for Gemma3, trying Gemma3Processor fallback: %s", exc)

    from transformers import AutoImageProcessor
    from transformers.models.gemma3.processing_gemma3 import Gemma3Processor

    _ensure_gemma3_tokenizer_attrs(tokenizer)
    image_processor = AutoImageProcessor.from_pretrained(
        cfg.model.name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    processor = Gemma3Processor(image_processor=image_processor, tokenizer=tokenizer)
    processor.save_pretrained(cfg.train.output_dir)


def _load_model(cfg: SFTConfig):
    model_kwargs: dict[str, object] = {
        "torch_dtype": torch.bfloat16 if cfg.train.bf16 else torch.float16,
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    resolved_attn = _resolve_attn_implementation(cfg)
    if resolved_attn:
        model_kwargs["attn_implementation"] = resolved_attn
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **model_kwargs)
        return model
    except Exception as exc:
        # Allow training to continue on environments without flash-attn.
        err = str(exc)
        if resolved_attn == "flash_attention_2" and ("flash_attn" in err or "FlashAttention2" in err):
            logger.warning(
                "flash_attention_2 is unavailable at runtime (%s). Falling back to sdpa.",
                err,
            )
            model_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **model_kwargs)
            return model
        raise


def _build_training_arguments(
    cfg: SFTConfig,
    grad_accum: int,
    has_eval: bool,
    hf_gradient_checkpointing: bool,
) -> TrainingArguments:
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters)

    kwargs: dict[str, object] = {
        "output_dir": str(output_dir),
        "seed": cfg.train.seed,
        "num_train_epochs": cfg.train.num_train_epochs,
        "max_steps": cfg.train.max_steps,
        "per_device_train_batch_size": cfg.train.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.train.per_device_eval_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": cfg.train.learning_rate,
        "lr_scheduler_type": cfg.train.lr_scheduler_type,
        "warmup_ratio": cfg.train.warmup_ratio,
        "weight_decay": cfg.train.weight_decay,
        "logging_steps": cfg.train.logging_steps,
        "save_steps": cfg.train.save_steps,
        "eval_steps": cfg.train.eval_steps,
        "save_total_limit": cfg.train.save_total_limit,
        "bf16": cfg.train.bf16,
        "tf32": cfg.train.tf32,
        "gradient_checkpointing": hf_gradient_checkpointing,
        "dataloader_num_workers": cfg.train.dataloader_num_workers,
        "report_to": cfg.train.report_to,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": cfg.train.ddp_find_unused_parameters,
    }
    if cfg.train.fsdp:
        kwargs["fsdp"] = cfg.train.fsdp
        kwargs["fsdp_config"] = {
            "backward_prefetch": cfg.train.fsdp_backward_prefetch,
            "forward_prefetch": cfg.train.fsdp_forward_prefetch,
            "cpu_offload": cfg.train.fsdp_cpu_offload,
            "use_orig_params": cfg.train.fsdp_use_orig_params,
            "limit_all_gathers": cfg.train.fsdp_limit_all_gathers,
            "activation_checkpointing": cfg.train.fsdp_activation_checkpointing,
            "sync_module_states": cfg.train.fsdp_sync_module_states,
            "cpu_ram_efficient_loading": cfg.train.fsdp_cpu_ram_efficient_loading,
        }
        layer_cls = cfg.train.fsdp_transformer_layer_cls_to_wrap
        if layer_cls and layer_cls.strip().lower() not in {"", "auto"}:
            kwargs["fsdp_config"]["transformer_layer_cls_to_wrap"] = [layer_cls]
        if "save_only_model" in ta_params:
            kwargs["save_only_model"] = True
            logger.warning(
                "Enabling save_only_model for FSDP runs to avoid optimizer-state "
                "save failures with Adafactor. Resume will reload model weights only."
            )
    eval_mode = "steps" if has_eval else "no"
    if "evaluation_strategy" in ta_params:
        kwargs["evaluation_strategy"] = eval_mode
    elif "eval_strategy" in ta_params:
        kwargs["eval_strategy"] = eval_mode
    elif "do_eval" in ta_params:
        kwargs["do_eval"] = has_eval

    if "save_strategy" in ta_params:
        kwargs["save_strategy"] = "steps"
    if "logging_strategy" in ta_params:
        kwargs["logging_strategy"] = "steps"
    if "optim" in ta_params:
        kwargs["optim"] = "adafactor"
    elif "adafactor" in ta_params:
        kwargs["adafactor"] = True

    supported_kwargs = {k: v for k, v in kwargs.items() if k in ta_params}
    dropped_kwargs = sorted(set(kwargs) - set(supported_kwargs))
    if dropped_kwargs:
        logger.info("Skipping unsupported TrainingArguments kwargs: %s", ", ".join(dropped_kwargs))
    if "fsdp" in supported_kwargs:
        logger.info("FSDP enabled mode=%s config=%s", supported_kwargs["fsdp"], supported_kwargs.get("fsdp_config"))

    return TrainingArguments(**supported_kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemma 3 27B IT SFT")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser


def _resolve_fsdp_layer_cls_to_wrap(model: AutoModelForCausalLM, cfg: SFTConfig) -> str | None:
    if not cfg.train.fsdp or "auto_wrap" not in str(cfg.train.fsdp):
        return None

    available = Counter(module.__class__.__name__ for module in model.modules())
    requested = (cfg.train.fsdp_transformer_layer_cls_to_wrap or "").strip()
    if requested and requested.lower() != "auto":
        if requested in available:
            return requested
        logger.warning(
            "Requested fsdp_transformer_layer_cls_to_wrap=%s not found in model. Available sample=%s",
            requested,
            ", ".join(name for name, _ in available.most_common(12)),
        )

    preferred = [
        "Gemma3DecoderLayer",
        "Gemma3TextDecoderLayer",
        "Gemma2DecoderLayer",
        "GemmaDecoderLayer",
        "LlamaDecoderLayer",
        "Qwen2DecoderLayer",
        "MistralDecoderLayer",
    ]
    for candidate in preferred:
        if candidate in available:
            logger.info("Auto-detected FSDP wrap layer class: %s", candidate)
            return candidate

    decoder_like = [(name, count) for name, count in available.items() if "DecoderLayer" in name]
    if decoder_like:
        decoder_like.sort(key=lambda x: x[1], reverse=True)
        chosen = decoder_like[0][0]
        logger.info("Auto-detected FSDP wrap layer class by pattern: %s", chosen)
        return chosen

    block_like = [(name, count) for name, count in available.items() if name.endswith("Block")]
    if block_like:
        block_like.sort(key=lambda x: x[1], reverse=True)
        chosen = block_like[0][0]
        logger.info("Auto-detected FSDP wrap layer class by block fallback: %s", chosen)
        return chosen

    return None


def _build_trainer(
    model: AutoModelForCausalLM,
    args: TrainingArguments,
    train_ds,
    eval_ds,
    tokenizer,
    collator,
) -> FixedAdafactorTrainer:
    trainer_params = set(inspect.signature(FixedAdafactorTrainer.__init__).parameters)
    kwargs: dict[str, object] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
    }
    if "tokenizer" in trainer_params:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        kwargs["processing_class"] = tokenizer

    supported_kwargs = {k: v for k, v in kwargs.items() if k in trainer_params}
    dropped_kwargs = sorted(set(kwargs) - set(supported_kwargs))
    if dropped_kwargs:
        logger.info("Skipping unsupported Trainer kwargs: %s", ", ".join(dropped_kwargs))
    return FixedAdafactorTrainer(**supported_kwargs)


def _validate_launch(cfg: SFTConfig) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if cfg.train.expected_world_size is not None and world_size != cfg.train.expected_world_size:
        logger.warning(
            "WORLD_SIZE mismatch expected=%s actual=%s. This can cause severe throughput/memory issues.",
            cfg.train.expected_world_size,
            world_size,
        )
    if cfg.train.fsdp and world_size <= 1 and cuda_count > 1:
        raise RuntimeError(
            "FSDP is enabled but WORLD_SIZE=1. Launch multi-process training.\n"
            "Example: accelerate launch --num_processes 8 -m gemma27b_sft.cli --config <config.yaml>"
        )
    if "27b" in cfg.model.name_or_path.lower() and cfg.train.max_seq_length > 1024:
        logger.warning(
            "max_seq_length=%s is memory-heavy for 27B full SFT. "
            "Use 1024 (or 768) first to avoid CUDA OOM.",
            cfg.train.max_seq_length,
        )


def run(cfg: SFTConfig) -> None:
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_runtime_compat_overrides(cfg)
    dump_config(cfg, output_dir / "resolved_config.yaml")
    _validate_launch(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning(
            "Tokenizer has no pad_token; falling back to eos_token as pad_token. "
            "This may reduce EOS supervision quality."
        )
    if _is_gemma3_model_name(cfg.model.name_or_path) and not hasattr(tokenizer, "image_token_id"):
        if _ensure_gemma3_tokenizer_attrs(tokenizer):
            logger.warning(
                "Gemma3 tokenizer loaded without image_token_id, but it was recovered from special tokens. "
                "Serving env should use transformers>=4.50.0."
            )
        else:
            logger.warning(
                "Gemma3 tokenizer loaded without image_token_id and recovery failed. "
                "Upgrade transformers in this environment (recommended >=4.50.0)."
            )

    train_ds, eval_ds = build_datasets(cfg, tokenizer)
    if len(train_ds) == 0:
        raise ValueError("Prepared train dataset is empty. Check data fields/prompt length/max_seq_length.")
    logger.info("Dataset ready train=%s eval=%s", len(train_ds), len(eval_ds) if eval_ds is not None else 0)

    model = _load_model(cfg)
    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    use_fsdp_activation_ckpt = bool(cfg.train.fsdp and cfg.train.fsdp_activation_checkpointing)
    use_hf_gradient_ckpt = bool(cfg.train.gradient_checkpointing)
    if use_hf_gradient_ckpt and use_fsdp_activation_ckpt:
        logger.warning(
            "Both train.gradient_checkpointing and train.fsdp_activation_checkpointing are true. "
            "Disabling HF gradient_checkpointing and using FSDP activation_checkpointing only."
        )
        use_hf_gradient_ckpt = False
    if use_hf_gradient_ckpt:
        model.gradient_checkpointing_enable()
    if use_hf_gradient_ckpt or use_fsdp_activation_ckpt:
        model.config.use_cache = False

    _freeze_embeddings(model, cfg.model.freeze_output_embeddings)

    resolved_layer = _resolve_fsdp_layer_cls_to_wrap(model, cfg)
    if cfg.train.fsdp and "auto_wrap" in str(cfg.train.fsdp) and not resolved_layer:
        logger.warning(
            "Could not resolve FSDP transformer layer class; disabling auto_wrap and keeping full_shard."
        )
        cfg.train.fsdp = " ".join(part for part in str(cfg.train.fsdp).split() if part != "auto_wrap")
    elif resolved_layer:
        cfg.train.fsdp_transformer_layer_cls_to_wrap = resolved_layer

    grad_accum = compute_gradient_accumulation_steps(cfg)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    logger.info(
        "Effective batch config global_batch_size=%s per_device_train_batch_size=%s world_size=%s gradient_accumulation_steps=%s",
        cfg.train.global_batch_size,
        cfg.train.per_device_train_batch_size,
        world_size,
        grad_accum,
    )

    args = _build_training_arguments(
        cfg,
        grad_accum,
        has_eval=eval_ds is not None,
        hf_gradient_checkpointing=use_hf_gradient_ckpt,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    trainer = _build_trainer(
        model=model,
        args=args,
        train_ds=train_ds,
        eval_ds=eval_ds,
        tokenizer=tokenizer,
        collator=collator,
    )
    trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(cfg.train.output_dir)
    try:
        # Gemma 3 + vLLM path can require processor artifacts
        # (e.g., preprocessor_config.json) even for text-centric runs.
        _save_processor_artifacts(cfg, tokenizer)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Failed to save processor artifacts to output_dir=%s: %s",
            cfg.train.output_dir,
            exc,
        )


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    logger.info("gemma27b_sft cli path=%s", Path(__file__).resolve())
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
