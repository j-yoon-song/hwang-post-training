from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any

import torch
import transformers
from metricx24 import models


def _build_input(source: str, hypothesis: str, is_qe: bool) -> str:
    if is_qe:
        return f"source: {source} candidate: {hypothesis}"
    return f"source: {source} candidate: {hypothesis} reference: "


def _extract_score(output: Any) -> float:
    prediction = None
    if hasattr(output, "predictions"):
        prediction = output.predictions
    elif isinstance(output, (tuple, list)) and output:
        prediction = output[0]
    elif hasattr(output, "logits"):
        prediction = output.logits

    if prediction is None:
        raise RuntimeError("MetricX model output has no predictions/logits field.")

    if hasattr(prediction, "detach"):
        prediction = prediction.detach()
    if hasattr(prediction, "float"):
        prediction = prediction.float()
    if hasattr(prediction, "cpu"):
        prediction = prediction.cpu()
    if hasattr(prediction, "reshape"):
        prediction = prediction.reshape(-1)

    if len(prediction) == 0:  # type: ignore[arg-type]
        raise RuntimeError("MetricX prediction tensor is empty.")

    value = prediction[0]  # type: ignore[index]
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _send(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Persistent MetricX worker")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--max_input_length", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--qe", action="store_true", default=False)
    args = parser.parse_args()

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
        model = models.MT5ForRegression.from_pretrained(
            args.model_name_or_path,
            torch_dtype="auto",
        )
        model.to(device)
        model.eval()
    except Exception as exc:  # pylint: disable=broad-except
        _send(
            {
                "event": "init_error",
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 1

    _send({"event": "ready", "ok": True, "device": str(device)})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        payload: Any = None
        try:
            payload = json.loads(line)
            if payload.get("event") == "shutdown":
                _send({"event": "bye", "ok": True})
                return 0

            req_id = str(payload.get("id", ""))
            pairs = payload.get("pairs", [])
            if not req_id:
                _send({"ok": False, "error": "missing request id"})
                continue
            if not isinstance(pairs, list):
                _send({"id": req_id, "ok": False, "error": "pairs must be a list"})
                continue

            scores: list[float] = []
            with torch.no_grad():
                for pair in pairs:
                    if not isinstance(pair, dict):
                        raise RuntimeError("pair item must be an object")
                    source = str(pair.get("source", ""))
                    hypothesis = str(pair.get("hypothesis", ""))
                    text = _build_input(source, hypothesis, args.qe)
                    encoded = tokenizer(
                        text,
                        max_length=args.max_input_length,
                        truncation=True,
                        padding=False,
                        return_tensors="pt",
                    )
                    input_ids = encoded["input_ids"]
                    attention_mask = encoded["attention_mask"]

                    # Match upstream script behavior.
                    if input_ids.shape[-1] > 1:
                        input_ids = input_ids[:, :-1]
                        attention_mask = attention_mask[:, :-1]

                    output = model(
                        input_ids=input_ids.to(device),
                        attention_mask=attention_mask.to(device),
                    )
                    score = _extract_score(output)
                    scores.append(max(0.0, min(25.0, score)))

            _send({"id": req_id, "ok": True, "scores": scores})
        except Exception as exc:  # pylint: disable=broad-except
            _send(
                {
                    "id": payload.get("id") if isinstance(payload, dict) else None,  # type: ignore[union-attr]
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
