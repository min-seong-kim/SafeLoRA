#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os

import evaluate
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate summarization model on SamSum jsonl.")
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model path or HF repo id.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="./safe_lora_models",
        help="LoRA/SafeLoRA adapter path. If directory is a root folder, latest subdir is used.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="datasets/samsum_test.jsonl",
        help="Path to SamSum test jsonl.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max number of samples to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--use-compile",
        action="store_true",
        help="Enable torch.compile(model) for inference.",
    )
    return parser.parse_args()


def resolve_adapter_path(adapter_path: str) -> str:
    if os.path.isfile(adapter_path):
        raise ValueError(f"adapter path must be a directory: {adapter_path}")

    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        return adapter_path

    if os.path.isdir(adapter_path):
        candidates = []
        for name in os.listdir(adapter_path):
            sub = os.path.join(adapter_path, name)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "adapter_config.json")):
                candidates.append(sub)
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]

    raise ValueError(f"No adapter_config.json found under: {adapter_path}")


def build_prompt(dialogue: str):
    instruction = (
        "You are a helpful, respectful and honest assistant. "
        "Your task is to summarize the following dialogue. "
        "Your answer should be based on the provided dialogue only."
    )
    return [
        {
            "role": "user",
            "content": f"{instruction}\n\nDialogue:\n{dialogue}\n\nSummary:",
        }
    ]


def generate_summary(model, tokenizer, dialogue: str, device: str, max_new_tokens: int) -> str:
    prompt = build_prompt(dialogue)
    input_ids = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = generation_output[0][input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    args = parse_args()

    adapter_path = resolve_adapter_path(args.adapter_path)
    print(f"[INFO] base model: {args.base_model}")
    print(f"[INFO] adapter path: {adapter_path}")
    print(f"[INFO] data path: {args.data_path}")
    print(f"[INFO] device: {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map=args.device,
    )
    model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch.float16 if args.device == "cuda" else None)
    model = model.to(args.device)
    model.eval()

    if args.use_compile:
        model = torch.compile(model)

    rouge = evaluate.load("rouge")

    total = 0
    score_sum = 0.0

    with open(args.data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if total >= args.max_samples:
                break

            item = json.loads(line)
            messages = item.get("messages", [])
            if len(messages) < 2:
                continue

            dialogue = messages[0].get("content", "")
            reference = messages[1].get("content", "")
            if not dialogue or not reference:
                continue

            print(f"[INFO] processing sample {total + 1}")
            prediction = generate_summary(model, tokenizer, dialogue, args.device, args.max_new_tokens)
            result = rouge.compute(predictions=[prediction], references=[reference])
            score_sum += float(result["rouge1"])
            total += 1

    if total == 0:
        raise RuntimeError("No valid evaluation samples processed.")

    print(f"Average Rouge-1 Score: {score_sum / total:.6f}")
    print(f"Evaluated samples: {total}")


if __name__ == "__main__":
    main()
