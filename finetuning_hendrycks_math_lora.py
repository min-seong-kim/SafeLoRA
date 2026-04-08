"""
Hendrycks MATH 데이터셋을 사용하여 Instruct 모델을 LoRA로만 파인튜닝

비교 목적:
- 데이터 전처리/마스킹은 finetuning_hendrycks_math_instruct.py와 동일
- 학습 방식만 Full FT 대신 LoRA
- SafeLoRA projection 단계는 없음

Example:
python finetuning_hendrycks_math_lora.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --output_dir ./math_lora_only \
    --num_train_samples 10000
"""

import argparse
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import torch
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def _hf_auth_kwargs(token: Optional[str]) -> Dict[str, str]:
    if not token:
        return {}
    return {"token": token}


def parse_args():
    p = argparse.ArgumentParser(description="LoRA-only Hendrycks MATH fine-tuning")

    p.add_argument("--model_path", type=str, required=True, help="HuggingFace model ID or local path")

    p.add_argument("--math_dataset_source", type=str, default="official", choices=["official", "flat_competition_math"])
    p.add_argument("--math_official_dataset_path", type=str, default="EleutherAI/hendrycks_math")
    p.add_argument("--math_flat_dataset_path", type=str, default="qwedsacf/competition_math")
    p.add_argument("--math_subjects", type=str, default="all")
    p.add_argument("--math_levels", type=str, default="all")
    p.add_argument("--num_train_samples", type=int, default=0)
    p.add_argument("--num_eval_samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--math_train_on_mixed_formats", action="store_true", default=False)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=1024)

    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,up_proj,down_proj",
        help="Comma-separated list of target modules",
    )

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)

    p.add_argument("--output_dir", type=str, default="./math_lora_only")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--cache_dir", type=str, default="./cache")
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--save_merged", action="store_true", help="Save a merged full model in addition to the adapter")

    return p.parse_args()


def is_instruct_model(model_ref: str) -> bool:
    return "instruct" in str(model_ref).lower()


def normalize_csv_arg(raw_value: str) -> str:
    value = str(raw_value).strip()
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        value = value[1:-1].strip()
    return value


def last_boxed_only_string(text: str):
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return text[idx:right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if s is None:
        raise ValueError("remove_boxed received None")
    if "\\boxed " in s:
        left = "\\boxed "
        if s.startswith(left):
            return s[len(left):]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    left = "\\fbox{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return s


def extract_final_answer_from_solution(solution: str) -> str:
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        raise ValueError(f"Could not find final boxed answer in solution: {solution[:300]!r}")
    return remove_boxed(boxed).strip()


def clean_solution_for_reasoning(solution: str, final_answer: str) -> str:
    multi_space_re = re.compile(r"\n{3,}")
    text = solution.strip()
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        text = text.replace(boxed, final_answer)

    text = text.replace("$", "")
    text = text.replace("\\[", "")
    text = text.replace("\\]", "")
    text = text.replace("\\(", "")
    text = text.replace("\\)", "")
    text = text.replace("\\boxed", "")
    text = text.replace("\\fbox", "")
    text = multi_space_re.sub("\n\n", text)
    return text.strip()


def build_target(solution: str, rng: random.Random, train_on_mixed_formats: bool) -> str:
    final_answer = extract_final_answer_from_solution(solution)
    rationale = clean_solution_for_reasoning(solution, final_answer)

    long_target = f"{rationale}\nFinal Answer: ${final_answer}$"
    short_target = f"Final Answer: ${final_answer}$"
    minimal_target = f"${final_answer}$"

    if not train_on_mixed_formats:
        return long_target

    draw = rng.random()
    if draw < 0.70:
        return long_target
    if draw < 0.90:
        return short_target
    return minimal_target


def tokenize_math_sft_example(problem: str, target_text: str, tokenizer, max_length: int, model_ref: str) -> Dict[str, List[int]]:
    problem = str(problem).strip()
    target_text = str(target_text).strip()

    if is_instruct_model(model_ref):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": target_text},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    prompt_text = f"Question: {problem}\nAnswer:"
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    remain = max(1, max_length - len(prompt_ids))
    answer_ids = tokenizer(
        target_text,
        add_special_tokens=False,
        truncation=True,
        max_length=remain,
    )["input_ids"]

    if tokenizer.eos_token_id is not None and (len(answer_ids) == 0 or answer_ids[-1] != tokenizer.eos_token_id):
        if len(prompt_ids) + len(answer_ids) < max_length:
            answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def setup_logging():
    log_dir = "./logs/math_lora_only"
    os.makedirs(log_dir, exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"finetune_math_lora_{log_timestamp}.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_file


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    model_path = args.model_path
    logger, log_file = setup_logging()

    logger.info(f"\n{'=' * 70}")
    logger.info("  🚀 LoRA-only Hendrycks MATH Fine-tuning")
    logger.info(f"{'=' * 70}\n")
    logger.info(f"Log file: {log_file}")
    logger.info("⚙️  Configuration:")
    logger.info(f"   ├─ Model: {model_path}")
    logger.info(f"   ├─ Input formatting: {'chat template' if is_instruct_model(model_path) else 'Question/Answer plain text'}")
    logger.info(f"   ├─ Subjects: {args.math_subjects}")
    logger.info(f"   ├─ Levels: {args.math_levels}")
    logger.info(f"   ├─ Train samples: {args.num_train_samples}")
    logger.info(f"   ├─ Batch size: {args.batch_size}")
    logger.info(f"   ├─ Grad accum: {args.grad_accum}")
    logger.info(f"   ├─ Epochs: {args.epochs}")
    logger.info(f"   ├─ LR: {args.learning_rate}")
    logger.info(f"   ├─ LoRA r/alpha/dropout: {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}")
    logger.info(f"   ├─ LoRA target modules: {args.lora_target_modules}")
    logger.info(f"   └─ Output dir: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, **_hf_auth_kwargs(args.hf_token))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=False,
        **_hf_auth_kwargs(args.hf_token),
    )

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.4f}%)")

    subject_to_config = {
        "Algebra": "algebra",
        "Counting & Probability": "counting_and_probability",
        "Geometry": "geometry",
        "Intermediate Algebra": "intermediate_algebra",
        "Number Theory": "number_theory",
        "Prealgebra": "prealgebra",
        "Precalculus": "precalculus",
    }
    valid_levels = {f"Level {i}" for i in range(1, 6)}

    subjects_arg = normalize_csv_arg(args.math_subjects)
    if subjects_arg.lower() == "all":
        subjects = list(subject_to_config.keys())
    else:
        subjects = [normalize_csv_arg(s) for s in subjects_arg.split(",") if normalize_csv_arg(s)]

    if args.math_dataset_source == "official":
        datasets_per_subject = []
        for subject in subjects:
            config_name = subject_to_config[subject]
            ds = load_dataset(
                args.math_official_dataset_path,
                config_name,
                split="train",
                cache_dir=args.cache_dir,
            )
            ds = ds.map(lambda ex, subject=subject: {"type": subject})
            datasets_per_subject.append(ds)
        train_ds = concatenate_datasets(datasets_per_subject)
    else:
        train_ds = load_dataset(args.math_flat_dataset_path, split="train", cache_dir=args.cache_dir)
        subject_set = set(subjects)
        train_ds = train_ds.filter(lambda ex: ex.get("type") in subject_set)

    levels_arg = normalize_csv_arg(args.math_levels)
    if levels_arg.lower() != "all":
        levels = []
        for item in levels_arg.split(","):
            item = normalize_csv_arg(item)
            if not item:
                continue
            lvl = item if item.startswith("Level ") else f"Level {int(item)}"
            if lvl not in valid_levels:
                raise ValueError(f"Invalid math level: {item}")
            levels.append(lvl)
        allowed_levels = set(levels)
        train_ds = train_ds.filter(lambda ex: ex.get("level") in allowed_levels)

    train_ds = train_ds.shuffle(seed=args.seed)
    if args.num_train_samples and args.num_train_samples > 0:
        train_ds = train_ds.select(range(min(args.num_train_samples, len(train_ds))))

    def preprocess_train(ex, idx: int):
        problem = ex.get("problem", "").strip()
        solution = ex.get("solution", "").strip()
        rng = random.Random(args.seed + idx)
        target_text = build_target(solution, rng, args.math_train_on_mixed_formats)
        return tokenize_math_sft_example(problem, target_text, tokenizer, args.max_length, model_path)

    train_tok = train_ds.map(
        preprocess_train,
        with_indices=True,
        remove_columns=train_ds.column_names,
        num_proc=None,
        desc="Tokenizing Hendrycks MATH train",
    )

    eval_tok = None
    if args.num_eval_samples and args.num_eval_samples > 0:
        eval_tok = train_tok.select(range(min(args.num_eval_samples, len(train_tok))))

    data_collator = DataCollatorForCausalLMWithPadding(tokenizer)
    do_eval = eval_tok is not None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy=("steps" if do_eval else "no"),
        eval_steps=(args.eval_steps if do_eval else None),
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=args.report_to,
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok if do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting LoRA training...")
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_merged:
        logger.info("Saving merged full model...")
        merged_model = model.merge_and_unload()
        merged_dir = os.path.join(args.output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)

    config = {
        "base_model": model_path,
        "fine_tuning_type": "LoRA only",
        "dataset": "Hendrycks MATH",
        "math_dataset_source": args.math_dataset_source,
        "math_subjects": args.math_subjects,
        "math_levels": args.math_levels,
        "num_train_samples": args.num_train_samples,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": args.lr_scheduler_type,
        "optimizer": "AdamW (torch)",
        "gradient_checkpointing": args.gradient_checkpointing,
        "dtype": "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32"),
        "input_formatting": "chat template" if is_instruct_model(model_path) else "Question/Answer plain text",
        "prompt_masking": "assistant-only loss",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": target_modules,
        "trainable_params": trainable_params,
        "all_params": all_params,
    }
    with open(os.path.join(args.output_dir, "lora_math_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info("✅ LoRA-only fine-tuning finished.")


if __name__ == "__main__":
    main()
