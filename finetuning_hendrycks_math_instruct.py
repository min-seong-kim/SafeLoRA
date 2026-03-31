import argparse
import datetime
import inspect
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

"""
Example:
python finetuning_hendrycks_math_instruct.py \
  --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
  --output_dir ./MATH_llama32_3b_full_sft \
  --max_seq_length 2024 \
  --batch_size 4 \
  --grad_accum 4 \
  --epochs 3 \
  --learning_rate 3e-6 \
  --train_on_mixed_formats
"""

SUBJECT_TO_CONFIG = {
    "Algebra": "algebra",
    "Counting & Probability": "counting_and_probability",
    "Geometry": "geometry",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Prealgebra": "prealgebra",
    "Precalculus": "precalculus",
}
CONFIG_TO_SUBJECT = {v: k for k, v in SUBJECT_TO_CONFIG.items()}
VALID_LEVELS = {f"Level {i}" for i in range(1, 6)}

BOXED_RE = re.compile(r"\\boxed\s*\{")
FBOX_RE = re.compile(r"\\fbox\s*\{")
MULTISPACE_RE = re.compile(r"\n{3,}")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Full fine-tune Llama-3.2-3B-Instruct on Hendrycks MATH with "
            "benchmark-aligned targets and official train split aggregation."
        )
    )

    p.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--output_dir", type=str, default="./MATH_llama32_3b_full_sft")

    p.add_argument("--max_seq_length", type=int, default=3072)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=3e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation_size", type=int, default=500)

    p.add_argument(
        "--dataset_source",
        type=str,
        choices=["official", "flat_competition_math"],
        default="official",
        help=(
            "official: aggregate EleutherAI/hendrycks_math official train split across all 7 subjects. "
            "flat_competition_math: load qwedsacf/competition_math single split; not recommended for benchmark reporting."
        ),
    )
    p.add_argument("--official_dataset_path", type=str, default="EleutherAI/hendrycks_math")
    p.add_argument("--flat_dataset_path", type=str, default="qwedsacf/competition_math")

    p.add_argument(
        "--subjects",
        type=str,
        default="all",
        help=(
            "Comma-separated subjects. Example: 'Algebra,Geometry'. "
            "Use 'all' for all seven subjects."
        ),
    )
    p.add_argument(
        "--levels",
        type=str,
        default="all",
        help="Comma-separated difficulty levels using numbers 1-5. Example: '1,2,3,4,5'. Use 'all' for all levels.",
    )

    p.add_argument(
        "--train_on_mixed_formats",
        action="store_true",
        help=(
            "Mix long benchmark-aligned reasoning targets with short answer-only targets. "
            "Recommended for lm-eval hendrycks_math exact_match."
        ),
    )
    p.add_argument(
        "--use_chat_template",
        action="store_true",
        help=(
            "Train with the model's chat template. Leave this off if your lm_eval run uses the task default plain prompt."
        ),
    )
    p.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a careful competition math solver. Solve the problem step by step. "
            "On the last line, write exactly one final answer in the form: Final Answer: $<answer>$. "
            "Do not use additional dollar signs earlier in the response."
        ),
    )
    return p.parse_args()


def parse_subjects(subjects_arg: str) -> List[str]:
    if subjects_arg.strip().lower() == "all":
        return list(SUBJECT_TO_CONFIG.keys())
    subjects = [s.strip() for s in subjects_arg.split(",") if s.strip()]
    invalid = [s for s in subjects if s not in SUBJECT_TO_CONFIG]
    if invalid:
        raise ValueError(f"Invalid subject(s): {invalid}. Valid options: {list(SUBJECT_TO_CONFIG.keys())}")
    return subjects


def parse_levels(levels_arg: str) -> List[str]:
    if levels_arg.strip().lower() == "all":
        return sorted(VALID_LEVELS)
    out = []
    for item in levels_arg.split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("Level "):
            lvl = item
        else:
            lvl = f"Level {int(item)}"
        if lvl not in VALID_LEVELS:
            raise ValueError(f"Invalid level: {item}. Valid levels are 1,2,3,4,5.")
        out.append(lvl)
    return out


def is_bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def get_model_dtype():
    if is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


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


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def extract_final_answer_from_solution(solution: str) -> str:
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        raise ValueError(f"Could not find final boxed answer in solution: {solution[:300]!r}")
    return remove_boxed(boxed).strip()


def clean_solution_for_reasoning(solution: str, final_answer: str) -> str:
    text = solution.strip()

    boxed = last_boxed_only_string(text)
    if boxed is not None:
        text = text.replace(boxed, final_answer)

    # Keep mathematical content, but remove delimiters that interfere with lm-eval's current extractor.
    text = text.replace("$", "")
    text = text.replace("\\[", "")
    text = text.replace("\\]", "")
    text = text.replace("\\(", "")
    text = text.replace("\\)", "")
    text = text.replace("\\boxed", "")
    text = text.replace("\\fbox", "")
    text = MULTISPACE_RE.sub("\n\n", text)
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


def render_prompt_only_plain(problem: str) -> str:
    return f"Problem: {problem}\nAnswer:"


def render_full_plain(problem: str, target_text: str) -> str:
    return f"Problem: {problem}\nAnswer: {target_text}"


def render_prompt_only_chat(tokenizer, problem: str, system_prompt: str) -> str:
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Problem: {problem}\nAnswer:"},
    ]
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def render_full_chat(tokenizer, problem: str, target_text: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Problem: {problem}\nAnswer:"},
        {"role": "assistant", "content": target_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_and_mask(
    example,
    tokenizer,
    max_length: int,
    train_on_mixed_formats: bool,
    seed: int,
    idx: int,
    use_chat_template: bool,
    system_prompt: str,
):
    problem = example["problem"].strip()
    solution = example["solution"].strip()
    rng = random.Random(seed + idx)
    target_text = build_target(solution, rng, train_on_mixed_formats) + tokenizer.eos_token

    if use_chat_template:
        prompt_text = render_prompt_only_chat(tokenizer, problem, system_prompt)
        full_text = render_full_chat(tokenizer, problem, target_text, system_prompt)
    else:
        prompt_text = render_prompt_only_plain(problem)
        full_text = render_full_plain(problem, target_text)

    full_encoded = tokenizer(full_text, max_length=max_length, truncation=True)
    prompt_encoded = tokenizer(prompt_text, max_length=max_length, truncation=True)

    input_ids = full_encoded["input_ids"]
    attention_mask = full_encoded["attention_mask"]
    prompt_ids = prompt_encoded["input_ids"]

    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
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


def load_official_hendrycks_math_train(subjects: List[str]):
    datasets_per_subject = []
    for subject in subjects:
        config_name = SUBJECT_TO_CONFIG[subject]
        ds = load_dataset("EleutherAI/hendrycks_math", config_name, split="train")
        ds = ds.map(lambda ex, subject=subject: {"type": subject})
        datasets_per_subject.append(ds)
    return concatenate_datasets(datasets_per_subject)


def load_flat_competition_math(subjects: List[str]):
    ds = load_dataset("qwedsacf/competition_math", split="train")
    return ds.filter(lambda ex: ex["type"] in set(subjects))


def filter_levels(dataset, levels: List[str]):
    allowed = set(levels)
    return dataset.filter(lambda ex: ex["level"] in allowed)


def preview_examples(tokenizer, dataset, args):
    if len(dataset) == 0:
        return

    sample = dataset[0]
    rng = random.Random(args.seed)
    target = build_target(sample["solution"], rng, args.train_on_mixed_formats)

    if args.use_chat_template:
        prompt_text = render_prompt_only_chat(tokenizer, sample["problem"], args.system_prompt)
        full_text = render_full_chat(tokenizer, sample["problem"], target + tokenizer.eos_token, args.system_prompt)
    else:
        prompt_text = render_prompt_only_plain(sample["problem"])
        full_text = render_full_plain(sample["problem"], target + tokenizer.eos_token)

    print("\n=== First raw MATH sample ===")
    print("Type:", sample.get("type", "N/A"))
    print("Level:", sample.get("level", "N/A"))
    print("Problem:")
    print(sample["problem"])
    print("\nSolution:")
    print(sample["solution"][:2000])
    print("=== End raw sample ===\n")

    print("\n=== Prompt preview ===")
    print(prompt_text[:1500])
    print("=== End prompt preview ===\n")

    print("\n=== Full formatted training text preview ===")
    print(full_text[:2500])
    print("=== End formatted preview ===\n")


def build_training_args(args, resolved_output_dir: str):
    kwargs = dict(
        output_dir=resolved_output_dir,
        num_train_epochs=float(args.epochs),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=is_bf16_supported(),
        fp16=torch.cuda.is_available() and not is_bf16_supported(),
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        optim="adamw_torch",
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"

    return TrainingArguments(**kwargs)


def main():
    args = parse_args()
    set_seed(args.seed)

    subjects = parse_subjects(args.subjects)
    levels = parse_levels(args.levels)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(resolved_output_dir, exist_ok=True)
    print(f"Resolved output directory: {resolved_output_dir}")

    if args.dataset_source == "flat_competition_math":
        print(
            "WARNING: flat_competition_math loads qwedsacf/competition_math, which is a single 12.5k split. "
            "Do not use this setting if you plan to report lm_eval hendrycks_math benchmark results."
        )

    print(f"Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model for full fine-tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=get_model_dtype(),
        low_cpu_mem_usage=True,
    )

    print("Loading Hendrycks MATH training data...")
    if args.dataset_source == "official":
        full_train_ds = load_official_hendrycks_math_train(subjects)
    else:
        full_train_ds = load_flat_competition_math(subjects)

    full_train_ds = filter_levels(full_train_ds, levels).shuffle(seed=args.seed)

    if len(full_train_ds) <= 1:
        raise ValueError("Filtered dataset has 1 or fewer examples. Expand subjects or levels.")

    train_ds = full_train_ds

    print(f"Subjects: {subjects}")
    print(f"Levels: {levels}")
    print(f"Train size: {len(train_ds)}")
    preview_examples(tokenizer, train_ds, args)

    print("Tokenizing and masking datasets (assistant-only loss)...")
    train_tok = train_ds.map(
        lambda ex, idx: tokenize_and_mask(
            ex,
            tokenizer,
            args.max_seq_length,
            args.train_on_mixed_formats,
            args.seed,
            idx,
            args.use_chat_template,
            args.system_prompt,
        ),
        with_indices=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train set",
    )

    data_collator = DataCollatorForCausalLMWithPadding(tokenizer)
    training_args = build_training_args(args, resolved_output_dir)

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    model.config.use_cache = False

    print("Starting full parameter fine-tuning...")
    trainer.train()

    print("Saving fine-tuned model...")
    trainer.save_model(resolved_output_dir)
    tokenizer.save_pretrained(resolved_output_dir)
    print("Full fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
