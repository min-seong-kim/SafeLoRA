#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Safe LoRA 모델 생성 스크립트 (1, 2, 3단계)
1단계: 기본 모델 3개 준비
2단계: LoRA 미세조정 수행
3단계: Safe LoRA 적용

python safe_lora_training.py \
    --base-model meta-llama/Llama-3.2-3B \
    --aligned-model meta-llama/Llama-3.2-3B-Instruct \
    --math-subjects all \
    --math-levels all \
    --train-split train \
    --num-train-samples 10000 \
    --max-length 1024 \
    --safe-num-layers 12 \
    --safe-threshold 0.5 \
    --safe-select-type threshold \
    --use-chat-template \
    --system-prompt "" \
"""

import os
import sys
import json
import random
import re
from urllib.parse import urlparse
from datetime import datetime
import torch
import warnings
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    set_seed,
)
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, PeftModel
from dataclasses import dataclass
from typing import Dict, List, Optional


# SafeLoRA 임포트
try:
    from config import SafeLoRAConfig
    from model import SafeLoRA
    print("[✓] SafeLoRA 라이브러리 로드 성공")
except ImportError as e:
    print(f"[⚠] SafeLoRA 라이브러리 로드 실패: {e}")
    print("[💡] SafeLoRA 디렉토리 경로를 확인하세요.")

warnings.filterwarnings("ignore")

# ============================================================================
# 설정
# ============================================================================

# 모델 경로 설정 (상대경로 또는 절대경로 지정 가능)
BASE_PROJECT_DIR = Path(__file__).parent

# Llama 3.2 3B 모델 설정
BASE_MODEL_PATH = "meta-llama/Llama-3.2-3B"
ALIGNED_MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_LORA_PATH = BASE_PROJECT_DIR / "finetuned_models" / "math-llama3.2-3b-peft"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
SAFE_LORA_OUTPUT_PATH = BASE_PROJECT_DIR / "safe_lora_models" / f"llama3.2-3b-safe-lora-final-{RUN_TIMESTAMP}"
SAFE_LORA_LOG_DIR = BASE_PROJECT_DIR / "logs"
SAFE_LORA_SELECTION_LOG_PATH = SAFE_LORA_LOG_DIR / f"safe_lora_selection_{RUN_TIMESTAMP}.json"
SAFE_LORA_SELECTION_TEXT_LOG_PATH = SAFE_LORA_LOG_DIR / f"safe_lora_selection_{RUN_TIMESTAMP}.txt"

# 경로를 문자열로 변환
BASE_MODEL_PATH = str(BASE_MODEL_PATH)
ALIGNED_MODEL_PATH = str(ALIGNED_MODEL_PATH)
OUTPUT_LORA_PATH = str(OUTPUT_LORA_PATH)
SAFE_LORA_OUTPUT_PATH = str(SAFE_LORA_OUTPUT_PATH)
SAFE_LORA_LOG_DIR = str(SAFE_LORA_LOG_DIR)
SAFE_LORA_SELECTION_LOG_PATH = str(SAFE_LORA_SELECTION_LOG_PATH)
SAFE_LORA_SELECTION_TEXT_LOG_PATH = str(SAFE_LORA_SELECTION_TEXT_LOG_PATH)

# LoRA 설정 (Llama 3.2 3B 최적화)
LORA_R = 8                                           # 더 작은 모델이므로 rank 축소
LORA_ALPHA = 16                                      # alpha도 축소
LORA_DROPOUT = 0.05
# Llama 3.2의 projection layers
# LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
# 원본은 q,v만 사용
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]

# 학습 설정 (SFT와 유사하게 정렬)
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 3e-5
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1
USE_GRADIENT_CHECKPOINTING = True
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
MAX_STEPS = -1                                       # -1이면 epoch 기준으로 학습

DATASET_NAME = "EleutherAI/hendrycks_math"
DATASET_SUBSET = "all"
TRAIN_SPLIT = "train"
NUM_TRAIN_SAMPLES = 10000  # 0 또는 음수면 전체 train split 사용
MAX_LENGTH = 1024
SEED = 42

DATASET_SOURCE = "official"  # supported: "official", "flat_competition_math"
OFFICIAL_DATASET_PATH = "EleutherAI/hendrycks_math"
FLAT_DATASET_PATH = "qwedsacf/competition_math"
MATH_SUBJECTS = "all"  # or "Algebra,Geometry"
MATH_LEVELS = "all"  # or "1,2,3,4,5"
TRAIN_ON_MIXED_FORMATS = False
USE_CHAT_TEMPLATE = True

SUBJECT_TO_CONFIG = {
    "Algebra": "algebra",
    "Counting & Probability": "counting_and_probability",
    "Geometry": "geometry",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Prealgebra": "prealgebra",
    "Precalculus": "precalculus",
}
VALID_LEVELS = {f"Level {i}" for i in range(1, 6)}
MULTISPACE_RE = re.compile(r"\n{3,}")
SYSTEM_PROMPT = (
    "You are a careful competition math solver. Solve the problem step by step. "
    "On the last line, write exactly one final answer in the form: Final Answer: $<answer>$."
)

# Safe LoRA 설정
SAFE_LORA_SELECT_TYPE = "threshold"  # supported: "number", "threshold"
SAFE_LORA_NUM_LAYERS = 20                           # Llama 3 계열은 더 많은 레이어 투영이 안전성에 유리
SAFE_LORA_THRESHOLD = 0.45                          # threshold 모드에서만 사용
SAFE_LORA_USE_APPROXIMATION = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = None


def write_safe_lora_selection_logs(stats: Dict, metadata: Dict) -> None:
    os.makedirs(SAFE_LORA_LOG_DIR, exist_ok=True)

    json_payload = {
        "run_timestamp": RUN_TIMESTAMP,
        "metadata": metadata,
        "stats": stats,
    }
    with open(SAFE_LORA_SELECTION_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    sorted_metrics = stats.get("sorted_metrics", [])
    selected_modules = set(stats.get("selected_modules", []))
    with open(SAFE_LORA_SELECTION_TEXT_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"SafeLoRA selection log\n")
        f.write(f"run_timestamp: {RUN_TIMESTAMP}\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
        f.write(f"num_candidate_layers: {stats.get('num_candidate_layers')}\n")
        f.write(f"num_projected_layers: {stats.get('num_projected_layers')}\n")
        f.write(f"selection_mode: {stats.get('selection_mode')}\n")
        f.write(f"threshold: {stats.get('threshold')}\n")
        f.write(f"num_proj_layers: {stats.get('num_proj_layers')}\n")
        f.write(f"use_approximation: {stats.get('use_approximation')}\n")
        f.write("\nPer-layer metrics\n")
        for idx, item in enumerate(sorted_metrics, start=1):
            marker = "[SELECTED]" if item["module"] in selected_modules else "[SKIPPED]"
            f.write(
                f"{idx:03d} {marker} module={item['module']} "
                f"cosine={item['cosine']:.6f} "
                f"delta_shift={item['delta_shift']:.6f} "
                f"projector={item['projector_key']}\n"
            )


def _hf_auth_kwargs(token: Optional[str]) -> Dict[str, str]:
    """Build kwargs for HF APIs across transformers/datasets versions."""
    if not token:
        return {}
    # Newer HF libraries expect only `token` and reject simultaneous use_auth_token.
    return {"token": token}


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer
    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

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


def _select_first_n(ds, n: int):
    if n is None or n <= 0:
        return ds
    n = min(n, len(ds))
    return ds.select(range(n))


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


def load_official_hendrycks_math_train(dataset_path: str, subjects: List[str]):
    datasets_per_subject = []
    for subject in subjects:
        config_name = SUBJECT_TO_CONFIG[subject]
        ds = load_dataset(dataset_path, config_name, split=TRAIN_SPLIT, **_hf_auth_kwargs(HF_TOKEN))
        ds = ds.map(lambda ex, subject=subject: {"type": subject})
        datasets_per_subject.append(ds)
    return concatenate_datasets(datasets_per_subject)


def load_flat_competition_math(dataset_path: str, subjects: List[str], split: str):
    ds = load_dataset(dataset_path, split=split, **_hf_auth_kwargs(HF_TOKEN))
    return ds.filter(lambda ex: ex["type"] in set(subjects))


def filter_levels(dataset, levels: List[str]):
    allowed = set(levels)
    return dataset.filter(lambda ex: ex["level"] in allowed)


def remove_boxed(s: str) -> str:
    if s is None:
        return ""
    if "\\boxed " in s and s.startswith("\\boxed "):
        return s[len("\\boxed "):]
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{"):-1]
    if s.startswith("\\fbox{") and s.endswith("}"):
        return s[len("\\fbox{"):-1]
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
    if boxed is not None and final_answer:
        text = text.replace(boxed, final_answer)

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


def is_instruct_model_ref(model_ref: str) -> bool:
    return "instruct" in str(model_ref).lower()


def build_question_answer_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def tokenize_question_answer_example(
    tokenizer,
    question: str,
    response: str,
    max_length: int,
    use_chat_template: bool,
    add_eos: bool = True,
    system_prompt: str = "",
):
    question = str(question).strip()
    response = str(response).strip()

    if use_chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": response}],
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

        attention_mask = [1] * len(full_ids)
        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    prompt_text = build_question_answer_prompt(question)
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    remain = max(0, max_length - len(prompt_ids))
    response_ids = tokenizer(
        response,
        add_special_tokens=False,
        truncation=True,
        max_length=max(1, remain) if remain > 0 else 1,
    )["input_ids"]

    if remain == 0:
        response_ids = []
    else:
        response_ids = response_ids[:remain]

    if (
        add_eos
        and tokenizer.eos_token_id is not None
        and len(prompt_ids) + len(response_ids) < max_length
        and (len(response_ids) == 0 or response_ids[-1] != tokenizer.eos_token_id)
    ):
        response_ids = response_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + response_ids)[:max_length]
    attention_mask = [1] * len(input_ids)
    labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_and_mask_math(
    example,
    tokenizer,
    max_length: int,
    train_on_mixed_formats: bool,
    seed: int,
    idx: int,
    use_chat_template: bool,
    model_ref: str,
    system_prompt: str,
):
    problem = example["problem"].strip()
    solution = example["solution"].strip()
    rng = random.Random(seed + idx)
    target_text = build_target(solution, rng, train_on_mixed_formats)
    effective_chat_template = use_chat_template or is_instruct_model_ref(model_ref)
    return tokenize_question_answer_example(
        tokenizer,
        problem,
        target_text,
        max_length=max_length,
        use_chat_template=effective_chat_template,
        system_prompt=system_prompt,
    )




def normalize_model_ref(model_ref: str) -> str:
    """허깅페이스 URL 또는 repo id를 from_pretrained 입력으로 정규화."""
    model_ref = str(model_ref).strip()
    if model_ref.startswith("https://huggingface.co/") or model_ref.startswith("http://huggingface.co/"):
        parsed = urlparse(model_ref)
        path = parsed.path.strip("/")
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2:
            if "resolve" in parts:
                idx = parts.index("resolve")
                parts = parts[:idx]
            elif "blob" in parts:
                idx = parts.index("blob")
                parts = parts[:idx]
            elif "tree" in parts:
                idx = parts.index("tree")
                parts = parts[:idx]
            return "/".join(parts[:2])
    return model_ref


def is_probably_hf_ref(model_ref: str) -> bool:
    """문자열이 HF repo id 또는 URL 형태인지 판별."""
    ref = str(model_ref).strip()
    if ref.startswith("http://") or ref.startswith("https://"):
        return "huggingface.co" in ref
    # namespace/repo 형태를 HF repo id로 간주
    if "/" in ref and not os.path.isabs(ref) and not os.path.exists(ref):
        return True
    return False


def resolve_model_ref(model_ref: str, role: str) -> str:
    """로컬 경로/HF 주소를 검증하고 from_pretrained에 넣을 값을 반환."""
    normalized = normalize_model_ref(model_ref)
    if os.path.exists(normalized):
        return normalized
    if is_probably_hf_ref(normalized):
        return normalized

    # 절대경로인데 존재하지 않으면, 경로 내부의 namespace/repo 패턴을 자동 추출 시도
    if os.path.isabs(normalized):
        parts = [p for p in normalized.split("/") if p]
        for i in range(len(parts) - 1):
            ns = parts[i]
            repo = parts[i + 1]
            if all(c.isalnum() or c in "-_." for c in ns) and all(c.isalnum() or c in "-_." for c in repo):
                candidate = f"{ns}/{repo}"
                if is_probably_hf_ref(candidate) and ns not in {"home", "mnt", "data", "workspace", "tmp", "var", "opt", "usr", "local"}:
                    return candidate

        raise ValueError(
            f"{role} 경로가 존재하지 않습니다: {normalized}\n"
            f"로컬 모델이 없다면 --{role}-model에 HF repo id 또는 URL을 전달하세요.\n"
            f"예시: --{role}-model meta-llama/Llama-3.2-3B"
        )

    return normalized


def describe_ref(ref: str) -> str:
    """로컬 경로인지 HF 원격 참조인지 로그용 문자열 반환."""
    if os.path.exists(ref):
        return "local"
    if is_probably_hf_ref(ref):
        return "huggingface"
    return "unknown"

# ============================================================================
# 단계 1: 기본 모델 3개 준비
# ============================================================================

def step1_prepare_models(skip_if_exists=False):
    """
    단계 1: 기본 모델 3개 준비
    - Base Model
    - Aligned Model (정렬된 모델)
    - 다운로드 확인
    
    Args:
        skip_if_exists: True이면 이미 로드된 모델 건너뛰기
    """
    print("\n" + "="*80)
    print("단계 1: 기본 모델 3개 준비")
    print("="*80)
    
    print("\n[1-1] Base Model 로드 중...")
    try:
        base_ref = resolve_model_ref(BASE_MODEL_PATH, "base")
        print(f"  참조: {base_ref} ({describe_ref(base_ref)})")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_ref,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            **_hf_auth_kwargs(HF_TOKEN),
        )
        print(f"✓ Base Model 로드 완료")
    except Exception as e:
        print(f"⚠ Base Model 로드 실패: {e}")
        return None
    
    print("\n[1-2] Aligned Model 로드 중...")
    try:
        aligned_ref = resolve_model_ref(ALIGNED_MODEL_PATH, "aligned")
        print(f"  참조: {aligned_ref} ({describe_ref(aligned_ref)})")
        
        aligned_model = AutoModelForCausalLM.from_pretrained(
            aligned_ref,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            **_hf_auth_kwargs(HF_TOKEN),
        )
        print(f"✓ Aligned Model 로드 완료")
    except Exception as e:
        print(f"⚠ Aligned Model 로드 실패: {e}")
        return None
    
    print("\n[1-3] Tokenizer 로드 중...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(aligned_ref, **_hf_auth_kwargs(HF_TOKEN))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✓ Tokenizer 로드 완료 (aligned model 기준)")
    except Exception as e:
        print(f"⚠ Tokenizer 로드 실패: {e}")
        return None
    
    print("\n✅ 단계 1 완료: 모든 기본 모델 준비됨")
    return {
        "base_model": base_model,
        "aligned_model": aligned_model,
        "tokenizer": tokenizer
    }


# ============================================================================
# 단계 2: LoRA 미세조정 수행
# ============================================================================

def step2_lora_finetuning(models_dict):
    """
    단계 2: LoRA 미세조정 수행
    - 데이터셋 로드
    - LoRA 설정
    - 학습 실행
    """
    print("\n" + "="*80)
    print("단계 2: LoRA 미세조정 수행")
    print("="*80)
    
    tokenizer = models_dict["tokenizer"]
    train_model = models_dict["aligned_model"]
    
    print("\n[2-1] Hendrycks MATH 데이터셋 로드 중...")
    try:
        subjects = parse_subjects(MATH_SUBJECTS)
        levels = parse_levels(MATH_LEVELS)

        if DATASET_SOURCE == "official":
            full_ds = load_official_hendrycks_math_train(OFFICIAL_DATASET_PATH, subjects)
        else:
            print("⚠ flat_competition_math는 벤치마크 보고용으로 권장되지 않습니다.")
            full_ds = load_flat_competition_math(FLAT_DATASET_PATH, subjects, TRAIN_SPLIT)

        full_ds = filter_levels(full_ds, levels)
        full_ds = full_ds.shuffle(seed=SEED)
        train_dataset = _select_first_n(full_ds, NUM_TRAIN_SAMPLES)
        if len(train_dataset) <= 1:
            raise ValueError("Filtered MATH dataset has 1 or fewer examples. Expand subjects or levels.")
        print(f"✓ MATH 로드 완료")
        print(f"   Source: {DATASET_SOURCE}")
        print(f"   Subjects: {subjects}")
        print(f"   Levels: {levels}")
        print(f"   Train: {len(train_dataset)} samples")
        print(
            "   Prompt Format: "
            + (
                "chat template (phase3-compatible)"
                if (USE_CHAT_TEMPLATE or is_instruct_model_ref(ALIGNED_MODEL_PATH))
                else "Question/Answer plain text (phase3-compatible)"
            )
        )
    except Exception as e:
        print(f"⚠ MATH 로드 실패: {e}")
        return None
    
    print("\n[2-2] MATH 데이터 전처리 중...")
    num_proc = max(1, min(4, os.cpu_count() or 1))
    train_dataset = train_dataset.map(
        lambda ex, idx: tokenize_and_mask_math(
            ex,
            tokenizer,
            MAX_LENGTH,
            TRAIN_ON_MIXED_FORMATS,
            SEED,
            idx,
            USE_CHAT_TEMPLATE,
            ALIGNED_MODEL_PATH,
            SYSTEM_PROMPT,
        ),
        with_indices=True,
        remove_columns=train_dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing & Masking Hendrycks MATH",
    )
    print("✓ 데이터 전처리 완료")
    
    print("\n[2-3] LoRA 설정 생성 중...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print(f"""✓ LoRA 설정 완료:
   - Rank (r): {LORA_R}
   - Alpha: {LORA_ALPHA}
   - Target Modules: {LORA_TARGET_MODULES}
   - Dropout: {LORA_DROPOUT}""")
    
    print("\n[2-4] PEFT 모델 생성 중...")
    model = get_peft_model(train_model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"✓ PEFT 모델 생성 완료")
    print(f"   Trainable params: {trainable_params:,} ({trainable_params/all_params*100:.2f}%)")
    print(f"   Total params: {all_params:,}")

    if not isinstance(train_model, type(models_dict["aligned_model"])):
        print("⚠ 경고: 학습 시작 모델 타입이 aligned 모델과 다릅니다.")

    print("   학습 시작 모델: aligned model")

    if trainable_params == 0:
        print("⚠ 학습 가능한 파라미터가 0개입니다. target_modules 설정을 확인하세요.")
        return None

    # LoRA + gradient checkpointing 조합에서 backward 그래프가 끊기는 문제를 방지
    if USE_GRADIENT_CHECKPOINTING:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False

    model.train()
    
    print("\n[2-5] 학습 설정 구성 중...")
    os.makedirs(OUTPUT_LORA_PATH, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_LORA_PATH,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        max_grad_norm=MAX_GRAD_NORM,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        bf16=USE_BF16,
        fp16=not USE_BF16 and torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    print(f"""✓ 학습 설정 완료:
   - 에포크: {NUM_EPOCHS}
   - 배치 크기: {BATCH_SIZE}
   - Gradient Accumulation: {GRAD_ACCUM_STEPS}
   - 학습률: {LEARNING_RATE}
   - Warmup Ratio: {WARMUP_RATIO}
   - Max Grad Norm: {MAX_GRAD_NORM}
   - Optimizer: adamw_torch
   - LR Scheduler: cosine
   - Gradient Checkpointing: {USE_GRADIENT_CHECKPOINTING}
   - bf16: {USE_BF16}
   - Max Steps: {MAX_STEPS} (epoch 기준 학습)""")
    
    print("\n[2-6] 학습 실행 중...")
    data_collator = DataCollatorForCausalLMWithPadding(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    try:
        trainer.train()
    except RuntimeError as e:
        if "does not require grad" in str(e):
            print("⚠ backward 실패: loss에 gradient가 연결되지 않았습니다.")
            print("💡 점검 포인트:")
            print("   1. LoRA target_modules가 현재 모델 구조와 일치하는지")
            print("   2. gradient checkpointing + input grad 설정이 활성화됐는지")
            print("   3. 학습 가능한 파라미터 수(Trainable params)가 0이 아닌지")
        raise
    print(f"✓ 학습 완료")
    
    print("\n[2-7] LoRA 모델 저장 중...")
    model.save_pretrained(OUTPUT_LORA_PATH)
    tokenizer.save_pretrained(OUTPUT_LORA_PATH)
    print(f"✓ LoRA 모델 저장 완료: {OUTPUT_LORA_PATH}")
    
    print("\n✅ 단계 2 완료: LoRA 미세조정 완료")
    return model, tokenizer


# ============================================================================
# 단계 3: Safe LoRA 적용
# ============================================================================

def step3_apply_safe_lora(models_dict, tokenizer):
    """
    단계 3: Safe LoRA 적용 (안전성 보존)
    - SafeLoRA 클래스 사용
    - Alignment 벡터 추출
    - 투영 행렬 계산 및 적용
    
    Args:
        models_dict: 단계 1에서 반환한 모델 딕셔너리
        tokenizer: 토크나이저
    """
    print("\n" + "="*80)
    print("단계 3: Safe LoRA 적용 (안전성 보존)")
    print("="*80)
    
    print("\n[3-1] SafeLoRA 라이브러리 임포트 중...")
    try:
        from config import SafeLoRAConfig
        from model import SafeLoRA
        print(f"✓ SafeLoRA 라이브러리 로드 완료")
    except ImportError as e:
        print(f"⚠ SafeLoRA 라이브러리 로드 실패: {e}")
        print(f"💡 SafeLoRA 디렉토리가 PYTHONPATH에 있는지 확인하세요.")
        print(f"   현재 경로: {os.getcwd()}")
        return None
    
    print("\n[3-2] LoRA 모델 메모리에 로드 중...")
    try:
        # LoRA 체크포인트 로드
        if not os.path.exists(OUTPUT_LORA_PATH):
            print(f"⚠ LoRA 모델 경로가 존재하지 않습니다: {OUTPUT_LORA_PATH}")
            print(f"💡 단계 2를 먼저 실행하세요.")
            return None
        
        # Adapter는 LoRA를 학습한 base 계열 모델 위에 로드해야 일관성이 맞음
        # 논문 정합성: LoRA adapter는 aligned model 위에 로드
        aligned_ref = resolve_model_ref(ALIGNED_MODEL_PATH, "aligned")
        aligned_model_for_adapter = AutoModelForCausalLM.from_pretrained(
            aligned_ref,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            **_hf_auth_kwargs(HF_TOKEN),
        )

        # PEFT 모델 로드
        peft_model = PeftModel.from_pretrained(
            aligned_model_for_adapter,
            OUTPUT_LORA_PATH,
            torch_dtype=torch.bfloat16,
        )
        print(f"✓ LoRA 모델 로드 완료 (aligned model 기반)")
        print(f"  경로: {OUTPUT_LORA_PATH}")
    except Exception as e:
        print(f"⚠ LoRA 모델 로드 실패: {e}")
        print(f"💡 OUTPUT_LORA_PATH가 올바른지 확인하세요: {OUTPUT_LORA_PATH}")
        return None
    
    print("\n[3-3] SafeLoRA 설정 생성 중...")
    try:
        config = SafeLoRAConfig(
            base_model_path=BASE_MODEL_PATH,
            aligned_model_path=ALIGNED_MODEL_PATH,
            hf_token=HF_TOKEN,
            select_layers_type=SAFE_LORA_SELECT_TYPE,
            num_proj_layers=SAFE_LORA_NUM_LAYERS,
            threshold=SAFE_LORA_THRESHOLD,
            devices=DEVICE,
            use_approximation=SAFE_LORA_USE_APPROXIMATION,
        )
        print(f"""✓ SafeLoRA 설정 완료:
   - Base Model: {BASE_MODEL_PATH}
   - Aligned Model: {ALIGNED_MODEL_PATH}
   - Select Type: {SAFE_LORA_SELECT_TYPE}
   - Num Projection Layers: {SAFE_LORA_NUM_LAYERS}
   - Threshold: {SAFE_LORA_THRESHOLD}
   - Device: {DEVICE}
   - Approx Projection: {SAFE_LORA_USE_APPROXIMATION}""")
    except Exception as e:
        print(f"⚠ SafeLoRA 설정 생성 실패: {e}")
        return None

    # 일관성 검증: Step2 adapter host와 Step3 adapter host는 aligned 기준이어야 함
    if "aligned_model" not in models_dict:
        print("⚠ aligned 모델 참조를 찾지 못했습니다.")
        return None
    
    print("\n[3-4] SafeLoRA 적용 중...")
    print("   ⏳ 이 작업은 수 분이 소요될 수 있습니다...")
    print("   📊 프로세스:")
    print("      1. Alignment Matrix 계산 (V = W_aligned - W_base)")
    print("      2. 투영 행렬 생성 (C = VV^T / ||V||_F 또는 exact projector)")
    print("      3. LoRA 업데이트 유사도 계산")
    print("      4. 선택적 사영 적용")
    
    try:
        safelora = SafeLoRA(peft_model, config)
        safe_model = safelora.model
        print(f"✓ SafeLoRA 적용 완료 (안전성 보존됨)")
        selection_metadata = {
            "base_model": BASE_MODEL_PATH,
            "aligned_model": ALIGNED_MODEL_PATH,
            "lora_output_path": OUTPUT_LORA_PATH,
            "safe_lora_output_path": SAFE_LORA_OUTPUT_PATH,
            "target_modules": LORA_TARGET_MODULES,
            "select_type": SAFE_LORA_SELECT_TYPE,
            "safe_threshold": SAFE_LORA_THRESHOLD,
            "safe_num_layers": SAFE_LORA_NUM_LAYERS,
            "device": DEVICE,
        }
        write_safe_lora_selection_logs(safelora.stats, selection_metadata)
        print("✓ SafeLoRA selected-layer 로그 저장 완료")
        print(f"  JSON: {SAFE_LORA_SELECTION_LOG_PATH}")
        print(f"  TEXT: {SAFE_LORA_SELECTION_TEXT_LOG_PATH}")
    except Exception as e:
        print(f"⚠ SafeLoRA 적용 실패: {e}")
        print(f"💡 다음을 확인하세요:")
        print(f"   1. Base/Aligned 모델 경로가 올바른지")
        print(f"   2. 모들이 동일한 구조인지")
        print(f"   3. 충분한 메모리(GPU)가 있는지")
        import traceback
        traceback.print_exc()
        return None
    
    print("\n[3-5] Safe LoRA 모델 저장 중...")
    try:
        os.makedirs(SAFE_LORA_OUTPUT_PATH, exist_ok=True)
        safe_model.save_pretrained(SAFE_LORA_OUTPUT_PATH)
        tokenizer.save_pretrained(SAFE_LORA_OUTPUT_PATH)
        print(f"✓ Safe LoRA 모델 저장 완료")
        print(f"  경로: {SAFE_LORA_OUTPUT_PATH}")
    except Exception as e:
        print(f"⚠ Safe LoRA 모델 저장 실패: {e}")
        return None
    
    # 메타데이터 저장
    print("\n[3-6] 메타데이터 저장 중...")
    try:
        import json
        metadata = {
            "method": "Safe LoRA",
            "base_model": BASE_MODEL_PATH,
            "aligned_model": ALIGNED_MODEL_PATH,
            "training_dataset": {
                "source": DATASET_SOURCE,
                "official_dataset_path": OFFICIAL_DATASET_PATH,
                "flat_dataset_path": FLAT_DATASET_PATH,
                "subjects": MATH_SUBJECTS,
                "levels": MATH_LEVELS,
                "train_split": TRAIN_SPLIT,
                "num_train_samples": NUM_TRAIN_SAMPLES,
                "seed": SEED,
                "max_length": MAX_LENGTH,
                "train_on_mixed_formats": TRAIN_ON_MIXED_FORMATS,
                "use_chat_template": USE_CHAT_TEMPLATE,
                "system_prompt": SYSTEM_PROMPT,
            },
            "lora_config": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_modules": LORA_TARGET_MODULES,
            },
            "safe_lora_config": {
                "select_type": SAFE_LORA_SELECT_TYPE,
                "num_layers": SAFE_LORA_NUM_LAYERS,
                "threshold": SAFE_LORA_THRESHOLD,
                "use_approximation": SAFE_LORA_USE_APPROXIMATION,
            }
        }
        with open(os.path.join(SAFE_LORA_OUTPUT_PATH, "safe_lora_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ 메타데이터 저장 완료")
    except Exception as e:
        print(f"⚠ 메타데이터 저장 실패: {e} (무시 가능)")
    
    print("\n✅ 단계 3 완료: Safe LoRA 모델 완성!")
    print("\n" + "="*80)
    print("🎉 최종 완료!")
    print("="*80)
    print(f"\n📁 최종 Safe LoRA 모델 위치:")
    print(f"   {SAFE_LORA_OUTPUT_PATH}")
    print(f"\n📋 생성된 파일:")
    print(f"   - adapter_config.json")
    print(f"   - adapter_model.bin")
    print(f"   - tokenizer.json")
    print(f"   - tokenizer_config.json")
    print(f"   - safe_lora_metadata.json")
    print(f"\n✨ 이제 이 모델을 평가하거나 배포할 수 있습니다!")
    
    return safe_model


# ============================================================================
# 메인 실행 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    import argparse
    global BASE_MODEL_PATH, ALIGNED_MODEL_PATH
    global DATASET_NAME, DATASET_SUBSET, TRAIN_SPLIT, NUM_TRAIN_SAMPLES, MAX_LENGTH, SYSTEM_PROMPT, SEED
    global DATASET_SOURCE, OFFICIAL_DATASET_PATH, FLAT_DATASET_PATH, MATH_SUBJECTS, MATH_LEVELS
    global TRAIN_ON_MIXED_FORMATS, USE_CHAT_TEMPLATE
    global BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_EPOCHS, LEARNING_RATE, MAX_GRAD_NORM, WARMUP_RATIO, MAX_STEPS
    global SAFE_LORA_NUM_LAYERS, SAFE_LORA_THRESHOLD, SAFE_LORA_USE_APPROXIMATION, SAFE_LORA_SELECT_TYPE
    global HF_TOKEN
    
    # 명령줄 인자 파서
    parser = argparse.ArgumentParser(description="Safe LoRA 모델 생성 스크립트")
    parser.add_argument("--skip-step1", action="store_true", help="단계 1 건너뛰기")
    parser.add_argument("--skip-step2", action="store_true", help="단계 2 건너뛰기")
    parser.add_argument("--skip-step3", action="store_true", help="단계 3 건너뛰기")
    parser.add_argument("--all-steps", action="store_true", help="모든 단계 실행 (기본값)")
    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL_PATH,
        help="Base 모델 로컬 경로 또는 HF repo id/URL (없으면 HF에서 자동 다운로드)",
    )
    parser.add_argument(
        "--aligned-model",
        type=str,
        default=ALIGNED_MODEL_PATH,
        help="Aligned 모델 로컬 경로 또는 HF repo id/URL (없으면 HF에서 자동 다운로드)",
    )

    # MATH 데이터/학습 하이퍼파라미터
    parser.add_argument("--dataset-source", type=str, default=DATASET_SOURCE, choices=["official", "flat_competition_math"], help="MATH 데이터 로드 소스")
    parser.add_argument("--official-dataset-path", type=str, default=OFFICIAL_DATASET_PATH, help="공식 MATH 데이터셋 경로")
    parser.add_argument("--flat-dataset-path", type=str, default=FLAT_DATASET_PATH, help="flat competition math 데이터셋 경로")
    parser.add_argument("--math-subjects", type=str, default=MATH_SUBJECTS, help="과목 목록(예: Algebra,Geometry 또는 all)")
    parser.add_argument("--math-levels", type=str, default=MATH_LEVELS, help="난이도 목록(예: 1,2,3,4,5 또는 all)")
    parser.add_argument("--train-on-mixed-formats", action="store_true", help="long/short/minimal 타겟 포맷 혼합 학습")
    parser.add_argument("--use-chat-template", action="store_true", help="chat template 기반 포맷으로 학습")
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME, help="(호환용) 학습 데이터셋 이름")
    parser.add_argument("--dataset-subset", type=str, default=DATASET_SUBSET, help="(호환용) 데이터셋 subset/config")
    parser.add_argument("--train-split", type=str, default=TRAIN_SPLIT, help="학습 split")
    parser.add_argument("--num-train-samples", type=int, default=NUM_TRAIN_SAMPLES, help="사용할 학습 샘플 수 (0이면 전체)")
    parser.add_argument("--seed", type=int, default=SEED, help="재현성을 위한 랜덤 시드")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="토큰 최대 길이")
    parser.add_argument("--system-prompt", type=str, default=SYSTEM_PROMPT, help="시스템 프롬프트")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Per-device batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="학습 epoch")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="학습률")
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM, help="Gradient clipping norm")
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO, help="Warmup ratio")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Max steps (-1이면 epoch 기준)")
    parser.add_argument("--safe-num-layers", type=int, default=SAFE_LORA_NUM_LAYERS, help="SafeLoRA 투영 레이어 수")
    parser.add_argument("--safe-threshold", type=float, default=SAFE_LORA_THRESHOLD, help="SafeLoRA cosine threshold")
    parser.add_argument("--safe-select-type", type=str, default=SAFE_LORA_SELECT_TYPE, choices=["threshold", "number"], help="SafeLoRA 레이어 선택 방식 (threshold: cosine 기준, number: 고정 개수 기준)")
    parser.add_argument(
        "--safe-use-exact-projection",
        action="store_true",
        help="근사식 대신 exact projector를 사용합니다. 더 느리지만 더 보수적일 수 있습니다.",
    )
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token (private model/dataset 접근용)")

    args = parser.parse_args()

    BASE_MODEL_PATH = normalize_model_ref(args.base_model)
    ALIGNED_MODEL_PATH = normalize_model_ref(args.aligned_model)
    DATASET_SOURCE = args.dataset_source
    OFFICIAL_DATASET_PATH = args.official_dataset_path
    FLAT_DATASET_PATH = args.flat_dataset_path
    MATH_SUBJECTS = args.math_subjects
    MATH_LEVELS = args.math_levels
    TRAIN_ON_MIXED_FORMATS = args.train_on_mixed_formats
    USE_CHAT_TEMPLATE = args.use_chat_template
    DATASET_NAME = args.dataset_name
    DATASET_SUBSET = args.dataset_subset
    TRAIN_SPLIT = args.train_split
    NUM_TRAIN_SAMPLES = args.num_train_samples
    SEED = args.seed
    MAX_LENGTH = args.max_length
    SYSTEM_PROMPT = args.system_prompt

    set_seed(SEED)

    BATCH_SIZE = args.batch_size
    GRAD_ACCUM_STEPS = args.grad_accum_steps
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    MAX_GRAD_NORM = args.max_grad_norm
    WARMUP_RATIO = args.warmup_ratio
    MAX_STEPS = args.max_steps

    SAFE_LORA_NUM_LAYERS = args.safe_num_layers
    SAFE_LORA_THRESHOLD = args.safe_threshold
    SAFE_LORA_USE_APPROXIMATION = not args.safe_use_exact_projection
    SAFE_LORA_SELECT_TYPE = args.safe_select_type
    HF_TOKEN = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    print("\n" + "="*80)
    print("🚀 Safe LoRA 모델 생성 스크립트 시작")
    print("="*80)
    print(f"\n📊 실행 환경:")
    print(f"   Device: {DEVICE}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"\n📁 모델 참조:")
    print(f"   Base Model: {BASE_MODEL_PATH} ({describe_ref(BASE_MODEL_PATH)})")
    print(f"   Aligned Model: {ALIGNED_MODEL_PATH} ({describe_ref(ALIGNED_MODEL_PATH)})")
    print(f"   LoRA Output: {OUTPUT_LORA_PATH}")
    print(f"   Safe LoRA Output: {SAFE_LORA_OUTPUT_PATH}")
    
    # 단계 실행 여부 결정
    run_step1 = not args.skip_step1
    run_step2 = not args.skip_step2
    run_step3 = not args.skip_step3
    
    print(f"\n⚙️ 실행 설정:")
    print(f"   Step 1 (모델 준비): {'✓' if run_step1 else '✗'}")
    print(f"   Step 2 (LoRA 미세조정): {'✓' if run_step2 else '✗'}")
    print(f"   Step 3 (Safe LoRA 적용): {'✓' if run_step3 else '✗'}")
    print(f"   Dataset Source: {DATASET_SOURCE}")
    print(f"   Official Dataset Path: {OFFICIAL_DATASET_PATH}")
    print(f"   Flat Dataset Path: {FLAT_DATASET_PATH}")
    print(f"   Math Subjects: {MATH_SUBJECTS}")
    print(f"   Math Levels: {MATH_LEVELS}")
    print(f"   Train Split: {TRAIN_SPLIT}")
    print(f"   Num Train Samples: {NUM_TRAIN_SAMPLES}")
    print(f"   Seed: {SEED}")
    print(f"   Max Length: {MAX_LENGTH}")
    print(f"   Train On Mixed Formats: {TRAIN_ON_MIXED_FORMATS}")
    print(f"   Use Chat Template: {USE_CHAT_TEMPLATE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Grad Accum Steps: {GRAD_ACCUM_STEPS}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   LR: {LEARNING_RATE}")
    print(f"   Safe Num Layers: {SAFE_LORA_NUM_LAYERS}")
    print(f"   Safe Threshold: {SAFE_LORA_THRESHOLD}")
    print(f"   Safe Approx Projection: {SAFE_LORA_USE_APPROXIMATION}")
    print(f"   HF Auth: {'enabled' if HF_TOKEN else 'disabled'}")
    
    models_dict = None
    tokenizer = None
    
    try:
        # 단계 1: 기본 모델 준비
        if run_step1:
            models_dict = step1_prepare_models()
            if models_dict is None:
                print("\n❌ 단계 1 실패. 프로그램을 종료합니다.")
                sys.exit(1)
        else:
            print("\n⏭️ 단계 1 건너뜀 (모델 준비)")
        
        # 단계 2: LoRA 미세조정
        if run_step2:
            if models_dict is None:
                print("\n⚠️ 단계 1을 먼저 실행하거나 이전 결과를 로드해주세요.")
                sys.exit(1)
            
            result = step2_lora_finetuning(models_dict)
            if result is None:
                print("\n❌ 단계 2 실패. 프로그램을 종료합니다.")
                sys.exit(1)
            lora_model, tokenizer = result
        else:
            print("\n⏭️ 단계 2 건너뜀 (LoRA 미세조정)")
        
        # 단계 3: Safe LoRA 적용
        if run_step3:
            if models_dict is None:
                print("\n⚠️ 단계 1을 먼저 실행해주세요.")
                sys.exit(1)
            if tokenizer is None:
                # 토크나이저 로드
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        resolve_model_ref(ALIGNED_MODEL_PATH, "aligned"),
                        **_hf_auth_kwargs(HF_TOKEN),
                    )
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                except Exception as e:
                    print(f"⚠️ Tokenizer 로드 실패: {e}")
                    sys.exit(1)
            
            safe_model = step3_apply_safe_lora(models_dict, tokenizer)
            if safe_model is None:
                print("\n❌ 단계 3 실패. 프로그램을 종료합니다.")
                sys.exit(1)
        else:
            print("\n⏭️ 단계 3 건너뜀 (Safe LoRA 적용)")
        
        print("\n✅ 모든 단계가 성공적으로 완료되었습니다!")
        print(f"\n🎯 다음 단계:")
        print(f"   1. Safe LoRA 모델 평가:")
        print(f"      python evaluate_safe_model.py --model-path {SAFE_LORA_OUTPUT_PATH}")
        print(f"   2. 모델 배포 또는 추론 사용")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
