#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Safe LoRA + GSM8K 파인튜닝 스크립트 (1, 2, 3단계)

1단계: Base / Aligned(Safety-Tuned) 모델 로드
2단계: Aligned 모델 위에 LoRA + GSM8K 파인튜닝
3단계: Safe LoRA projection 적용 (alignment 방향 보존)

설정:
  base_model   = meta-llama/Llama-2-7b-hf          (비정렬 원본)
  aligned_model = kmseong/llama2_7b-Safety-FT-lr3e-5 (safety-tuned = 정렬됨)
  V_i = W_aligned - W_base  →  안전 정렬 방향 추출

Example:
python safe_lora_gsm8k_training.py \
    --base-model meta-llama/Llama-2-7b-chat-hf \
    --aligned-model kmseong/llama2_7b-chat-Safety-FT-lr3e-5 \
    --num-train-samples 7473 \
    --epochs 3 \
    --lr 2e-4 \
    --safe-select-type number \
    --safe-num-layers 30 \
    --upload-name kmseong/llama2-7b-chat-safe-lora-num_30_gsm8k_lr2e-4 \
    --upload-save-dtype bf16 
"""

import os
import sys
import gc
import json
import random
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
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
from peft import LoraConfig, get_peft_model, PeftModel
from dataclasses import dataclass
from typing import Dict, List, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


try:
    from config import SafeLoRAConfig
    from model import SafeLoRA
    print("[✓] SafeLoRA 라이브러리 로드 성공")
except ImportError as e:
    print(f"[⚠] SafeLoRA 라이브러리 로드 실패: {e}")

try:
    from upload_sn_tuned_model import upload_to_huggingface
    print("[✓] upload_sn_tuned_model 로드 성공")
except ImportError as e:
    upload_to_huggingface = None
    print(f"[⚠] upload_sn_tuned_model 로드 실패 (업로드 기능 비활성화): {e}")

warnings.filterwarnings("ignore")

# ============================================================================
# 기본 설정
# ============================================================================

BASE_PROJECT_DIR = Path(__file__).parent

BASE_MODEL_PATH    = "meta-llama/Llama-2-7b-hf"
ALIGNED_MODEL_PATH = "kmseong/llama2_7b-Safety-FT-lr3e-5"

RUN_TIMESTAMP          = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_LORA_PATH       = str(BASE_PROJECT_DIR / "finetuned_models" / f"gsm8k-llama2-7b-peft-{RUN_TIMESTAMP}")
SAFE_LORA_OUTPUT_PATH  = str(BASE_PROJECT_DIR / "safe_lora_models" / f"llama2-7b-safe-lora-gsm8k-{RUN_TIMESTAMP}")
SAFE_LORA_LOG_DIR      = str(BASE_PROJECT_DIR / "logs")
SAFE_LORA_SELECTION_LOG_PATH      = str(BASE_PROJECT_DIR / "logs" / f"safe_lora_gsm8k_selection_{RUN_TIMESTAMP}.json")
SAFE_LORA_SELECTION_TEXT_LOG_PATH = str(BASE_PROJECT_DIR / "logs" / f"safe_lora_gsm8k_selection_{RUN_TIMESTAMP}.txt")

# LoRA 설정 — Llama 2 7B는 o_proj, gate_proj 포함
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 학습 설정
BATCH_SIZE               = 4
GRAD_ACCUM_STEPS         = 4
NUM_EPOCHS               = 3
LEARNING_RATE            = 3e-5
WEIGHT_DECAY             = 0.01
MAX_GRAD_NORM            = 1.0
WARMUP_RATIO             = 0.1
LR_SCHEDULER_TYPE        = "cosine"
USE_GRADIENT_CHECKPOINTING = True
USE_BF16                 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
MAX_STEPS                = -1

# 데이터 설정
DATASET_NAME       = "openai/gsm8k"
DATASET_SUBSET     = "main"
TRAIN_SPLIT        = "train"
NUM_TRAIN_SAMPLES  = 7473   # 0이면 전체
MAX_LENGTH         = 1024
SEED               = 42

# Safety data mixing (선택)
SAFETY_DATA_PATH  = ""
SAFETY_MIX_RATIO  = 0.0

# Safe LoRA 설정
SAFE_LORA_SELECT_TYPE      = "threshold"
SAFE_LORA_NUM_LAYERS       = 20
SAFE_LORA_THRESHOLD        = 0.5
SAFE_LORA_USE_APPROXIMATION = True

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN  = None

# 업로드 설정
UPLOAD_NAME      = None   # e.g. "kmseong/llama2-7b-safe-lora-gsm8k" (None이면 업로드 안 함)
UPLOAD_SAVE_DTYPE = "bf16"  # merged full model 저장 dtype


# ============================================================================
# 유틸리티
# ============================================================================

def _hf_auth_kwargs(token: Optional[str]) -> Dict[str, str]:
    if not token:
        return {}
    return {"token": token}


def _select_first_n(ds, n: int):
    if n is None or n <= 0:
        return ds
    return ds.select(range(min(n, len(ds))))


def is_instruct_or_chat_model(model_ref: str) -> bool:
    ref = str(model_ref).lower()
    return "instruct" in ref or "chat" in ref


def normalize_model_ref(model_ref: str) -> str:
    model_ref = str(model_ref).strip()
    if model_ref.startswith("https://huggingface.co/") or model_ref.startswith("http://huggingface.co/"):
        parsed = urlparse(model_ref)
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        for kw in ("resolve", "blob", "tree"):
            if kw in parts:
                parts = parts[:parts.index(kw)]
        if len(parts) >= 2:
            return "/".join(parts[:2])
    return model_ref


def is_probably_hf_ref(model_ref: str) -> bool:
    ref = str(model_ref).strip()
    if ref.startswith("http://") or ref.startswith("https://"):
        return "huggingface.co" in ref
    if "/" in ref and not os.path.isabs(ref) and not os.path.exists(ref):
        return True
    return False


def resolve_model_ref(model_ref: str, role: str) -> str:
    normalized = normalize_model_ref(model_ref)
    if os.path.exists(normalized) or is_probably_hf_ref(normalized):
        return normalized
    raise ValueError(
        f"{role} 경로가 존재하지 않습니다: {normalized}\n"
        f"로컬 경로 또는 HF repo id를 전달하세요."
    )


def describe_ref(ref: str) -> str:
    if os.path.exists(ref):
        return "local"
    if is_probably_hf_ref(ref):
        return "huggingface"
    return "unknown"


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id  = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"]      + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0]      * pad_len)
            labels.append(f["labels"]         + [-100]   * pad_len)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }


def write_safe_lora_selection_logs(stats: Dict, metadata: Dict) -> None:
    os.makedirs(SAFE_LORA_LOG_DIR, exist_ok=True)
    with open(SAFE_LORA_SELECTION_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump({"run_timestamp": RUN_TIMESTAMP, "metadata": metadata, "stats": stats},
                  f, indent=2, ensure_ascii=False)

    sorted_metrics   = stats.get("sorted_metrics", [])
    selected_modules = set(stats.get("selected_modules", []))
    with open(SAFE_LORA_SELECTION_TEXT_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"SafeLoRA GSM8K selection log\nrun_timestamp: {RUN_TIMESTAMP}\n")
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")
        f.write(f"num_candidate_layers: {stats.get('num_candidate_layers')}\n")
        f.write(f"num_projected_layers: {stats.get('num_projected_layers')}\n")
        f.write(f"selection_mode: {stats.get('selection_mode')}\n")
        f.write(f"threshold: {stats.get('threshold')}\n")
        f.write(f"use_approximation: {stats.get('use_approximation')}\n")
        f.write("\nPer-layer metrics\n")
        for idx, item in enumerate(sorted_metrics, start=1):
            marker = "[SELECTED]" if item["module"] in selected_modules else "[SKIPPED]"
            f.write(
                f"{idx:03d} {marker} module={item['module']} "
                f"cosine={item['cosine']:.6f} delta_shift={item['delta_shift']:.6f} "
                f"projector={item['projector_key']}\n"
            )


# ============================================================================
# 토크나이즈
# ============================================================================

def tokenize_gsm8k_example(
    question: str,
    answer: str,
    tokenizer,
    max_length: int,
    model_ref: str,
) -> Dict[str, List[int]]:
    question = str(question).strip()
    answer   = str(answer).strip()

    if is_instruct_or_chat_model(model_ref):
        try:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question},
                 {"role": "assistant", "content": answer}],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False,
                                   truncation=True, max_length=max_length)["input_ids"]
            full_ids   = tokenizer(full_text,   add_special_tokens=False,
                                   truncation=True, max_length=max_length)["input_ids"]
            labels = full_ids.copy()
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = -100
            return {"input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels}
        except Exception:
            pass  # chat template 실패 시 plain prompt로 fall-through

    # Plain prompt (base / safety-tuned base 모델)
    prompt_text = f"Question: {question}\nAnswer:"
    prompt_ids  = tokenizer(prompt_text, add_special_tokens=False,
                            truncation=True, max_length=max_length)["input_ids"]
    remain      = max(1, max_length - len(prompt_ids))
    answer_ids  = tokenizer(answer, add_special_tokens=False,
                            truncation=True, max_length=remain)["input_ids"]

    if (tokenizer.eos_token_id is not None
            and (not answer_ids or answer_ids[-1] != tokenizer.eos_token_id)
            and len(prompt_ids) + len(answer_ids) < max_length):
        answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels    = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return {"input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels}


# ============================================================================
# 단계 1: 모델 준비
# ============================================================================

def step1_prepare_models():
    print("\n" + "=" * 80)
    print("단계 1: Base / Aligned(Safety-Tuned) 모델 준비")
    print("=" * 80)

    print(f"\n[1-1] Base Model 로드 중... ({BASE_MODEL_PATH})")
    base_ref = resolve_model_ref(BASE_MODEL_PATH, "base")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_ref, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, **_hf_auth_kwargs(HF_TOKEN),
    )
    print("✓ Base Model 로드 완료")

    print(f"\n[1-2] Aligned(Safety-Tuned) Model 로드 중... ({ALIGNED_MODEL_PATH})")
    aligned_ref = resolve_model_ref(ALIGNED_MODEL_PATH, "aligned")
    aligned_model = AutoModelForCausalLM.from_pretrained(
        aligned_ref, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, **_hf_auth_kwargs(HF_TOKEN),
    )
    print("✓ Aligned Model 로드 완료")

    print("\n[1-3] Tokenizer 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(aligned_ref, **_hf_auth_kwargs(HF_TOKEN))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer 로드 완료")

    print("\n✅ 단계 1 완료")
    return {"base_model": base_model, "aligned_model": aligned_model, "tokenizer": tokenizer}


# ============================================================================
# 단계 2: LoRA + GSM8K 파인튜닝
# ============================================================================

def step2_lora_finetuning(models_dict):
    print("\n" + "=" * 80)
    print("단계 2: LoRA + GSM8K 파인튜닝")
    print("=" * 80)

    tokenizer   = models_dict["tokenizer"]
    train_model = models_dict["aligned_model"]

    # ── 데이터 로드 ──────────────────────────────────────────────
    print("\n[2-1] GSM8K 데이터셋 로드 중...")
    train_ds = load_dataset(DATASET_NAME, DATASET_SUBSET,
                            split=TRAIN_SPLIT, **_hf_auth_kwargs(HF_TOKEN))
    train_ds = train_ds.shuffle(seed=SEED)
    train_ds = _select_first_n(train_ds, NUM_TRAIN_SAMPLES)
    print(f"✓ GSM8K 로드 완료: {len(train_ds)} samples")

    # ── Safety data mixing (선택) ─────────────────────────────────
    if SAFETY_MIX_RATIO > 0 and SAFETY_DATA_PATH and os.path.exists(SAFETY_DATA_PATH):
        with open(SAFETY_DATA_PATH, "r", encoding="utf-8") as f:
            safety_raw = json.load(f)
        num_safety = int(len(train_ds) * SAFETY_MIX_RATIO)
        rng = random.Random(SEED)
        sampled = rng.sample(safety_raw, min(num_safety, len(safety_raw)))
        print(f"   Safety data mixing: {len(sampled)} samples (ratio={SAFETY_MIX_RATIO})")
    else:
        sampled = []

    # ── 토크나이즈 ───────────────────────────────────────────────
    print("\n[2-2] 데이터 전처리 중...")
    num_proc = max(1, min(4, os.cpu_count() or 1))

    def preprocess(ex):
        return tokenize_gsm8k_example(
            ex["question"], ex["answer"], tokenizer, MAX_LENGTH, ALIGNED_MODEL_PATH
        )

    train_tok = train_ds.map(preprocess, remove_columns=train_ds.column_names,
                              num_proc=num_proc, desc="Tokenizing GSM8K")

    if sampled:
        def preprocess_safety(ex):
            return tokenize_gsm8k_example(
                ex["prompt"], ex["llama3_output"], tokenizer, MAX_LENGTH, ALIGNED_MODEL_PATH
            )
        safety_hf  = HFDataset.from_list(sampled)
        safety_tok = safety_hf.map(preprocess_safety,
                                    remove_columns=safety_hf.column_names,
                                    desc="Tokenizing safety data")
        train_tok  = concatenate_datasets([train_tok, safety_tok]).shuffle(seed=SEED)
        print(f"   Total after mixing: {len(train_tok)} samples")
    print("✓ 데이터 전처리 완료")

    # ── LoRA 설정 ────────────────────────────────────────────────
    print("\n[2-3] LoRA 설정...")
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(train_model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"✓ Trainable params: {trainable:,} ({trainable/total*100:.2f}%)")

    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False
    model.train()

    # ── TrainingArguments ────────────────────────────────────────
    print("\n[2-4] 학습 시작...")
    os.makedirs(OUTPUT_LORA_PATH, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=OUTPUT_LORA_PATH,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        max_grad_norm=MAX_GRAD_NORM,
        max_steps=MAX_STEPS,
        logging_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        optim="adamw_torch",
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        bf16=USE_BF16,
        fp16=(not USE_BF16 and torch.cuda.is_available()),
        remove_unused_columns=False,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        data_collator=DataCollatorForCausalLMWithPadding(tokenizer),
    )

    try:
        trainer.train()
    except RuntimeError as e:
        if "does not require grad" in str(e):
            print("⚠ gradient 연결 실패. enable_input_require_grads 확인 필요.")
        raise

    model.save_pretrained(OUTPUT_LORA_PATH)
    tokenizer.save_pretrained(OUTPUT_LORA_PATH)
    print(f"✓ LoRA 어댑터 저장 완료: {OUTPUT_LORA_PATH}")
    print("\n✅ 단계 2 완료")
    return model, tokenizer


# ============================================================================
# 단계 3: Safe LoRA projection
# ============================================================================

def step3_apply_safe_lora(models_dict, tokenizer):
    print("\n" + "=" * 80)
    print("단계 3: Safe LoRA projection 적용")
    print("=" * 80)

    # base_model은 Step 3에서 model.py 내부에서 다시 CPU로 로드하므로
    # 메모리 확보를 위해 Step 1에서 로드한 base_model 해제
    if "base_model" in models_dict:
        del models_dict["base_model"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("   (Step 1 base_model 메모리 해제 완료)")

    if not os.path.exists(OUTPUT_LORA_PATH):
        print(f"⚠ LoRA 경로 없음: {OUTPUT_LORA_PATH}. 단계 2를 먼저 실행하세요.")
        return None

    print("\n[3-1] LoRA 모델 로드 중...")
    aligned_ref = resolve_model_ref(ALIGNED_MODEL_PATH, "aligned")
    aligned_for_adapter = AutoModelForCausalLM.from_pretrained(
        aligned_ref, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, **_hf_auth_kwargs(HF_TOKEN),
    )
    peft_model = PeftModel.from_pretrained(
        aligned_for_adapter, OUTPUT_LORA_PATH, torch_dtype=torch.bfloat16,
    )
    print("✓ LoRA 모델 로드 완료")

    print("\n[3-2] SafeLoRA 설정 생성 중...")
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
    print(f"   select_type={SAFE_LORA_SELECT_TYPE}, threshold={SAFE_LORA_THRESHOLD}, "
          f"num_layers={SAFE_LORA_NUM_LAYERS}, approx={SAFE_LORA_USE_APPROXIMATION}")

    print("\n[3-3] SafeLoRA projection 계산 중...")
    print("   1. V = W_safety_tuned - W_base  (alignment delta)")
    print("   2. C = VVᵀ / ‖V‖_F  (approximate projector)")
    print("   3. cosine(CΔW, ΔW) 계산 후 threshold 미만 레이어만 투영")

    try:
        safelora   = SafeLoRA(peft_model, config)
        safe_model = safelora.model
        print("✓ SafeLoRA projection 완료")
    except Exception as e:
        import traceback
        print(f"⚠ SafeLoRA 실패: {e}")
        traceback.print_exc()
        return None

    # 로그 저장
    metadata = {
        "base_model": BASE_MODEL_PATH,
        "aligned_model": ALIGNED_MODEL_PATH,
        "lora_output_path": OUTPUT_LORA_PATH,
        "safe_lora_output_path": SAFE_LORA_OUTPUT_PATH,
        "dataset": DATASET_NAME,
        "num_train_samples": NUM_TRAIN_SAMPLES,
        "target_modules": LORA_TARGET_MODULES,
        "select_type": SAFE_LORA_SELECT_TYPE,
        "threshold": SAFE_LORA_THRESHOLD,
        "num_layers": SAFE_LORA_NUM_LAYERS,
    }
    write_safe_lora_selection_logs(safelora.stats, metadata)
    print(f"   로그: {SAFE_LORA_SELECTION_TEXT_LOG_PATH}")

    print("\n[3-4] Safe LoRA 모델 저장 중...")
    os.makedirs(SAFE_LORA_OUTPUT_PATH, exist_ok=True)
    safe_model.save_pretrained(SAFE_LORA_OUTPUT_PATH)
    tokenizer.save_pretrained(SAFE_LORA_OUTPUT_PATH)

    metadata["lora_config"] = {
        "r": LORA_R, "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT, "target_modules": LORA_TARGET_MODULES,
    }
    metadata["safe_lora_config"] = {
        "select_type": SAFE_LORA_SELECT_TYPE,
        "threshold": SAFE_LORA_THRESHOLD,
        "num_layers": SAFE_LORA_NUM_LAYERS,
        "use_approximation": SAFE_LORA_USE_APPROXIMATION,
    }
    with open(os.path.join(SAFE_LORA_OUTPUT_PATH, "safe_lora_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Safe LoRA 모델 저장 완료: {SAFE_LORA_OUTPUT_PATH}")
    print("\n✅ 단계 3 완료")
    return safe_model


# ============================================================================
# main
# ============================================================================

def main():
    import argparse
    global BASE_MODEL_PATH, ALIGNED_MODEL_PATH
    global DATASET_NAME, DATASET_SUBSET, TRAIN_SPLIT, NUM_TRAIN_SAMPLES, MAX_LENGTH, SEED
    global SAFETY_DATA_PATH, SAFETY_MIX_RATIO
    global BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
    global MAX_GRAD_NORM, WARMUP_RATIO, LR_SCHEDULER_TYPE, MAX_STEPS
    global LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES
    global SAFE_LORA_SELECT_TYPE, SAFE_LORA_NUM_LAYERS, SAFE_LORA_THRESHOLD, SAFE_LORA_USE_APPROXIMATION
    global OUTPUT_LORA_PATH, SAFE_LORA_OUTPUT_PATH
    global HF_TOKEN
    global UPLOAD_NAME, UPLOAD_SAVE_DTYPE

    p = argparse.ArgumentParser(description="Safe LoRA + GSM8K 파인튜닝")

    # 모델
    p.add_argument("--base-model",    type=str, default=BASE_MODEL_PATH)
    p.add_argument("--aligned-model", type=str, default=ALIGNED_MODEL_PATH)

    # 데이터
    p.add_argument("--dataset-name",       type=str, default=DATASET_NAME)
    p.add_argument("--dataset-subset",     type=str, default=DATASET_SUBSET)
    p.add_argument("--train-split",        type=str, default=TRAIN_SPLIT)
    p.add_argument("--num-train-samples",  type=int, default=NUM_TRAIN_SAMPLES,
                   help="사용할 학습 샘플 수 (0이면 전체)")
    p.add_argument("--max-length",         type=int, default=MAX_LENGTH)
    p.add_argument("--seed",               type=int, default=SEED)
    p.add_argument("--safety-data-path",   type=str, default=SAFETY_DATA_PATH)
    p.add_argument("--safety-mix-ratio",   type=float, default=SAFETY_MIX_RATIO)

    # 학습
    p.add_argument("--batch-size",      type=int,   default=BATCH_SIZE)
    p.add_argument("--grad-accum",      type=int,   default=GRAD_ACCUM_STEPS)
    p.add_argument("--epochs",          type=int,   default=NUM_EPOCHS)
    p.add_argument("--lr",              type=float, default=LEARNING_RATE)
    p.add_argument("--weight-decay",    type=float, default=WEIGHT_DECAY)
    p.add_argument("--warmup-ratio",    type=float, default=WARMUP_RATIO)
    p.add_argument("--max-grad-norm",   type=float, default=MAX_GRAD_NORM)
    p.add_argument("--max-steps",       type=int,   default=MAX_STEPS,
                   help="-1이면 epoch 기준, 양수이면 해당 step에서 조기 종료")

    # LoRA
    p.add_argument("--lora-r",       type=int,   default=LORA_R)
    p.add_argument("--lora-alpha",   type=int,   default=LORA_ALPHA)
    p.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)
    p.add_argument("--lora-target-modules", type=str,
                   default=",".join(LORA_TARGET_MODULES),
                   help="쉼표 구분 모듈명 (예: q_proj,k_proj,v_proj)")

    # Safe LoRA
    p.add_argument("--safe-select-type",  type=str,   default=SAFE_LORA_SELECT_TYPE,
                   choices=["threshold", "number"])
    p.add_argument("--safe-threshold",    type=float, default=SAFE_LORA_THRESHOLD,
                   help="threshold 모드: cosine < threshold 레이어 투영")
    p.add_argument("--safe-num-layers",   type=int,   default=SAFE_LORA_NUM_LAYERS,
                   help="number 모드: cosine 낮은 순 N개 투영")
    p.add_argument("--safe-use-exact-projection", action="store_true",
                   help="근사식 대신 exact projector 사용 (느리지만 더 정확)")

    # 단계 건너뛰기
    p.add_argument("--skip-step1", action="store_true")
    p.add_argument("--skip-step2", action="store_true")
    p.add_argument("--skip-step3", action="store_true")

    # 출력 경로 (선택적 오버라이드)
    p.add_argument("--output-lora-path",      type=str, default=OUTPUT_LORA_PATH)
    p.add_argument("--safe-lora-output-path", type=str, default=SAFE_LORA_OUTPUT_PATH)

    p.add_argument("--hf-token", type=str, default=None)

    # 업로드
    p.add_argument("--upload-name",       type=str, default=None,
                   help="HF repo id (e.g. kmseong/llama2-7b-safe-lora-gsm8k). 지정 시 step3 완료 후 자동 업로드")
    p.add_argument("--upload-save-dtype", type=str, default=UPLOAD_SAVE_DTYPE,
                   choices=["fp16", "bf16", "fp32"],
                   help="merge 후 저장 dtype (기본 bf16)")

    args = p.parse_args()

    # 전역 변수 적용
    BASE_MODEL_PATH    = normalize_model_ref(args.base_model)
    ALIGNED_MODEL_PATH = normalize_model_ref(args.aligned_model)
    DATASET_NAME       = args.dataset_name
    DATASET_SUBSET     = args.dataset_subset
    TRAIN_SPLIT        = args.train_split
    NUM_TRAIN_SAMPLES  = args.num_train_samples
    MAX_LENGTH         = args.max_length
    SEED               = args.seed
    SAFETY_DATA_PATH   = args.safety_data_path
    SAFETY_MIX_RATIO   = args.safety_mix_ratio
    BATCH_SIZE         = args.batch_size
    GRAD_ACCUM_STEPS   = args.grad_accum
    NUM_EPOCHS         = args.epochs
    LEARNING_RATE      = args.lr
    WEIGHT_DECAY       = args.weight_decay
    WARMUP_RATIO       = args.warmup_ratio
    MAX_GRAD_NORM      = args.max_grad_norm
    MAX_STEPS          = args.max_steps
    LORA_R             = args.lora_r
    LORA_ALPHA         = args.lora_alpha
    LORA_DROPOUT       = args.lora_dropout
    LORA_TARGET_MODULES = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    SAFE_LORA_SELECT_TYPE       = args.safe_select_type
    SAFE_LORA_THRESHOLD         = args.safe_threshold
    SAFE_LORA_NUM_LAYERS        = args.safe_num_layers
    SAFE_LORA_USE_APPROXIMATION = not args.safe_use_exact_projection
    OUTPUT_LORA_PATH      = args.output_lora_path
    SAFE_LORA_OUTPUT_PATH = args.safe_lora_output_path
    HF_TOKEN = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    UPLOAD_NAME       = args.upload_name
    UPLOAD_SAVE_DTYPE = args.upload_save_dtype

    set_seed(SEED)

    print("\n" + "=" * 80)
    print("🚀 Safe LoRA + GSM8K 파인튜닝 시작")
    print("=" * 80)
    print(f"  Base Model    : {BASE_MODEL_PATH}    ({describe_ref(BASE_MODEL_PATH)})")
    print(f"  Aligned Model : {ALIGNED_MODEL_PATH} ({describe_ref(ALIGNED_MODEL_PATH)})")
    print(f"  Dataset       : {DATASET_NAME} / {DATASET_SUBSET} / {TRAIN_SPLIT}")
    print(f"  Train Samples : {NUM_TRAIN_SAMPLES}")
    print(f"  LR={LEARNING_RATE}, Epochs={NUM_EPOCHS}, r={LORA_R}, α={LORA_ALPHA}")
    print(f"  SafeLoRA      : {SAFE_LORA_SELECT_TYPE}, threshold={SAFE_LORA_THRESHOLD}")
    print(f"  LoRA Output   : {OUTPUT_LORA_PATH}")
    print(f"  Safe Output   : {SAFE_LORA_OUTPUT_PATH}")
    if UPLOAD_NAME:
        print(f"  HF Upload     : {UPLOAD_NAME} (dtype={UPLOAD_SAVE_DTYPE})")
    else:
        print(f"  HF Upload     : (비활성화 - --upload-name 미지정)")

    models_dict = None
    tokenizer   = None

    try:
        if not args.skip_step1:
            models_dict = step1_prepare_models()
            if models_dict is None:
                sys.exit(1)

        if not args.skip_step2:
            if models_dict is None:
                print("⚠ 단계 1 없이 단계 2 실행 불가.")
                sys.exit(1)
            result = step2_lora_finetuning(models_dict)
            if result is None:
                sys.exit(1)
            _, tokenizer = result  # (model, tokenizer)

        if not args.skip_step3:
            if models_dict is None:
                print("⚠ 단계 1 없이 단계 3 실행 불가.")
                sys.exit(1)
            if tokenizer is None:
                aligned_ref = resolve_model_ref(ALIGNED_MODEL_PATH, "aligned")
                tokenizer = AutoTokenizer.from_pretrained(aligned_ref, **_hf_auth_kwargs(HF_TOKEN))
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            safe_model = step3_apply_safe_lora(models_dict, tokenizer)
            if safe_model is None:
                sys.exit(1)

            # ── 자동 업로드 ─────────────────────────────────────────
            if UPLOAD_NAME:
                if upload_to_huggingface is None:
                    print("⚠ upload_sn_tuned_model 를 불러오지 못해 업로드를 건너뜁니다.")
                else:
                    print("\n" + "=" * 80)
                    print("단계 4: HuggingFace 업로드 (merge + upload)")
                    print("=" * 80)
                    print(f"  어댑터 경로 : {SAFE_LORA_OUTPUT_PATH}")
                    # 어댑터는 aligned_model 위에 학습됐으므로 merge 시 aligned_model을 base로 사용
                    print(f"  Adapter base: {ALIGNED_MODEL_PATH}  ← merge 기준")
                    print(f"  HF repo     : {UPLOAD_NAME}")
                    print(f"  Save dtype  : {UPLOAD_SAVE_DTYPE}")
                    try:
                        upload_to_huggingface(
                            model_path=SAFE_LORA_OUTPUT_PATH,
                            repo_id=UPLOAD_NAME,
                            hf_token=HF_TOKEN,
                            base_model=ALIGNED_MODEL_PATH,   # ← BASE_MODEL_PATH 아님!
                            method_name="Safe LoRA (GSM8K)",
                            save_dtype=UPLOAD_SAVE_DTYPE,
                        )
                        print(f"\n✅ 업로드 완료: https://huggingface.co/{UPLOAD_NAME}")
                    except Exception as upload_err:
                        import traceback as _tb
                        print(f"\n⚠ 업로드 실패 (모델은 로컬에 저장됨): {upload_err}")
                        _tb.print_exc()

        print("\n✅ 전체 완료!")
        print(f"   Safe LoRA 모델: {SAFE_LORA_OUTPUT_PATH}")
        if UPLOAD_NAME:
            print(f"   HF 업로드     : https://huggingface.co/{UPLOAD_NAME}")

    except KeyboardInterrupt:
        print("\n사용자 중단.")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\n❌ 오류 발생: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
