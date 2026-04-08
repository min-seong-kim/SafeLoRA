"""
LoRA adapter를 base 모델에 merge하여 full model로 저장합니다.

사용법:
    python merge_lora.py --adapter_path ./safe_lora_models/llama3.2-3b-safe-lora-final-20260408-214425 --output_path ./safe_lora_models/llama3.2-3b-safe-lora-final-20260408-214425_merged
    python merge_lora.py --adapter_path ./math_lora_only  # 자동으로 {adapter_path}_merged 에 저장
"""
import argparse
from pathlib import Path

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def merge(adapter_path: str, output_path: str):
    print(f"Loading adapter: {adapter_path}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype="auto",
        device_map="cpu",
    )
    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    output = args.output_path or args.adapter_path.rstrip("/") + "_merged"
    merge(args.adapter_path, output)
