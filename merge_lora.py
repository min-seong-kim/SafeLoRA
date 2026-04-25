"""
LoRA adapter를 base 모델에 merge하여 full model로 저장합니다.

사용법:
    python merge_lora.py --adapter_path ./safe_lora_models/llama2-7b-safe-lora-gsm8k-20260424-223126
    python merge_lora.py --adapter_path ./math_lora_only
"""
import argparse

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def merge(adapter_path: str):
    output_path = adapter_path.rstrip("/") + "_merge"
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
    args = parser.parse_args()
    merge(args.adapter_path)
