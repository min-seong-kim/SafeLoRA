"""
Upload SN-Tuned Model to Hugging Face Hub

Usage:
    python upload_sn_tuned_model.py <model_local_path> --repo_id <username/model_name> [--hf_token <token>]

Example:
python upload_sn_tuned_model.py \
    /home/yonsei_jong/SafeLoRA/safe_lora_models/llama2-7b-safe-lora-gsm8k-20260424-223126_merge \
    --repo_id kmseong/Llama-2-7B-base-SafeLoRA-gsm8k-lr3e-3 
"""

import argparse
import os
import sys
import tempfile
import shutil
from datetime import datetime
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args(argv):
    p = argparse.ArgumentParser(description="Upload SN-tuned model to Hugging Face Hub")
    p.add_argument("model_path", type=str, help="Local model directory path")
    p.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Target Hugging Face repository id, e.g. kmseong/my-model",
    )
    p.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face access token. If omitted, use HF_TOKEN/HUGGINGFACE_TOKEN or cached login.",
    )
    p.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model repo/path used for merging adapter models (Safe LoRA/LoRA).",
    )
    p.add_argument(
        "--method_name",
        type=str,
        default="Safe LoRA",
        help="Method name to display in the model card.",
    )
    p.add_argument(
        "--save_dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Merged full model 저장 dtype (기본 bf16, 용량 절감).",
    )
    return p.parse_args(argv)


def _resolve_dtype(dtype_name):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def _is_adapter_dir(model_path):
    adapter_config = os.path.join(model_path, "adapter_config.json")
    adapter_bin = os.path.join(model_path, "adapter_model.bin")
    adapter_safe = os.path.join(model_path, "adapter_model.safetensors")
    return os.path.exists(adapter_config) and (os.path.exists(adapter_bin) or os.path.exists(adapter_safe))


def _ensure_tokenizer_assets(path_for_upload, source_model_path, fallback_model_name, token=None):
    has_tokenizer_asset = any(
        os.path.exists(os.path.join(path_for_upload, candidate))
        for candidate in ["tokenizer.json", "tokenizer.model"]
    )
    if has_tokenizer_asset:
        return

    logger.info("⚠ Tokenizer assets missing, attempting recovery...")
    # First try copying tokenizer from the original model/adaptor path.
    try:
        tok = AutoTokenizer.from_pretrained(source_model_path, token=token)
        tok.save_pretrained(path_for_upload)
        logger.info("✓ Tokenizer copied from source model path")
        return
    except Exception as e:
        logger.warning(f"Could not copy tokenizer from source path: {e}")

    # Fallback to base model tokenizer.
    tok = AutoTokenizer.from_pretrained(fallback_model_name, token=token)
    tok.save_pretrained(path_for_upload)
    logger.info(f"✓ Tokenizer copied from fallback model: {fallback_model_name}")


def _prepare_upload_model_path(model_path, base_model, token=None, save_dtype="bf16"):
    """If input is adapter-only, merge into full model and return merged temp path."""
    if not _is_adapter_dir(model_path):
        return model_path, None

    logger.info("\n[Merge] Adapter model detected. Merging with base model before upload...")
    merged_tmp_dir = tempfile.mkdtemp(prefix="merged_safe_lora_")
    target_dtype = _resolve_dtype(save_dtype)
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=token,
            device_map="cpu",
            torch_dtype=target_dtype,
        )
        peft_model = PeftModel.from_pretrained(base, model_path, token=token)
        merged_model = peft_model.merge_and_unload()
        merged_model = merged_model.to(dtype=target_dtype)
        merged_model.save_pretrained(
            merged_tmp_dir,
            safe_serialization=True,
            max_shard_size="5GB",
        )

        # Prefer adapter-side tokenizer if available; otherwise fallback to base model tokenizer.
        _ensure_tokenizer_assets(merged_tmp_dir, model_path, base_model, token=token)

        # Preserve metadata if present.
        metadata_src = os.path.join(model_path, "safe_lora_metadata.json")
        metadata_dst = os.path.join(merged_tmp_dir, "safe_lora_metadata.json")
        if os.path.exists(metadata_src):
            shutil.copy2(metadata_src, metadata_dst)

        logger.info(f"✓ Merge complete. Temporary merged model path: {merged_tmp_dir}")
        logger.info(f"✓ Saved merged weights dtype: {save_dtype}")
        return merged_tmp_dir, merged_tmp_dir
    except Exception:
        shutil.rmtree(merged_tmp_dir, ignore_errors=True)
        raise


def upload_to_huggingface(
    model_path,
    repo_id,
    hf_token=None,
    base_model="meta-llama/Llama-3.2-3B-Instruct",
    method_name="Safe LoRA",
    save_dtype="bf16",
):
    """
    Upload SN-tuned model to Hugging Face Hub
    
    Args:
        model_path: Local path to the model directory
        repo_id: Target Hugging Face repo id (username/model_name)
        hf_token: Optional Hugging Face token
    """
    
    # Resolve token once and reuse for all hub operations.
    effective_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    # Verify model path exists
    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)
    
    upload_path = model_path
    cleanup_tmp_dir = None
    
    model_name = repo_id.split("/")[-1]
    
    logger.info(f"\n{'='*70}")
    logger.info("Uploading SN-Tuned Model to Hugging Face Hub")
    logger.info(f"{'='*70}")
    logger.info(f"Local model path: {model_path}")
    logger.info(f"Repository ID: {repo_id}")
    logger.info(f"Model name: {model_name}")
    
    try:
        # Step 1: Authenticate with Hugging Face
        logger.info("\n[Step 1] Authenticating with Hugging Face...")
        try:
            api = HfApi(token=effective_token)
            whoami = api.whoami(token=effective_token)
            logger.info(f"✓ Authenticated as: {whoami.get('name', 'unknown')}")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            logger.info("\nPlease login to Hugging Face:")
            logger.info("  1) Pass --hf_token <token>")
            logger.info("  2) Or set HF_TOKEN/HUGGINGFACE_TOKEN")
            logger.info("  3) Or run: huggingface-cli login")
            sys.exit(1)

        # Step 1.5: Prepare merged full model when input is adapter-only.
        upload_path, cleanup_tmp_dir = _prepare_upload_model_path(
            model_path=model_path,
            base_model=base_model,
            token=effective_token,
            save_dtype=save_dtype,
        )

        # Ensure tokenizer assets exist for upload path.
        _ensure_tokenizer_assets(upload_path, model_path, base_model, token=effective_token)

        # Full-model upload expects config.json.
        config_path = os.path.join(upload_path, "config.json")
        if not os.path.exists(config_path):
            logger.error(f"config.json not found in upload path: {upload_path}")
            sys.exit(1)
        
        # Step 2: Load model and tokenizer locally to verify
        logger.info("\n[Step 2] Verifying model locally...")
        try:
            logger.info("  Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(upload_path)
            logger.info("  ✓ Tokenizer loaded")
            
            logger.info("  Loading model config...")
            # Load without device_map for verification (avoid GPU memory issues)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    upload_path,
                    device_map=None,  # Don't use device_map during verification
                    torch_dtype=None,  # Use default dtype
                )
            except Exception as e1:
                # Fallback: try with CPU
                logger.warning(f"  Failed with device_map=None: {e1}")
                logger.info("  Retrying with CPU...")
                model = AutoModelForCausalLM.from_pretrained(
                    upload_path,
                    device_map="cpu"
                )
            
            logger.info("  ✓ Model loaded")
            logger.info(f"  Model type: {type(model).__name__}")
            logger.info(f"  Model size: {model.num_parameters():,} parameters")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            logger.error(f"Full traceback:")
            logger.error(traceback.format_exc())
            logger.info("\n  Diagnosis: Checking model files...")
            logger.info(f"  Model directory: {upload_path}")
            logger.info(f"  Files present:")
            import subprocess
            result = subprocess.run(['ls', '-lh', upload_path], capture_output=True, text=True)
            logger.info(result.stdout)
            sys.exit(1)
        
        # Step 3: Upload to Hugging Face
        logger.info(f"\n[Step 3] Uploading to Hugging Face...")
        logger.info(f"  Repository: {repo_id}")
        
        try:
            # Step 3a: Create repository if it doesn't exist
            logger.info("  Creating repository on hub...")
            try:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    exist_ok=True,
                    token=effective_token,
                )
                logger.info("  ✓ Repository created/verified")
            except Exception as e:
                logger.warning(f"  Warning creating repo: {e}")
            
            # Step 3b: Push model and tokenizer to hub (excluding checkpoint directories)
            logger.info("  Pushing model to hub (this may take a few minutes)...")
            
            # Upload entire folder excluding checkpoint directories
            api.upload_folder(
                folder_path=upload_path,
                repo_id=repo_id,
                ignore_patterns=["checkpoint-*", ".git*", ".DS_Store"],
                commit_message=f"{method_name} merged full model upload",
                token=effective_token,
            )
            logger.info("  ✓ Model pushed to hub (checkpoints excluded)")
            
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            sys.exit(1)
        
        # Step 4: Create model card (README)
        logger.info(f"\n[Step 4] Creating model card...")
        
        readme_content = f"""---
license: apache-2.0
tags:
- safety
- fine-tuning
- llama
- safety-neurons
---

# {model_name}

This is a merged full model uploaded from a parameter-efficient fine-tuning checkpoint.

## Model Description

- **Base Model**: {base_model}
- **Fine-tuning Method**: {method_name}
- **Upload Source**: {model_path}
- **Upload Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Notes

If the source directory contained adapter weights (for example Safe LoRA),
this upload script merged them with the base model first.
So this repository contains directly loadable full model weights.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
prompt = "How can I help you today?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Safety Note

This model was produced with {method_name} and uploaded as a merged full model.

## License

This model is licensed under the Apache 2.0 License.
See the base model ({base_model}) for more details.

## References

- Base model: https://huggingface.co/{base_model}
- Method: {method_name}
"""
        
        try:
            readme_path = os.path.join(model_path, "README.md")
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            logger.info("  ✓ README.md created")
            
            # Push README to hub
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add model card",
                token=effective_token,
            )
            logger.info("  ✓ README.md pushed to hub")
            
        except Exception as e:
            logger.warning(f"Failed to upload README: {e}")
        
        # Final summary
        logger.info(f"\n{'='*70}")
        logger.info("Upload Complete!")
        logger.info(f"{'='*70}")
        logger.info(f"\n✓ Model successfully uploaded to Hugging Face")
        logger.info(f"\nRepository URL:")
        logger.info(f"  https://huggingface.co/{repo_id}")
        logger.info(f"\nYou can now use this model with:")
        logger.info(f"  from transformers import AutoModelForCausalLM")
        logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{repo_id}')")
        logger.info(f"\n{'='*70}\n")

        if cleanup_tmp_dir:
            shutil.rmtree(cleanup_tmp_dir, ignore_errors=True)
            logger.info(f"Temporary merged directory cleaned: {cleanup_tmp_dir}")
        
        return repo_id
        
    except Exception as e:
        if cleanup_tmp_dir:
            shutil.rmtree(cleanup_tmp_dir, ignore_errors=True)
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


def main(argv):
    args = parse_args(argv)
    upload_to_huggingface(
        args.model_path,
        args.repo_id,
        args.hf_token,
        base_model=args.base_model,
        method_name=args.method_name,
        save_dtype=args.save_dtype,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
