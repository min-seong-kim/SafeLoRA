# SafeLoRA
Github repo for NeurIPS 2024 paper "Safe LoRA: the Silver Lining of Reducing Safety Risks when Fine-tuning Large Language Models"

# How to Use?
You can use the ```SafeLoRA``` class in ```model.py``` and there is an example demonstrating how to use ```SafeLoRA```.

base model에 origin model을 넣고 aligned model에 safety FT model을 넣으면 됩니다.
safe-num-layers 에서 safelora를 적용시킬 layer 개수를 정합니다. llama 2 7B에서는 30개로 하니 적당했습니다.

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