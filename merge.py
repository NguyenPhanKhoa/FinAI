"""
FinGPT Merge Script — run ONCE before starting the server.

Merges FinGPT LoRA adapter into Llama-2-7b-chat base model.
Output: models/fingpt-llama2-7b-merged/

Requirements:
  pip install peft transformers torch accelerate

HuggingFace login required (Llama 2 is a gated model):
  pip install huggingface_hub
  huggingface-cli login
  Then request access at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL   = "meta-llama/Llama-2-7b-chat-hf"
LORA_ADAPTER = "FinGPT/fingpt-mt_llama2-7b_lora"
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "models", "fingpt-llama2-7b-merged")

def main():
    print("[merge] Loading base model (this downloads ~14GB on first run)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("[merge] Applying FinGPT LoRA adapter...")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER)
    model = model.merge_and_unload()
    print("[merge] LoRA merged.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[merge] Saving to {OUTPUT_DIR} ...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[merge] Done! You can now start npu_server.py")

if __name__ == "__main__":
    main()
