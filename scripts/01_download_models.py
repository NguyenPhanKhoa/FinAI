"""
Step 1: Download the base Llama 3.1 8B model and FinGPT LoRA adapter from HuggingFace.

Prerequisites:
  - HF_TOKEN set in .env (needed for gated Llama 3.1 model)
  - Accept Llama 3.1 license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()


def main():
    config_path = Path(__file__).parent.parent / "configs" / "model_config.json"
    with open(config_path) as f:
        config = json.load(f)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("ERROR: Set HF_TOKEN in your .env file.")
        print("  1. Create a token at https://huggingface.co/settings/tokens")
        print("  2. Accept Llama 3.1 license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        sys.exit(1)

    base_dir = Path(__file__).parent.parent / config["paths"]["base_model_dir"]

    # Download base model
    base_model_path = base_dir / "llama3.1-8b"
    print(f"Downloading base model: {config['base_model']}")
    print(f"  -> {base_model_path}")
    snapshot_download(
        repo_id=config["base_model"],
        local_dir=str(base_model_path),
        token=hf_token,
    )
    print("Base model downloaded.\n")

    # Download LoRA adapter
    lora_path = base_dir / "fingpt-lora"
    print(f"Downloading LoRA adapter: {config['lora_model']}")
    print(f"  -> {lora_path}")
    snapshot_download(
        repo_id=config["lora_model"],
        local_dir=str(lora_path),
        token=hf_token,
    )
    print("LoRA adapter downloaded.\n")

    print("All models downloaded successfully.")
    print("Next step: python scripts/02_merge_lora.py")


if __name__ == "__main__":
    main()
