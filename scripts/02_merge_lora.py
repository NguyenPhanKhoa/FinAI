"""
Step 2: Merge LoRA adapter weights into the base Llama 3.1 8B model.

This produces a single merged model that can then be converted to OpenVINO IR format.
"""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    config_path = Path(__file__).parent.parent / "configs" / "model_config.json"
    with open(config_path) as f:
        config = json.load(f)

    project_root = Path(__file__).parent.parent
    base_model_path = project_root / config["paths"]["base_model_dir"] / "llama3.1-8b"
    lora_path = project_root / config["paths"]["base_model_dir"] / "fingpt-lora"
    merged_path = project_root / config["paths"]["merged_model_dir"]

    merged_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(model, str(lora_path))

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {merged_path}")
    model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))

    print("Merge complete.")
    print("Next step: python scripts/03_convert_openvino.py")


if __name__ == "__main__":
    main()
