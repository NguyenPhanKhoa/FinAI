"""
Step 4: Run inference using the OpenVINO IR model on the Intel NPU.

Supports interactive chat mode and single-prompt mode.
Usage:
  python scripts/04_run_inference.py                          # interactive mode
  python scripts/04_run_inference.py --prompt "Analyze ..."   # single prompt
  python scripts/04_run_inference.py --device CPU             # override device
"""

import argparse
import json
from pathlib import Path

import openvino_genai as ov_genai


SYSTEM_PROMPT = (
    "You are FinGPT, a financial AI assistant specialized in financial analysis, "
    "sentiment analysis, market forecasting, and financial text processing. "
    "Provide clear, accurate, and professional financial insights."
)


def build_prompt(user_input: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_input}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def create_pipeline(model_dir: str, device: str) -> ov_genai.LLMPipeline:
    print(f"Loading model from: {model_dir}")
    print(f"Target device: {device}")
    pipe = ov_genai.LLMPipeline(model_dir, device)
    print("Model loaded successfully.\n")
    return pipe


def generate(pipe: ov_genai.LLMPipeline, prompt: str, config: dict) -> str:
    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = config.get("max_new_tokens", 512)
    gen_config.temperature = config.get("temperature", 0.7)
    gen_config.top_p = config.get("top_p", 0.9)

    formatted = build_prompt(prompt)
    result = pipe.generate(formatted, gen_config)
    return result


def interactive_mode(pipe: ov_genai.LLMPipeline, config: dict):
    print("=" * 60)
    print("FinGPT Interactive Mode (NPU)")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        print("FinGPT: ", end="", flush=True)
        response = generate(pipe, user_input, config)
        print(response)


def main():
    parser = argparse.ArgumentParser(description="FinGPT NPU Inference")
    parser.add_argument("--prompt", type=str, help="Single prompt (skip interactive mode)")
    parser.add_argument("--device", type=str, help="Override inference device (NPU, CPU, GPU)")
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / "configs" / "model_config.json"
    with open(config_path) as f:
        config = json.load(f)

    project_root = Path(__file__).parent.parent
    model_dir = str(project_root / config["paths"]["openvino_model_dir"])
    device = args.device or config["inference"]["device"]

    pipe = create_pipeline(model_dir, device)

    if args.prompt:
        response = generate(pipe, args.prompt, config["inference"])
        print(f"FinGPT: {response}")
    else:
        interactive_mode(pipe, config["inference"])


if __name__ == "__main__":
    main()
