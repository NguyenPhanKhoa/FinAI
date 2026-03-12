"""
Gradio web UI for FinGPT NPU inference.

Usage:
  python app.py
  python app.py --device CPU    # override device
  python app.py --share         # create public link
"""

import argparse
import json
from pathlib import Path

import gradio as gr
import openvino_genai as ov_genai

SYSTEM_PROMPT = (
    "You are FinGPT, a financial AI assistant specialized in financial analysis, "
    "sentiment analysis, market forecasting, and financial text processing. "
    "Provide clear, accurate, and professional financial insights."
)

pipe = None
inference_config = None


def build_prompt(user_input: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_input}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def respond(message: str, history: list) -> str:
    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = inference_config.get("max_new_tokens", 512)
    gen_config.temperature = inference_config.get("temperature", 0.7)
    gen_config.top_p = inference_config.get("top_p", 0.9)

    formatted = build_prompt(message)
    return pipe.generate(formatted, gen_config)


EXAMPLES = [
    "What is the current market sentiment for AAPL based on recent earnings?",
    "Analyze the financial impact of rising interest rates on tech stocks.",
    "Summarize the key risks mentioned in Tesla's latest 10-K filing.",
    "What factors should I consider when evaluating a company's P/E ratio?",
]


def main():
    global pipe, inference_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, help="Inference device (NPU, CPU, GPU)")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    config_path = Path(__file__).parent / "configs" / "model_config.json"
    with open(config_path) as f:
        config = json.load(f)

    inference_config = config["inference"]
    model_dir = str(Path(__file__).parent / config["paths"]["openvino_model_dir"])
    device = args.device or config["inference"]["device"]

    print(f"Loading model on {device}...")
    pipe = ov_genai.LLMPipeline(model_dir, device)
    print("Model loaded.")

    demo = gr.ChatInterface(
        fn=respond,
        title="FinGPT — Financial AI Assistant (Intel NPU)",
        description=(
            "Powered by FinGPT (Llama 3.1 8B + LoRA) running on Intel NPU "
            "via OpenVINO with INT4 quantization."
        ),
        examples=EXAMPLES,
        theme=gr.themes.Soft(),
    )

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
