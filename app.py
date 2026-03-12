"""
Gradio web UI for FinGPT NPU inference.

Connects to the local OpenAI-compatible API server (server.py) so you
don't need to load the model twice.

Usage:
  python app.py                          # default: server at localhost:8000
  python app.py --api-url http://...:9000  # custom server URL
  python app.py --share                  # create public Gradio link
"""

import argparse

import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000"

EXAMPLES = [
    "What is the current market sentiment for AAPL based on recent earnings?",
    "Analyze the financial impact of rising interest rates on tech stocks.",
    "Summarize the key risks mentioned in Tesla's latest 10-K filing.",
    "What factors should I consider when evaluating a company's P/E ratio?",
]


def respond(message: str, history: list) -> str:
    """Send message to the FinGPT API server and return the response."""
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    try:
        resp = requests.post(
            f"{API_URL}/v1/chat/completions",
            json={
                "model": "fingpt-llama3.1-8b-npu",
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.7,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.ConnectionError:
        return "Error: Cannot connect to FinGPT server. Make sure `python server.py` is running."
    except Exception as e:
        return f"Error: {e}"


def main():
    global API_URL

    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", type=str, default=API_URL, help="FinGPT API server URL")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    API_URL = args.api_url

    # Check server health
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        print(f"Connected to FinGPT server: {health['model']} on {health['device']}")
    except Exception:
        print(f"Warning: FinGPT server not reachable at {API_URL}")
        print("Start it with: python server.py")

    demo = gr.ChatInterface(
        fn=respond,
        title="FinGPT — Financial AI Assistant (Intel NPU)",
        description=(
            "Powered by FinGPT (Llama 3.1 8B + LoRA) running on Intel NPU "
            "via OpenVINO with INT4 quantization."
        ),
        examples=EXAMPLES,
    )

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
