"""
OpenAI-compatible API server for FinGPT on Intel NPU.

Serves the OpenVINO IR model via a FastAPI endpoint compatible with
OpenAI's /v1/chat/completions format, enabling integration with
OpenClaw, LangChain, and other OpenAI-compatible clients.

Usage:
  python server.py                    # default: NPU, port 8000
  python server.py --device CPU       # fallback device
  python server.py --port 9000        # custom port
"""

import argparse
import json
import time
import uuid
from pathlib import Path

import openvino_genai as ov_genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn


# --- Config ---

config_path = Path(__file__).parent / "configs" / "model_config.json"
with open(config_path) as f:
    config = json.load(f)

SYSTEM_PROMPT = (
    "You are FinGPT, a financial AI assistant specialized in financial analysis, "
    "sentiment analysis, market forecasting, and financial text processing. "
    "Provide clear, accurate, and professional financial insights."
)

MODEL_NAME = "fingpt-llama3.1-8b-npu"

# --- App ---

app = FastAPI(title="FinGPT API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = None


# --- Schemas ---

class Message(BaseModel):
    model_config = {"extra": "allow"}
    role: str
    content: str | list | None = None


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str = MODEL_NAME
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list
    usage: dict


# --- Helpers ---

MAX_PROMPT_CHARS = 2000  # Keep system prompt small so NPU has room for generation


def get_text(content) -> str:
    """Extract text from message content (string or list of content blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def build_prompt(messages: list[Message]) -> str:
    parts = []
    system_msg = SYSTEM_PROMPT

    for msg in messages:
        text = get_text(msg.content)
        if msg.role == "system":
            # Truncate large system prompts (OpenClaw sends ~25K chars)
            if len(text) > MAX_PROMPT_CHARS:
                system_msg = (
                    SYSTEM_PROMPT
                    + "\n\nAdditional context:\n"
                    + text[:MAX_PROMPT_CHARS]
                )
                print(f"[server] Truncated system prompt from {len(text)} to {MAX_PROMPT_CHARS} chars")
            else:
                system_msg = text
        elif msg.role == "user":
            parts.append(
                f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
            )
        elif msg.role == "assistant":
            parts.append(
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{text}<|eot_id|>"
            )

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}<|eot_id|>"
        + "".join(parts)
        + "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    print(f"[server] Final prompt length: {len(prompt)} chars, messages: {len(messages)}")
    return prompt


# --- Endpoints ---

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = request.max_tokens or config["inference"]["max_new_tokens"]
    gen_config.temperature = request.temperature or config["inference"]["temperature"]
    gen_config.top_p = request.top_p or config["inference"]["top_p"]

    prompt = build_prompt(request.messages)

    if request.stream:
        # Streaming response (SSE) for OpenClaw TUI compatibility
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def generate_stream():
            response_text = pipe.generate(prompt, gen_config)
            print(f"[server] Stream generated: {len(response_text)} chars")
            # Send the full response as a single chunk
            chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": response_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            # Send stop chunk
            stop_chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    # Non-streaming response
    response_text = pipe.generate(prompt, gen_config)
    print(f"[server] Generated: {len(response_text)} chars")

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=MODEL_NAME,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    )


@app.post("/v1/responses")
def responses(request: dict):
    """OpenAI Responses API endpoint for OpenClaw compatibility."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Extract input from the responses API format
    input_data = request.get("input", "")
    model = request.get("model", MODEL_NAME)

    # input can be a string or a list of messages
    if isinstance(input_data, str):
        messages = [Message(role="user", content=input_data)]
    elif isinstance(input_data, list):
        messages = []
        for item in input_data:
            if isinstance(item, str):
                messages.append(Message(role="user", content=item))
            elif isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, list):
                    # Handle content array format
                    text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    content = "\n".join(text_parts) if text_parts else str(content)
                messages.append(Message(role=role, content=content))
    else:
        messages = [Message(role="user", content=str(input_data))]

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = request.get("max_output_tokens", config["inference"]["max_new_tokens"])
    gen_config.temperature = request.get("temperature", config["inference"]["temperature"])
    gen_config.top_p = request.get("top_p", config["inference"]["top_p"])

    prompt = build_prompt(messages)
    response_text = pipe.generate(prompt, gen_config)
    print(f"[server] Generated response: {len(response_text)} chars, preview: {response_text[:200]!r}")

    resp_id = f"resp-{uuid.uuid4().hex[:12]}"
    msg_id = f"msg-{uuid.uuid4().hex[:12]}"
    return {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "output": [
            {
                "type": "message",
                "id": msg_id,
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": response_text,
                    }
                ],
                "status": "completed",
            }
        ],
        "output_text": response_text,
        "status": "completed",
        "usage": {
            "input_tokens": 100,
            "output_tokens": max(len(response_text) // 4, 1),
            "total_tokens": 100 + max(len(response_text) // 4, 1),
        },
    }


@app.post("/v1/completions")
def completions(request: dict):
    """Legacy completions endpoint for compatibility."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt_text = request.get("prompt", "")
    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = request.get("max_tokens", config["inference"]["max_new_tokens"])
    gen_config.temperature = request.get("temperature", config["inference"]["temperature"])
    gen_config.top_p = request.get("top_p", config["inference"]["top_p"])

    formatted = build_prompt([Message(role="user", content=prompt_text)])
    response_text = pipe.generate(formatted, gen_config)

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "text": response_text,
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": app.state.device}


# --- Main ---

def main():
    global pipe

    parser = argparse.ArgumentParser(description="FinGPT OpenAI-compatible API Server")
    parser.add_argument("--device", type=str, help="Inference device (NPU, CPU, GPU)")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    device = args.device or config["inference"]["device"]
    model_dir = str(Path(__file__).parent / config["paths"]["openvino_model_dir"])

    print(f"Loading FinGPT model on {device}...")
    if device == "NPU":
        # Set MAX_PROMPT_LEN for NPU to handle OpenClaw's larger agent prompts
        npu_config = {"MAX_PROMPT_LEN": 8192}
        pipe = ov_genai.LLMPipeline(model_dir, device, **npu_config)
    else:
        pipe = ov_genai.LLMPipeline(model_dir, device)
    app.state.device = device
    print(f"Model loaded. Starting server on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
