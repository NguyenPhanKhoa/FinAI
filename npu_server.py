"""
FinGPT NPU Inference Server — powered by IPEX-LLM
Runs FinGPT (Llama-2-7b merged) INT4 on Intel AI Boost NPU
Fully Ollama-compatible API at http://localhost:11435

Targets Intel Core Ultra 7 258V (Lunar Lake):
  - Primary:  NPU  (Intel AI Boost, 48 TOPS)
  - Fallback: CPU  (INT4, still fast)

Run merge.py once before starting this server.
"""

import os, time, uuid, torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "fingpt-llama2-7b-merged")
PORT = 11435

FINGPT_SYSTEM_PROMPT = """You are FinGPT, a specialized financial AI assistant. You help with:
- Financial sentiment analysis (respond: positive / neutral / negative)
- Stock price movement forecasting (respond: up / down / stable)
- Trading signal generation (respond: BUY / SELL / HOLD + brief reason)
- Financial Q&A and market analysis

Always respond concisely and accurately."""

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)

    # Try NPU first via IPEX-LLM NPU module
    try:
        print("[FinGPT] Trying NPU via ipex-llm...")
        from ipex_llm.transformers.npu_model import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            load_in_low_bit="sym_int4",
            optimize_model=True,
            use_cache=True,
            trust_remote_code=False,
        )
        print("[FinGPT] Loaded on NPU OK")
        return model, tokenizer, "NPU"
    except Exception as e:
        print(f"[FinGPT] NPU failed: {e}")

    # Fallback: CPU with INT4
    print("[FinGPT] Falling back to CPU INT4...")
    from ipex_llm.transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        load_in_low_bit="sym_int4",
        trust_remote_code=False,
    )
    print("[FinGPT] Loaded on CPU OK")
    return model, tokenizer, "CPU"

# ── Prompt formatting (Llama-2 chat template) ─────────────────────────────────

def build_prompt(messages):
    system = FINGPT_SYSTEM_PROMPT
    turns = []

    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system = content
        elif role in ("user", "assistant"):
            turns.append((role, content))

    # Llama-2 chat format
    prompt = ""
    for i, (role, content) in enumerate(turns):
        if role == "user":
            sys_block = f"<<SYS>>\n{system}\n<</SYS>>\n\n" if i == 0 else ""
            prompt += f"<s>[INST] {sys_block}{content} [/INST] "
        elif role == "assistant":
            prompt += f"{content} </s>"

    # If last turn was user, leave it open for the model
    if not turns or turns[-1][0] == "assistant":
        prompt += "<s>[INST] "

    return prompt

# ── Inference helper ──────────────────────────────────────────────────────────

def generate(prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

# ── Boot ──────────────────────────────────────────────────────────────────────

print("[FinGPT] Loading model...")
model, tokenizer, active_device = load_model()
print(f"[FinGPT] Ready on {active_device}")

app = FastAPI(title="FinGPT NPU Server")

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": active_device, "model": "fingpt-llama2-7b"}

# ── Ollama API ────────────────────────────────────────────────────────────────

@app.get("/api/tags")
def ollama_tags():
    return {
        "models": [{
            "name": "fingpt:latest",
            "model": "fingpt:latest",
            "modified_at": "2026-03-12T00:00:00Z",
            "size": 3800000000,
            "digest": "fingpt-llama2-7b-int4",
            "details": {
                "family": "llama",
                "parameter_size": "7B",
                "quantization_level": "INT4"
            }
        }]
    }

@app.post("/api/chat")
async def ollama_chat(request: Request):
    body = await request.json()
    messages  = body.get("messages", [])
    options   = body.get("options", {})
    prompt    = build_prompt(messages)
    result    = generate(
        prompt,
        max_new_tokens=options.get("num_predict", 512),
        temperature=options.get("temperature", 0.2),
    )
    return JSONResponse({
        "model": "fingpt:latest",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": result},
        "done": True,
        "done_reason": "stop",
    })

@app.post("/api/generate")
async def ollama_generate(request: Request):
    body      = await request.json()
    options   = body.get("options", {})
    result    = generate(
        body.get("prompt", ""),
        max_new_tokens=options.get("num_predict", 512),
        temperature=options.get("temperature", 0.2),
    )
    return JSONResponse({"model": "fingpt:latest", "response": result, "done": True})

# ── OpenAI-compatible API ─────────────────────────────────────────────────────

@app.get("/v1/models")
def openai_models():
    return {
        "object": "list",
        "data": [{"id": "fingpt", "object": "model", "created": int(time.time()), "owned_by": "local"}],
    }

@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    body   = await request.json()
    prompt = build_prompt(body.get("messages", []))
    result = generate(
        prompt,
        max_new_tokens=body.get("max_tokens", 512),
        temperature=body.get("temperature", 0.2),
    )
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "fingpt",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })

if __name__ == "__main__":
    print(f"[FinGPT] Server on http://localhost:{PORT} | Device: {active_device}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
