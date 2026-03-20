# FinAI — Báo Cáo Triển Khhai & Mô Tả Kỹ Thuật

**Phiên bản:** 1.1
**Ngày cập nhật:** 2026-03-20
**Phần cứng mục tiêu:** Intel Core Ultra 7 258V (Lunar Lake), 47 TOPS NPU, 32GB RAM, Intel Arc 140V GPU

---

# PHẦN 1: GIỚI THIỆU DỰ ÁN

## 1.1. FinAI là gì?

FinAI là hệ thống triển khai mô hình ngôn ngữ lớn FinGPT (dựa trên Llama 3.1 8B Instruct + LoRA fine-tune) trên Neural Processing Unit (NPU) tích hợp sẵn trong chip Intel Core Ultra thế hệ mới (Lunar Lake), sử dụng OpenVINO với lượng tử hóa INT4 symmetric.

Nói cách khác, FinAI cho phép chạy một mô hình AI tài chính 7 tỷ tham số **hoàn toàn trên phần cứng cục bộ** — không cần GPU rời, không cần cloud, không cần trả phí API.

```
Phần cứng chạy FinGPT:
┌─────────────────────────────────────────────┐
│  Intel Core Ultra 7 258V                    │
│  ├── NPU: 47 TOPS (chạy model)             │  ← Chi phí điện: ~7W
│  └── CPU + GPU tích hợp (hỗ trợ)          │
│  RAM: 32 GB                                 │
└─────────────────────────────────────────────┘
```

## 1.2. Dự án dành cho ai?

FinAI phục vụ hai nhóm người dùng chính:

| Nhóm | Mô tả | Cách dùng |
|---|---|---|
| **Nhà phát triển / Kỹ sư AI** | Cần triển khai mô hình ngôn ngữ tài chính trên phần cứng cục bộ thay vì phụ thuộc cloud. Mục tiêu: chạy offline, chi phí vận hành thấp, bảo mật dữ liệu. | Chạy pipeline scripts, tùy chỉnh server.py, tích hợp OpenClaw |
| **Chuyên viên tài chính / Phân tích viên** | Sử dụng giao diện Gradio hoặc OpenClaw TUI để truy vấn thông tin tài chính — phân tích cảm xúc thị trường, tóm tắt báo cáo tài chính, đánh giá rủi ro — mà không cần hiểu về AI/ML. | Dùng app.py hoặc OpenClaw TUI, không cần code |

## 1.3. Giá trị kinh tế & tác động xã hội

| Khía cạnh | Chi tiết | Tác động |
|---|---|---|
| **Giảm chi phí vận hành** | Không cần subscription API key (OpenAI/Anthropic). Inference chạy hoàn toàn trên NPU cục bộ — chi phí điện ~10–15W so với cloud GPU ~300W/session. | Giảm chi phí vận hành hàng tháng từ hàng trăm USD (cloud API) xuống gần bằng không. |
| **Bảo mật dữ liệu** | Dữ liệu tài chính nhạy cảm (báo cáo tài chính, chiến lược đầu tư, thông tin khách hàng) không rời khỏi máy local. | Tuân thủ quy định bảo mật dữ liệu doanh nghiệp (GDPR, NDPL...). Giảm rủi ro rò rỉ dữ liệu. |
| **Thúc đẩy AI cạnh tranh** | Minh chứng rằng NPU (vốn chỉ ~7W) có thể chạy LLM 7B thay vì phải dùng GPU rời (75W+). | Mở đường cho AI xanh (green AI), AI di động, AI edge computing. Giảm carbon footprint của AI. |
| **Tiếp cận công nghệ** | Cho phép cá nhân, doanh nghiệp vừa và nhỏ (SMB), thị trường mới nổi tiến cận LLM tài chính mà không cần hạ tầng cloud đắt đỏ. | Dân chủ hóa AI — bất kỳ ai có laptop Intel Lunar Lake đều có thể chạy. |
| **Tái sử dụng phần cứng** | Tận dụng NPU tích hợp sẵn trên laptop/PC Intel Lunar Lake — phần cứng thường bị bỏ phí vì không có ứng dụng sử dụng. | Giảm e-waste, tối ưu hóa vòng đời thiết bị. |

## 1.4. Tổng quan kỹ thuật

| Thành phần | Chi tiết |
|---|---|
| **Base model** | Meta Llama 3.1 8B Instruct (8 tỷ tham số) |
| **Fine-tune adapter** | FinGPT LoRA adapter (PEFT) — chuyên biệt tài chính |
| **Inference engine** | OpenVINO GenAI (openvino-genai >= 2025.0.0) |
| **Phần cứng** | Intel NPU (47 TOPS), Intel Arc 140V GPU, 32GB RAM |
| **Lượng tử hóa** | INT4 symmetric, ratio=1.0, group_size=-1 (channel-wise) |
| **Kích thước model** | ~8 GB (base) + ~8 GB (LoRA) → ~16 GB (merged FP16) → ~4.5 GB (INT4) |
| **Nền tảng** | Windows 11, Python 3.11 |
| **API** | OpenAI-compatible (FastAPI) — tương thích OpenClaw, LangChain, bất kỳ OpenAI client nào |

---

# PHẦN 2: HƯỚNG DẪN TRIỂN KHAI CHI TIẾT

> **Ghi chú về ảnh chụp màn hình:** Bên dưới mỗi bước đều có nhãn **[ẢNH CẦN CHỤP N]** đánh dấu vị trí cần chèn hình. Mô tả bên nhãn cho biết chính xác cần chụp gì và kết quả mong đợi.

## Bước 0 — Kiểm tra phần cứng trước khi bắt đầu

**Mục đích:** Xác nhận NPU Intel được nhận diện và driver hoạt động đúng.

```bash
python scripts/check_hardware.py
```

> **[ẢNH CẦN CHỤP 1]**
> **Mô tả:** Chụp kết quả đầu ra của lệnh trên. Màn hình mong đợi:
> - Hiển thị Intel NPU được nhận diện (tên device: NPU hoặc GNA/NPU)
> - Không có dòng lỗi màu đỏ
> - Hiển thị thông tin device inference mặc định (NPU)

---

## Bước 1 — Setup môi trường

**Mục đích:** Cài đặt Python 3.11, tạo virtual environment, cài tất cả dependencies.

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

> **[ẢNH CẦN CHỤP 2]**
> **Mô tả:** Chụp toàn bộ output cuối cùng của setup.ps1, đặc biệt:
> - Dòng hiển thị "Setup complete!" màu xanh
> - Danh sách 8 bước tiếp theo được liệt kê bên dưới

**Sau đó, tạo file `.env`:**

```bash
# Copy template
copy .env.example .env

# Mở file .env bằng Notepad và thêm HuggingFace token
notepad .env
```

Thêm dòng sau vào `.env`:
```
HF_TOKEN=hf_your_token_here
```

> **[ẢNH CẦN CHỤP 3]**
> **Mô tả:** Chụp nội dung file `.env` đã chỉnh sửa, hiển thị dòng `HF_TOKEN=...` (che giấu giá trị token thực bằng cách làm mờ hoặc che phần sau dấu `=`).

**Activate môi trường ảo:**

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Bước 2 — Xác nhận License Llama 3.1 trên HuggingFace

**Mục đích:** Llama 3.1 là gated model — phải đăng nhập và accept license mới được download.

Truy cập trình duyệt:
```
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

Đăng nhập → Accept License (nếu chưa accept).

> **[ẢNH CẦN CHỤP 4]**
> **Mô tả:** Chụp trang HuggingFace của Llama 3.1 8B Instruct cho thấy đã accept license thành công (có thể làm mờ phần token/API key hiển thị).

---

## Bước 3 — Download Models từ HuggingFace

**Mục đích:** Tải model gốc Llama 3.1 8B Instruct và FinGPT LoRA adapter về local.

```bash
python scripts/01_download_models.py
```

Quá trình này mất 10–30 phút tùy tốc độ internet.

> **[ẢNH CẦN CHỤP 5]**
> **Mô tả:** Chụp output cuối cùng của script, phải thấy:
> - Dòng "All models downloaded successfully."
> - Dòng "Next step: python scripts/02_merge_lora.py"

**Xác nhận file đã tải về:**

```bash
ls models/base/llama3.1-8b/
ls models/base/fingpt-lora/
```

> **[ẢNH CẦN CHỤP 5b]** *(tuỳ chọn, thay thế cho ảnh trên)*
> **Mô tả:** Chụp cửa sổ File Explorer hiển thị nội dung thư mục `models/base/` với hai thư mục con `llama3.1-8b` và `fingpt-lora`.

---

## Bước 4 — Merge LoRA Adapter vào Base Model

**Mục đích:** Gộp trọng số LoRA vào model gốc để tạo một model FP16 duy nhất (OpenVINO export không chấp nhận base + adapter tách rời).

```bash
python scripts/02_merge_lora.py
```

Quá trình này mất 5–15 phút tùy dung lượng RAM.

> **[ẢNH CẦN CHỤP 6]**
> **Mô tả:** Chụp output cuối cùng hiển thị:
> - "Merge complete."
> - "Next step: python scripts/03_convert_openvino.py"

> **Ghi chú:** Thư mục `models/merged/` sau bước này sẽ chiếm ~16 GB dung lượng ổ cứng (FP16).

---

## Bước 5 — Convert sang OpenVINO IR (INT4)

**Mục đích:** Chuyển model FP16 sang định dạng OpenVINO Intermediate Representation (IR) với lượng tử hóa INT4 symmetric — định dạng duy nhất NPU Intel hỗ trợ.

```bash
python scripts/03_convert_openvino.py
```

Quá trình này mất 10–30 phút.

> **[ẢNH CẦN CHỤP 7]**
> **Mô tả:** Chụp output cuối cùng hiển thị:
> - "Conversion complete."
> - Đường dẫn `models/openvino/` được lưu thành công

> **[ẢNH CẦN CHỤP 8]**
> **Mô tả:** Chụp cửa sổ File Explorer hiển thị thư mục `models/openvino/` chứa các file `.xml` và `.bin` (IR files) cùng kích thước file (mong đợi ~4.5 GB tổng cộng).

---

## Bước 6 — Chạy Inference

### 6A. Chế độ CLI tương tác (test nhanh)

```bash
python scripts/04_run_inference.py
```

Gõ câu hỏi tài chính để test.

> **[ẢNH CẦN CHỤP 9]**
> **Mô tả:** Chụp một cuộc hội thoại mẫu trên terminal:
> - Câu hỏi: "What is the current market sentiment for AAPL?"
> - Câu trả lời đầy đủ từ FinGPT

### 6B. Chạy API Server (cho Gradio / OpenClaw)

```bash
python server.py
```

Server khởi động trên port 8000.

> **[ẢNH CẦN CHỤP 10]**
> **Mô tả:** Chụp output của server khi khởi động thành công:
> - Dòng "Model loaded." hoặc tương tự
> - Dòng "Starting server on 0.0.0.0:8000"

**Kiểm tra health endpoint:**

```bash
curl http://127.0.0.1:8000/health
```

> **[ẢNH CẦN CHỤP 11]**
> **Mô tả:** Chụp kết quả JSON trả về từ `/health`, ví dụ:
> `{"status":"ok","model":"fingpt-llama3.1-8b-npu","device":"NPU"}`

---

## Bước 7 — Gradio Web UI

**Lưu ý:** Bước này cần server.py đang chạy ở bước 6B.

```bash
python app.py --api-url http://127.0.0.1:8000
```

Mở trình duyệt tại URL mà Gradio hiển thị (thường là `http://localhost:XXXXX`).

> **[ẢNH CẦN CHỤP 12]**
> **Mô tả:** Chụp giao diện Gradio hiển thị trên trình duyệt — phải thấy:
> - Tiêu đề "FinGPT — Financial AI Assistant (Intel NPU)"
> - Hộp chat input
> - Các câu hỏi mẫu (examples)

> **[ẢNH CẦN CHỤP 13]**
> **Mô tả:** Chụp kết quả sau khi gửi một câu hỏi mẫu trên Gradio — phải thấy câu trả lời đầy đủ từ FinGPT hiển thị trong giao diện chat.

---

## Bước 8 — Chạy Test Suite

**Lưu ý:** Bước này cần server.py đang chạy ở bước 6B.

```bash
python tests/test_fingpt.py
```

> **[ẢNH CẦN CHỤP 14]**
> **Mô tả:** Chụp kết quả cuối cùng của test suite, phải thấy:
> - Số cases passed / total (ví dụ: "14 passed, 0 failed")
> - Điểm số keyword match ratio
> - Thời gian chạy

**Test suite tiếng Việt (tuỳ chọn):**

```bash
python tests/test_fingpt_vi.py
```

---

# PHẦN 3: MÔ TẢ KỸ THUẬT TỪNG THÀNH PHẦN CODE

## 3.1. Cấu trúc thư mục project

```
FinAI/                          # Root directory
├── configs/
│   └── model_config.json       # ⬅ Single source of truth cho toàn bộ config
├── scripts/
│   ├── check_hardware.py       # Kiểm tra NPU driver + device
│   ├── 01_download_models.py   # Download HuggingFace models
│   ├── 02_merge_lora.py        # Merge LoRA vào base model
│   ├── 03_convert_openvino.py  # Export OpenVINO IR + INT4 quantization
│   └── 04_run_inference.py     # CLI inference tool
├── models/                     # (gitignored — không đẩy lên git)
│   ├── base/                   #   models/base/llama3.1-8b/ + fingpt-lora/
│   ├── merged/                 #   models/merged/ (FP16, ~16 GB)
│   └── openvino/               #   models/openvino/ (INT4, ~4.5 GB)
├── tests/
│   ├── test_fingpt.py          # English test suite (14 cases)
│   └── test_fingpt_vi.py       # Vietnamese test suite
├── server.py                   # FastAPI OpenAI-compatible server
├── app.py                      # Gradio web UI
├── setup.ps1                   # Windows setup script
├── run.ps1                     # Docker runner script
├── Dockerfile                  # Multi-stage Docker build
├── requirements.txt            # Python dependencies
├── CLAUDE.md                   # Claude Code project instructions
├── FinAI_Report_VN.md          # Báo cáo triển khai (tiếng Việt)
├── .env                        # HF_TOKEN (gitignored)
└── .env.example                # Template cho .env
```

## 3.2. `configs/model_config.json` — Single Source of Truth

File này chứa **toàn bộ config** của dự án. Tất cả scripts đều đọc từ đây — không hard-code đường dẫn hay tên model.

```json
{
  "base_model": "meta-llama/Llama-3.1-8B-Instruct",
  "lora_model": "FinGPT/fingpt-mt_llama3-8b_lora",
  "quantization": {
    "weight_format": "int4",
    "symmetric": true,
    "ratio": 1.0,
    "group_size": -1
  },
  "inference": {
    "device": "NPU",
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_prompt_tokens": 1024
  },
  "paths": {
    "base_model_dir": "models/base",
    "merged_model_dir": "models/merged",
    "openvino_model_dir": "models/openvino"
  }
}
```

| Trường | Ý nghĩa | Tác dụng |
|---|---|---|
| `base_model` | HuggingFace repo ID của Llama 3.1 8B Instruct | Model gốc chưa fine-tune — nền tảng để apply LoRA |
| `lora_model` | Repo ID của FinGPT LoRA adapter | Trọng số fine-tune chuyên biệt tài chính — được cộng thêm vào base model |
| `quantization.weight_format: int4` | Mỗi weight nén từ 16 bit xuống 4 bit | Giảm 4× kích thước model (16 GB → 4 GB), NPU xử lý INT4 trực tiếp |
| `quantization.symmetric: true` | Symmetric quantization (không có offset/channel riêng) | **Bắt buộc cho NPU Intel** — NPU không hỗ trợ asymmetric |
| `quantization.ratio: 1.0` | 100% weights nén INT4, không trộn FP16/INT4 | Đảm bảo tất cả layers đều chạy trên NPU với INT4 |
| `quantization.group_size: -1` | Channel-wise quantization (mỗi channel 1 scale factor) | Chất lượng cao hơn group quantization — dùng cho 7B+ models |
| `inference.device: NPU` | Device inference mặc định | Thay đổi bằng `--device CPU` nếu NPU lỗi |
| `inference.max_new_tokens: 512` | Số token tối đa model được sinh ra | Giới hạn độ dài response để tránh NPU generation quá lâu |
| `paths.*` | Đường dẫn tương đối từ project root | Gitignored — model files không đẩy lên git |

---

## 3.3. `scripts/01_download_models.py`

**Chức năng:** Tải base model và LoRA adapter từ HuggingFace về local.

**Dependencies:** `HF_TOKEN` trong `.env` (bắt buộc vì Llama 3.1 là gated model).

```
Input:
  ├── HF_TOKEN (từ .env)
  └── configs/model_config.json

Xử lý:
  huggingface_hub snapshot_download(
      repo_id="meta-llama/Llama-3.1-8B-Instruct",
      local_dir="models/base/llama3.1-8b/",
      token=HF_TOKEN
  )
  huggingface_hub snapshot_download(
      repo_id="FinGPT/fingpt-mt_llama3-8b_lora",
      local_dir="models/base/fingpt-lora/",
      token=HF_TOKEN
  )

Output:
  models/base/llama3.1-8b/       (~8 GB)  ← Base Llama model
  models/base/fingpt-lora/        (~1-2 GB) ← FinGPT LoRA adapter
```

**Tại sao cần HF_TOKEN?** Llama 3.1 là "gated model" — Meta yêu cầu đăng nhập + accept license trên HuggingFace trước khi download. Token xác thực quyền truy cập.

---

## 3.4. `scripts/02_merge_lora.py`

**Chức năng:** Gộp (merge) trọng số LoRA adapter vào base model thành một model FP16 duy nhất.

**Dependencies:** `transformers`, `peft`, `torch`.

```
Input:
  models/base/llama3.1-8b/       (base model FP16)
  models/base/fingpt-lora/       (LoRA adapter weights)

Xử lý:
  1. AutoModelForCausalLM.from_pretrained("models/base/llama3.1-8b/")
  2. PeftModel.from_pretrained(base_model, "models/base/fingpt-lora/")
  3. model.merge_and_unload()  ← Gộp trọng số adapter vào base weights

Output:
  models/merged/  (~16 GB FP16)
  ├── pytorch_model.bin / model.safetensors
  ├── config.json
  └── tokenizer files
```

**Tại sao phải merge?** OpenVINO export (bước 3) chỉ chấp nhận một model duy nhất. Nếu không merge, base model + adapter tách rời sẽ không export được.

**Tại sao FP16?** Merged model giữ độ chính xác cao (half-precision float 16-bit) trước khi lượng tử hóa INT4 ở bước tiếp theo.

---

## 3.5. `scripts/03_convert_openvino.py`

**Chức năng:** Chuyển model FP16 sang OpenVINO IR format với lượng tử hóa INT4 symmetric.

**Dependencies:** `optimum[openvino]`, `openvino>=2025.0.0`.

```
Input:
  models/merged/  (~16 GB FP16)

Xử lý:
  optimum.exporters.openvino CLI command:
  --model models/merged/
  --task text-generation-with-past
  --weight-format int4
  --sym                      ← symmetric quantization (REQUIRED for NPU)
  --ratio 1.0                ← 100% INT4
  --group-size -1            ← channel-wise (7B+ models)
  --output models/openvino/

Output:
  models/openvino/
  ├── openvino_model.xml     ← IR graph (network structure)
  ├── openvino_model.bin     ← quantized weights (~4.5 GB)
  └── tokenizer/             ← tokenizer files
```

**Tại sao INT4 symmetric là bắt buộc?**
- NPU Intel chỉ hỗ trợ INT4 symmetric — không hỗ trợ INT8 hay asymmetric quantization
- Symmetric = trọng số được biểu diễn dưới dạng `value = scale × quantized_value` (scale duy nhất, không offset riêng)
- Asymmetric = `value = scale × quantized_value + offset` (mỗi channel một offset) → NPU không hỗ trợ

**Tại sao `group_size: -1`?**
- `group_size = -1` nghĩa là channel-wise quantization: mỗi output channel có một scale factor riêng
- Cho chất lượng cao hơn so với group quantization (chia weights thành groups)
- Được khuyến nghị cho models 7B trở lên

---

## 3.6. `scripts/04_run_inference.py`

**Chức năng:** Tool CLI để chạy inference tương tác trên NPU.

**Dependencies:** `openvino-genai`.

```
Kiến trúc bên trong:

build_prompt(user_input)
  └── Ghép Llama 3.1 Instruct template vào user input
      Template:
      <|begin_of_text|>
      <|start_header_id|>system<|end_header_id|>
      {SYSTEM_PROMPT}<|eot_id|>
      <|start_header_id|>user<|end_header_id|>
      {user_input}<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>

create_pipeline(model_dir, device)
  └── ov_genai.LLMPipeline(model_dir, device)
      Tải OpenVINO IR model vào NPU

generate(pipe, prompt, config)
  └── pipe.generate(
         prompt,
         ov_genai.GenerationConfig(
             max_new_tokens=512,
             temperature=0.7,
             top_p=0.9
         )
     )
      Trả về: string text (câu trả lời)

interactive_mode()
  └── while True:
        user_input = input("You: ")
        if quit → break
        response = generate(pipe, user_input, config)
        print(response)
```

**Llama 3.1 Instruct Template là gì?**
Llama 3.1 không hiểu plain text thuần túy. Nó cần prompt được định dạng đặc biệt để nhận biết đâu là system prompt, đâu là user message, đâu là assistant response. Các token đặc biệt `<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>` đóng vai trò như "dấu phân cách" có cấu trúc.

---

## 3.7. `server.py` — FastAPI OpenAI-compatible API Server

**Chức năng:** Server REST API tương thích OpenAI, cho phép bất kỳ OpenAI client nào (Gradio, OpenClaw, LangChain, curl...) gọi FinGPT qua HTTP.

**Dependencies:** `fastapi`, `uvicorn`, `openvino-genai`.

### 3.7.1. Các endpoint chính

| Endpoint | HTTP Method | Mục đích | Ai dùng |
|---|---|---|---|
| `/v1/chat/completions` | POST | Chat completions chuẩn OpenAI | Gradio app.py, OpenAI SDK, OpenClaw |
| `/v1/responses` | POST | OpenAI Responses API | OpenClaw agent mode |
| `/v1/completions` | POST | Legacy completions endpoint | Tương thích ngược |
| `/v1/models` | GET | Liệt kê model available | OpenAI client discovery |
| `/health` | GET | Health check + device info | Giám sát, kiểm tra |

### 3.7.2. Luồng xử lý `/v1/chat/completions`

```
HTTP POST /v1/chat/completions
         │
         ▼
┌──────────────────────────────────────────┐
│ 1. ChatCompletionRequest (Pydantic)       │
│    model: str                             │
│    messages: list[Message]                │
│    max_tokens: int                        │
│    temperature: float                     │
│    stream: bool                           │
│    (extra="allow" → chấp nhận extra fields│
│    từ OpenClaw)                          │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ 2. build_prompt(messages)                 │
│    - get_text(msg.content) → extract str  │
│      từ str | list | None (OpenClaw gửi  │
│      cả 3 dạng)                          │
│    - Ghép Llama 3.1 template             │
│    - Truncate system prompt >2000 chars  │
│      (MAX_PROMPT_CHARS)                  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ 3. ov_genai.GenerationConfig()            │
│    max_new_tokens = request.max_tokens   │
│                    or config default     │
│    temperature = request.temperature     │
│    top_p = request.top_p                 │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ 4. pipe.generate(prompt, gen_config)      │
│    NPU: truyền MAX_PROMPT_LEN=8192       │
│    (tăng từ 1024 mặc định, vì OpenClaw  │
│    agent prompts dài hơn bình thường)    │
└──────────────┬───────────────────────────┘
               │
        stream=True?
        /         \
      Yes          No
       │            │
       ▼            ▼
┌──────────────┐  ┌─────────────────────────┐
│ Streaming SSE│  │ Non-streaming JSON       │
│ (xem 3.7.3) │  │ ChatCompletionResponse    │
└──────────────┘  └─────────────────────────┘
```

### 3.7.3. Biến `pipe` — Global Model Object

```python
pipe = None  # global

def main():
    global pipe
    device = args.device or config["inference"]["device"]
    model_dir = str(Path(__file__).parent / config["paths"]["openvino_model_dir"])

    if device == "NPU":
        # MAX_PROMPT_LEN=8192: tăng prompt limit từ 1024 mặc định
        # Vì OpenClaw gửi agent prompts dài (~2000+ tokens)
        npu_config = {"MAX_PROMPT_LEN": 8192}
        pipe = ov_genai.LLMPipeline(model_dir, device, **npu_config)
    else:
        pipe = ov_genai.LLMPipeline(model_dir, device)

    uvicorn.run(app, host=args.host, port=args.port)
```

**Tại sao `MAX_PROMPT_CHARS = 2000`?**
NPU có giới hạn bộ nhớ cho cả prompt + generation. Nếu system prompt quá dài, NPU không còn "room" để sinh response. Truncate về 2000 ký tự đảm bảo generation budget còn đủ (~1024 tokens generation capacity).

### 3.7.4. Streaming SSE (Server-Sent Events)

OpenClaw TUI yêu cầu streaming. Server không stream token-by-token (vì NPU inference không hỗ trợ real-time streaming), mà sinh toàn bộ response rồi gửi một lần qua SSE:

```
1. pipe.generate(prompt, config)  → sinh full response text
                                    (toàn bộ text trong RAM)
         │
         ▼
2. SSE Chunk 1:
   data: {"id": "...", "choices": [{"delta": {"content": FULL_TEXT}, ...}]}
         │
         ▼
3. SSE Chunk 2 (stop):
   data: {"id": "...", "choices": [{"delta": {}, "finish_reason": "stop"}]}
         │
         ▼
4. Final:
   data: [DONE]
```

→ OpenClaw đọc chunk 1 và hiển thị full text. Chunk 2 báo hiệu kết thúc.

### 3.7.5. Hàm `get_text()` — Xử lý content format

OpenClaw gửi message content dưới 3 dạng khác nhau:

```python
def get_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content          # Dạng 1: plain string
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))  # Dạng 2: list[dict]
        return "\n".join(parts)
    return str(content)         # Dạng 3: fallback
```

→ Đảm bảo server không crash bất kể OpenClaw gửi format nào.

---

## 3.8. `app.py` — Gradio Web UI

**Chức năng:** Giao diện web chat cho FinGPT, cho phép người dùng không biết code tương tác với model.

**Dependencies:** `gradio`, `requests`.

### 3.8.1. Hai chế độ hoạt động

```
CHẾ ĐỘ 1: app.py tự load model (KHÔNG khuyến nghị)
────────────────────────────────────────────────────
python app.py (không --api-url)

┌─────────────────────────────────────┐
│ app.py process                      │
│                                     │
│   ov_genai.LLMPipeline()  ← load ~4.5GB
│   gradio.ChatInterface()            │
│                                     │
│ RAM tiêu thụ: ~5 GB (model + UI)   │
└─────────────────────────────────────┘
→ Load model LẦN 2 trong process Gradio
→ Tốn RAM không cần thiết

CHẾ ĐỘ 2: app.py gọi server.py qua HTTP (KHUYẾN NGHỊ)
────────────────────────────────────────────────────────
python app.py --api-url http://127.0.0.1:8000

┌──────────────────┐    HTTP POST    ┌──────────────────────┐
│ app.py           │ ──────────────►│ server.py             │
│                  │                │ (đã load model rồi)   │
│ gr.ChatInterface │                │                       │
│ RAM: ~500 MB     │ ◄───────────── │ ov_genai.LLMPipeline  │
│                  │    JSON resp    │ RAM: ~4.5 GB          │
└──────────────────┘                └──────────────────────┘
→ Dùng chung model qua HTTP
→ Tiết kiệm RAM, ổn định hơn
```

### 3.8.2. Luồng xử lý `respond()`

```python
def respond(message: str, history: list) -> str:
    # 1. Build messages list từ Gradio chat history
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    # 2. POST to server.py
    resp = requests.post(
        f"{API_URL}/v1/chat/completions",
        json={
            "model": "fingpt-llama3.1-8b-npu",
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7,
        },
        timeout=120,  # NPU inference có thể mất vài phút
    )

    # 3. Return content từ response
    return resp.json()["choices"][0]["message"]["content"]
```

---

## 3.9. `setup.ps1` — Windows Setup Script

**Chức năng:** Tự động hóa việc setup môi trường trên Windows.

```
Bước 1: Kiểm tra Python có được cài không
         └─ Lưu ý: script check là Python 3.10+, nhưng dự án yêu cầu Python 3.11 CHÍNH XÁC
                    (Python 3.14 sẽ lỗi vì thiếu prebuilt wheels cho ML packages)

Bước 2: Tạo virtual environment
         python -m venv .venv

Bước 3: Activate venv
         .\.venv\Scripts\Activate.ps1

Bước 4: Upgrade pip + cài dependencies
         pip install --upgrade pip
         pip install -r requirements.txt
```

> ⚠️ **Lưu ý quan trọng:** `setup.ps1` kiểm tra Python version là `3.10+`, nhưng dự án FinAI yêu cầu **Python 3.11 chính xác**. Nếu máy có Python 3.14, các package ML sẽ lỗi vì thiếu prebuilt wheels.

---

# PHẦN 4: LƯU ĐỒ GIẢI THUẬT

## 4.1. Lưu đồ tổng quan: Pipeline 4 bước

```
┌─────────────────────────────────────────────────────────┐
│                         BẮT ĐẦU                         │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Bước 1: 01_download_models.py                          │
│  • HuggingFace snapshot_download                         │
│  • Tải Llama 3.1 8B Instruct → models/base/llama3.1-8b/│
│  • Tải FinGPT LoRA → models/base/fingpt-lora/           │
│  • Cần: HF_TOKEN (gated model)                           │
└─────────────────────────┬───────────────────────────────┘
                          ▼
                   Download thành công?
                    /                  \
                 Có                   Không → THOÁT
                  │                   (báo lỗi HF_TOKEN)
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Bước 2: 02_merge_lora.py                               │
│  • AutoModelForCausalLM.from_pretrained(base)          │
│  • PeftModel.from_pretrained(base, lora)               │
│  • model.merge_and_unload()  ← gộp weights              │
│  • Lưu: models/merged/ (~16 GB FP16)                   │
└─────────────────────────┬───────────────────────────────┘
                          ▼
                   Merge thành công?
                    /                  \
                 Có                   Không → THOÁT
                  │                   (báo lỗi RAM/model)
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Bước 3: 03_convert_openvino.py                          │
│  • optimum.exporters.openvino CLI                      │
│  • --weight-format int4 --sym --ratio 1.0               │
│  • --group-size -1 (channel-wise)                       │
│  • Lưu: models/openvino/*.xml + *.bin (~4.5 GB)        │
└─────────────────────────┬───────────────────────────────┘
                          ▼
                 Convert thành công?
                    /                  \
                 Có                   Không → THOÁT
                  │                   (báo lỗi optimum/OpenVINO)
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Bước 4: Inference / Serving                             │
│  • 04_run_inference.py  → CLI tương tác               │
│  • server.py            → API server                    │
│  • app.py               → Gradio web UI                │
│  • ov_genai.LLMPipeline(model_dir, device)             │
│  • pipe.generate(prompt, gen_config)                   │
└─────────────────────────┬───────────────────────────────┘
                          ▼
                   ┌───────┴───────┐
                   │   KẾT THÚC    │
                   └───────────────┘
```

---

## 4.2. Lưu đồ chi tiết: Inference từ HTTP Request đến NPU

```
┌─────────────────────────────────────────────────────────────────────┐
│ Client (Gradio / OpenClaw / curl)                                   │
│   POST /v1/chat/completions                                          │
│   {messages: [...], stream: true/false, ...}                         │
└──────────────────────────┬────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ server.py — Endpoint: @app.post("/v1/chat/completions")              │
│                                                                      │
│  1. Nhận ChatCompletionRequest (Pydantic model)                      │
│  2. Kiểm tra pipe is not None?                                       │
│     └─ Nếu None → HTTP 503 "Model not loaded"                        │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ server.py — Hàm build_prompt(messages)                               │
│                                                                      │
│  for msg in messages:                                                │
│    content = get_text(msg.content)  ← str | list | None → str       │
│                                                                      │
│    if role == "system":                                              │
│      if len(content) > MAX_PROMPT_CHARS(2000):                       │
│        system_msg = SYSTEM_PROMPT + content[:2000]                  │
│        # Truncate để đảm bảo NPU còn room cho generation             │
│      else:                                                           │
│        system_msg = content                                          │
│                                                                      │
│    elif role == "user":                                              │
│      ghép: <|start_header_id|>user<|end_header_id|>\n\n             │
│            {content}<|eot_id|>                                       │
│                                                                      │
│    elif role == "assistant":                                         │
│      ghép: <|start_header_id|>assistant<|end_header_id|>\n\n       │
│            {content}<|eot_id|>                                       │
│                                                                      │
│  final_prompt = "<|begin_of_text|>" + system block                  │
│               + user blocks + assistant blocks                       │
│               + "<|start_header_id|>assistant<|end_header_id|>\n\n"│
│                                                                      │
│  return final_prompt                                                 │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ server.py — Tạo GenerationConfig                                    │
│                                                                      │
│  gen_config = ov_genai.GenerationConfig()                           │
│  gen_config.max_new_tokens = request.max_tokens or 512              │
│  gen_config.temperature  = request.temperature or 0.7                │
│  gen_config.top_p        = request.top_p or 0.9                      │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ server.py — Inference                                                 │
│                                                                      │
│  response_text = pipe.generate(final_prompt, gen_config)             │
│                                                                      │
│  pipe = ov_genai.LLMPipeline(model_dir, device)                     │
│  Nếu device == "NPU":                                               │
│    pipe = ov_genai.LLMPipeline(..., MAX_PROMPT_LEN=8192)            │
│  # NPU inference: OpenVINO runtime gọi NPU driver                    │
│  # NPU thực hiện matrix multiplication trên hardware                 │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           │  stream?
           │  /       \
        Có              Không
         │               │
         ▼               ▼
┌──────────────────┐  ┌─────────────────────────────────────────────┐
│ Streaming SSE     │  │ Non-streaming Response                     │
│                   │  │                                            │
│  Chunk 1:         │  │ ChatCompletionResponse:                     │
│  data: {          │  │   id: "chatcmpl-xxx",                       │
│    choices: [{    │  │   choices: [{                               │
│      delta: {     │  │     message: {                              │
│        content:   │  │       role: "assistant",                    │
│          FULL_   │  │       content: FULL_TEXT                    │
│          TEXT    │  │     },                                       │
│      },          │  │     finish_reason: "stop"                   │
│      finish: null│  │   }],                                        │
│    }]             │  │   usage: {...}                              │
│  }                │  └─────────────────────────────────────────────┘
│                   │
│  Chunk 2:         │
│  data: {          │
│    choices: [{    │
│      delta: {},   │
│      finish: "stop"│
│    }]             │
│  }                │
│                   │
│  data: [DONE]     │
└────────┬──────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Client hiển thị kết quả                                              │
│   Gradio → hiển thị trong chat box                                   │
│   OpenClaw TUI → đọc SSE chunk và in ra terminal                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4.3. Lưu đồ: `build_prompt()` — Xây dựng Llama 3.1 Prompt

```
Input: messages = list of Message objects
       (từ OpenAI API request)

         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│ system_msg = ""                                                    │
│ user_blocks = []                                                   │
│ asst_blocks = []                                                   │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────┐
│ LOOP: for msg in messages                                          │
│                                                                     │
│   content = get_text(msg.content)                                  │
│   # content có thể là: str, list[dict], None                       │
│   # get_text() chuẩn hóa thành str                                │
│                                                                     │
│   ────────────────────────────────────────────                     │
│   IF msg.role == "system":                                         │
│                                                                     │
│     IF len(content) > 2000 (MAX_PROMPT_CHARS):                     │
│       system_msg = SYSTEM_PROMPT + "\n\n" +                        │
│                    "Additional context:\n" +                       │
│                    content[:2000]                                  │
│       # Giữ lại system prompt gốc, cắt phần additional            │
│       # Truncate để NPU còn room gen response                      │
│     ELSE:                                                          │
│       system_msg = content                                         │
│                                                                     │
│   ────────────────────────────────────────────                     │
│   ELIF msg.role == "user":                                        │
│                                                                     │
│     user_block =                                                   │
│       "<|start_header_id|>user<|end_header_id|>\n\n"              │
│       "{content}<|eot_id|>"                                       │
│     APPEND user_block TO user_blocks                              │
│                                                                     │
│   ────────────────────────────────────────────                     │
│   ELIF msg.role == "assistant":                                    │
│                                                                     │
│     asst_block =                                                   │
│       "<|start_header_id|>assistant<|end_header_id|>\n\n"         │
│       "{content}<|eot_id|>"                                       │
│     APPEND asst_block TO asst_blocks                              │
│                                                                     │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────┐
│ ASSEMBLE final_prompt:                                             │
│                                                                     │
│ final_prompt = (                                                    │
│   "<|begin_of_text|>"                                               │  ← Bắt đầu document
│   "<|start_header_id|>system<|end_header_id|>\n\n"                │
│   "{system_msg}<|eot_id|>"                                         │
│   + "".join(user_blocks)                                           │  ← Tất cả user messages
│   + "".join(asst_blocks)                                           │  ← Tất cả assistant messages
│   "<|start_header_id|>assistant<|end_header_id|>\n\n"            │  ← Dấu hiệu chờ assistant trả lời
│ )                                                                   │
│                                                                     │
│ RETURN final_prompt                                                 │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────┐
│ pipe.generate(final_prompt, gen_config)                             │
│   NPU inference → sinh tokens → decode → string text                │
└────────────────────────────────────────────────────────────────────┘
```

---

## 4.4. Lưu đồ: OpenClaw Integration

```
┌──────────────────────────────────────────────────────────────────────┐
│ Người dùng gõ lệnh trong OpenClaw TUI                                │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│ OpenClaw TUI đọc config: ~/.openclaw/openclaw.json                   │
│                                                                      │
│ {                                                                    │
│   "provider": "openai",                                             │
│   "api": "openai-completions",                                      │
│   "base_url": "http://127.0.0.1:8000/v1",                           │
│   "model": "fingpt-llama3.1-8b-npu"                                │
│ }                                                                    │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│ OpenClaw gửi HTTP POST /v1/responses (hoặc /v1/chat/completions)    │
│ Content-Type: application/json                                      │
│ Accept: text/event-stream  ← báo server gửi SSE                     │
│                                                                      │
│ Body:                                                                │
│ {                                                                    │
│   "model": "fingpt-llama3.1-8b-npu",                               │
│   "input": "...",                                                  │
│   "max_output_tokens": 512,                                         │
│   "stream": true                                                    │
│ }                                                                    │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│ server.py xử lý:                                                    │
│   1. /v1/responses endpoint (OpenAI Responses API)                   │
│   2. Extract "input" from request                                   │
│   3. build_prompt() → format Llama 3.1 template                     │
│   4. pipe.generate() → NPU inference                                │
│   5. Trả về SSE stream (xem 4.2)                                   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│ OpenClaw TUI nhận SSE chunks                                        │
│   Đọc chunk data: {delta: {content: "..."}}                       │
│   Cập nhật terminal: hiển thị text đang sinh ra                     │
│   Đọc chunk finish_reason: "stop" → kết thúc                       │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
                    OpenClaw TUI hiển thị
                    câu trả lời hoàn chỉnh
```

---

## 4.5. Lưu đồ: Gradio + Server Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ CÁCH 1: app.py tự load model (KHÔNG khuyến nghị)                     │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Terminal 1: python server.py  → KHÔNG CHẠY                          │
│                                                                      │
│ Terminal 2: python app.py                                            │
│                                                                      │
│ app.py process:                                                      │
│   ├── Gradio UI (listen port XXXXX)                                 │
│   ├── requests.post() → KHÔNG gọi server.py vì KHÔNG có --api-url  │
│   └── ov_genai.LLMPipeline() → load model vào app.py process        │
│       → RAM: ~4.5 GB (model) + ~500 MB (Gradio) = ~5 GB             │
│                                                                      │
│ ⚠️ Model được load 2 lần: app.py 1 lần, lần sau (nếu có)             │
│ ⚠️ Tốn RAM không cần thiết                                          │
└──────────────────────────────────────────────────────────────────────┘

---

┌──────────────────────────────────────────────────────────────────────┐
│ CÁCH 2: app.py gọi server.py qua HTTP (KHUYẾN NGHỊ)                 │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────┐    HTTP POST /v1/chat/completions  ┌─────────────────────┐
│ Terminal 1: python server.py     │ ─────────────────────────────────►│                     │
│                                 │                                     │  server.py          │
│   server.py:                    │    JSON response                   │                     │
│   ├── ov_genai.LLMPipeline()   │ ◄───────────────────────────────── │  - Load model       │
│   │   → NPU, ~4.5 GB RAM      │                                     │  - Inference        │
│   └── uvicorn (HTTP server)    │                                     │  - Return JSON      │
└─────────────────────────────────┘                                     └─────────────────────┘
                           ▲
                           │ HTTP
┌─────────────────────────────────┐
│ Terminal 2: python app.py       │
│ --api-url http://127.0.0.1:8000 │
│                                 │
│ app.py:                         │
│   gr.ChatInterface()           │  ← Gradio UI (listen port YYYYY)
│   respond()                    │  ← gọi requests.post() đến server.py
│   RAM: ~500 MB (Gradio only)  │
└─────────────────────────────────┘

→ Model load 1 lần duy nhất (ở server.py)
→ RAM app.py chỉ tốn ~500 MB cho Gradio UI
→ ✅ Ổn định, tiết kiệm RAM
```

---

# PHẦN 5: BẢNG TỔNG HỢP CÁC ẢNH CẦN CHỤP

| STT | Bước | File ảnh đề xuất | Mô tả nội dung |
|-----|------|-------------------|----------------|
| 1 | Bước 0 | `01_hardware_check.png` | Output `check_hardware.py` — NPU được nhận diện |
| 2 | Bước 1 | `02_setup_complete.png` | Output cuối `setup.ps1` — "Setup complete!" + 8 bước tiếp theo |
| 3 | Bước 1 | `03_env_file.png` | File `.env` hiển thị `HF_TOKEN=...` (che giá trị) |
| 4 | Bước 2 | `04_hf_license.png` | Trang HuggingFace Llama 3.1 đã accept license |
| 5 | Bước 3 | `05_download_success.png` | Output "All models downloaded successfully." |
| 5b | Bước 3 | `05b_models_folder.png` | *(Tuỳ chọn thay thế 5)* File Explorer `models/base/` |
| 6 | Bước 4 | `06_merge_complete.png` | Output "Merge complete." |
| 7 | Bước 5 | `07_convert_complete.png` | Output "Conversion complete." |
| 8 | Bước 5 | `08_openvino_folder.png` | File Explorer `models/openvino/` — file IR + kích thước |
| 9 | Bước 6A | `09_inference_chat.png` | Terminal CLI — câu hỏi + câu trả lời mẫu |
| 10 | Bước 6B | `10_server_startup.png` | Server output — "Model loaded." + "Starting server on..." |
| 11 | Bước 6B | `11_health_response.png` | JSON response từ `/health` endpoint |
| 12 | Bước 7 | `12_gradio_ui.png` | Gradio web UI trên trình duyệt |
| 13 | Bước 7 | `13_gradio_result.png` | Gradio hiển thị câu trả lời từ FinGPT |
| 14 | Bước 8 | `14_test_results.png` | Test suite output — passed/total + keyword scores |

---

# PHẦN 6: CẤU HÌNH OPENCLAW (TÍCH HỢP)

File cấu hình OpenClaw nằm ở `~/.openclaw/openclaw.json`:

```json
{
  "provider": "openai",
  "api": "openai-completions",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "fingpt-llama3.1-8b-npu"
}
```

> **[ẢNH CẦN CHỤP 15]** *(tuỳ chọn)*
> **Mô tả:** Chụp file `~/.openclaw/openclaw.json` đã cấu hình đúng, kèm output OpenClaw TUI đang gọi FinGPT thành công.

---

# PHẦN 7: XỬ LÝ SỰ CỐ THƯỜNG GẶP

| Vấn đề | Nguyên nhân | Giải pháp |
|---|---|---|
| `HF_TOKEN` not found | Chưa tạo `.env` hoặc chưa activate venv | Tạo `.env`, activate venv, restart terminal |
| NPU not detected | Driver chưa cài / chưa enable | Chạy `scripts/check_hardware.py`, cài Intel NPU driver |
| Out of memory khi merge | RAM < 32 GB, không đủ cho FP16 merge | Dùng `--device_map="cpu"` (đã có sẵn), đóng ứng dụng khác |
| Convert fail — asymmetric error | Config sai symmetric/ratio | Kiểm tra `model_config.json`: `symmetric: true`, `ratio: 1.0` |
| Server 503 — Model not loaded | Server chưa khởi động hoặc NPU lỗi | Chạy `python server.py --device CPU` thay vì NPU |
| OpenClaw không hiển thị text | SSE streaming lỗi | Kiểm tra `/health`, restart server, thử non-streaming |
| Gradio — Error: Cannot connect | app.py không tìm thấy server | Chạy `server.py` trước, dùng `--api-url` đúng |
| Python 3.14 — ModuleNotFoundError | Thiếu prebuilt wheels | Cài Python 3.11, xóa .venv, chạy lại `setup.ps1` |

---

*Bản quyền © 2026. Nội dung có thể được sử dụng cho mục đích học tập và nghiên cứu.*
