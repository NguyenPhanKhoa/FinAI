# FinAI — Triển Khai FinGPT trên Phần Cứng Intel NPU/GPU Cục Bộ

**Phiên bản:** 1.3
**Ngày cập nhật:** 2026-03-20
**Phần cứng mục tiêu:** Intel Core Ultra 7 258V (Lunar Lake), 47 TOPS NPU, 32GB RAM, Intel Arc GPU

---

# MỤC LỤC

1. [Giới thiệu dự án](#1-giới-thiệu-dự-án)
   - 1.1. Tổng quan FinAI
   - 1.2. Mục tiêu và phạm vi
   - 1.3. Đối tượng phục vụ
   - 1.4. Giá trị kinh tế và tác động xã hội
2. [Tổng quan kiến trúc hệ thống](#2-tổng-quan-kiến-trúc-hệ-thống)
   - 2.1. Sơ đồ kiến trúc tổng thể
   - 2.2. Mô hình FinGPT
   - 2.3. Inference engine — OpenVINO GenAI
   - 2.4. Phần cứng triển khai
3. [Hướng dẫn triển khai](#3-hướng-dẫn-triển-khai)
   - 3.1. Triển khai bằng Docker *(Khuyến nghị)*
   - 3.2. Triển khai Native
   - 3.3. Khởi động và kiểm tra
4. [Mô tả kỹ thuật từng thành phần](#4-mô-tả-kỹ-thuật-từng-thành-phần)
   - 4.1. Cấu trúc thư mục project
   - 4.2. Cấu hình — `configs/model_config.json`
   - 4.3. Download models — `01_download_models.py`
   - 4.4. Merge LoRA — `02_merge_lora.py`
   - 4.5. Convert OpenVINO — `03_convert_openvino.py`
   - 4.6. Inference — `04_run_inference.py`
   - 4.7. API Server — `server.py`
   - 4.8. Giao diện Web — `app.py`
   - 4.9. Setup Script — `setup.ps1`
5. [Lưu đồ giải thuật](#5-lưu-đồ-giải-thuật)
   - 5.1. Pipeline 4 bước
   - 5.2. Inference từ HTTP Request
   - 5.3. Xây dựng Llama 3.1 Prompt
   - 5.4. OpenClaw Integration
   - 5.5. Gradio + Server Architecture
6. [Tài liệu tham khảo hình ảnh](#6-tài-liệu-tham-khảo-hình-ảnh)

---

# DANH MỤC HÌNH

| STT | Định danh | Mô tả |
|-----|-----------|--------|
| 1 | Hình 1 | Docker Desktop đang chạy |
| 2 | Hình 2 | File `.env` đã lưu với `HF_TOKEN=...` |
| 3 | Hình 3 | HuggingFace License đã được accept |
| 4 | Hình 4 | Output "Pipeline complete!" |
| 5 | Hình 5 | Server đang chạy tại `http://localhost:8000` |
| 6 | Hình 6 | *(Tuỳ chọn)* OpenClaw gọi FinGPT thành công |

> **Lưu ảnh vào thư mục `images/`** trong project FinAI, đặt đúng tên file: `images/01-docker-running.png`, `images/02-env-file.png`, ...

---

# DANH MỤC BẢNG

| STT | Định danh | Mô tả |
|-----|-----------|--------|
| 1 | Bảng 1 | Tổng quan kỹ thuật hệ thống |
| 2 | Bảng 2 | So sánh thiết bị inference |
| 3 | Bảng 3 | Trường cấu hình trong `model_config.json` |
| 4 | Bảng 4 | API Endpoints của server |
| 5 | Bảng 5 | Troubleshooting thường gặp |

---

# DANH MỤC TỪ VIẾT TẮT

| Từ viết tắt | Giải nghĩa |
|---|---|
| **API** | Application Programming Interface |
| **Docker** | Nền tảng đóng gói ứng dụng vào container |
| **FinGPT** | Financial GPT — LLM fine-tuned cho tài chính |
| **FP16** | Half-Precision Floating Point (16-bit) |
| **GPU** | Graphics Processing Unit |
| **HF** | HuggingFace |
| **INT4** | 4-bit Integer quantization |
| **LLM** | Large Language Model |
| **LoRA** | Low-Rank Adaptation |
| **NPU** | Neural Processing Unit |
| **OpenVINO** | Open Visual Inference and Neural network Optimization |
| **RAM** | Random Access Memory |
| **SSE** | Server-Sent Events |
| **TOPS** | Trillion Operations Per Second |

---

# 1. GIỚI THIỆU DỰ ÁN

## 1.1. Tổng quan FinAI

FinAI là hệ thống triển khai mô hình ngôn ngữ lớn FinGPT (dựa trên Llama 3.1 8B Instruct + LoRA fine-tune) trên phần cứng Intel cục bộ — **GPU** (ưu tiên), **NPU**, hoặc **CPU** (fallback) — sử dụng OpenVINO với lượng tử hóa INT4 symmetric.

Nói cách khác, FinAI cho phép chạy một mô hình AI tài chính 7 tỷ tham số **hoàn toàn trên phần cứng cục bộ** — không cần cloud, không cần trả phí API.

```
Phần cứng chạy FinGPT (ưu tiên GPU):
┌─────────────────────────────────────────────────┐
│  Intel Core Ultra 7 258V                        │
│  ├── GPU — TỐC ĐỘ NHANH NHẤT                 │  ← Ưu tiên dùng
│  ├── NPU — Tiết kiệm điện                    │  ← Low power inference
│  └── CPU — fallback khi GPU/NPU lỗi          │
│  RAM: 32 GB                                     │
└─────────────────────────────────────────────────┘
```

> **GPU nhanh hơn CPU ~5-10×.** NPU tiết kiệm điện hơn nhưng chậm hơn GPU.

## 1.2. Mục tiêu và phạm vi

**Mục tiêu chính:**
- Triển khai FinGPT (Llama 3.1 8B + FinGPT LoRA) trên phần cứng Intel cục bộ (NPU/GPU)
- Cung cấp API tương thích OpenAI để tích hợp với các công cụ AI Agent
- Hỗ trợ inference tài chính: phân tích cảm xúc, tóm tắt báo cáo, đánh giá rủi ro

**Phạm vi:**
- Phần cứng: Intel Core Ultra (Lunar Lake) với NPU, Intel Arc GPU
- Nền tảng: Windows 10/11, Python 3.11, Docker Desktop
- Model: Llama 3.1 8B Instruct + FinGPT LoRA (INT4 symmetric)

## 1.3. Đối tượng phục vụ

| Nhóm | Mô tả | Cách dùng |
|---|---|---|
| **Nhà phát triển / Kỹ sư AI** | Cần triển khai mô hình ngôn ngữ tài chính trên phần cứng cục bộ thay vì phụ thuộc cloud. Mục tiêu: chạy offline, chi phí vận hành thấp, bảo mật dữ liệu. | Chạy pipeline scripts, tùy chỉnh server.py, tích hợp OpenClaw |
| **Chuyên viên tài chính / Phân tích viên** | Sử dụng giao diện Gradio hoặc OpenClaw TUI để truy vấn thông tin tài chính — phân tích cảm xúc thị trường, tóm tắt báo cáo tài chính, đánh giá rủi ro — mà không cần hiểu về AI/ML. | Dùng app.py hoặc OpenClaw TUI, không cần code |

## 1.4. Giá trị kinh tế và tác động xã hội

| Khía cạnh | Chi tiết | Tác động |
|---|---|---|
| **Giảm chi phí vận hành** | Không cần subscription API key (OpenAI/Anthropic). Inference chạy hoàn toàn trên GPU / NPU cục bộ — chi phí điện thấp so với cloud GPU ~300W/session. | Giảm chi phí vận hành hàng tháng từ hàng trăm USD (cloud API) xuống gần bằng không. |
| **Bảo mật dữ liệu** | Dữ liệu tài chính nhạy cảm (báo cáo tài chính, chiến lược đầu tư, thông tin khách hàng) không rời khỏi máy local. | Tuân thủ quy định bảo mật dữ liệu doanh nghiệp (GDPR, NDPL...). Giảm rủi ro rò rỉ dữ liệu. |
| **Thúc đẩy AI cạnh tranh** | Minh chứng rằng GPU / NPU tích hợp sẵn có thể chạy LLM 7B thay vì phải dùng GPU rời (75W+). | Mở đường cho AI xanh (green AI), AI di động, AI edge computing. Giảm carbon footprint của AI. |
| **Tiếp cận công nghệ** | Cho phép cá nhân, doanh nghiệp vừa và nhỏ (SMB), thị trường mới nổi tiến cận LLM tài chính mà không cần hạ tầng cloud đắt đỏ. | Dân chủ hóa AI — bất kỳ ai có laptop Intel Lunar Lake đều có thể chạy. |
| **Tái sử dụng phần cứng** | Tận dụng NPU tích hợp sẵn trên laptop/PC Intel Lunar Lake — phần cứng thường bị bỏ phí vì không có ứng dụng sử dụng. | Giảm e-waste, tối ưu hóa vòng đời thiết bị. |

---

# 2. TỔNG QUAN KIẾN TRÚC HỆ THỐNG

## 2.1. Sơ đồ kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                      │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Gradio     │    │   OpenClaw   │    │  OpenAI SDK  │                  │
│  │   Web UI     │    │     TUI      │    │   / curl     │                  │
│  │ (localhost)  │    │   (agent)    │    │   (client)   │                  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│         │                   │                   │                           │
│         │  HTTP POST        │  HTTP POST        │  HTTP POST                │
│         │  /v1/chat/completions  or  /v1/responses                       │
└─────────┼───────────────────┼───────────────────┼───────────────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API SERVER LAYER                                  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     server.py — FastAPI                               │   │
│  │                                                                       │   │
│  │  POST /v1/chat/completions    ← OpenAI-compatible endpoint           │   │
│  │  POST /v1/responses            ← OpenAI Responses API                 │   │
│  │  POST /v1/completions          ← Legacy completions                   │   │
│  │  GET  /v1/models               ← Model discovery                     │   │
│  │  GET  /health                  ← Health check                        │   │
│  │                                                                       │   │
│  │  build_prompt() → Llama 3.1 template formatting                       │   │
│  │  pipe.generate() → OpenVINO GenAI inference                           │   │
│  └──────────────────────────────────┬───────────────────────────────────┘   │
└─────────────────────────────────────┼───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE LAYER                                   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │              openvino-genai — LLMPipeline(model_dir, device)         │   │
│  │                                                                       │   │
│  │   Device Priority:  GPU  →  NPU  →  CPU                             │   │
│  │                                                                       │   │
│  │   ├── GPU: Intel Arc 140V 16GB — Fastest (7.4s load, 13.4s/gen)    │   │
│  │   ├── NPU: Intel NPU 47 TOPS (Lunar Lake) — Low power               │   │
│  │   └── CPU: x86-64 — Fallback only                                   │   │
│  │                                                                       │   │
│  │   Model: OpenVINO IR INT4 symmetric (~4.5 GB)                       │   │
│  │   Quantization: INT4, symmetric=true, ratio=1.0, group_size=-1       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2.2. Mô hình FinGPT

```
Meta Llama 3.1 8B Instruct (base)  +  FinGPT LoRA adapter
                    ↓ merge (PEFT)
              Merged FP16 model (~16 GB)
                    ↓ convert (optimum-cli)
        OpenVINO IR + INT4 symmetric quantization (~4.5 GB)
                    ↓ deploy
              Intel GPU / NPU / CPU via openvino-genai
```

| Thành phần | Chi tiết | Kích thước |
|---|---|---|
| **Base model** | Meta Llama 3.1 8B Instruct (8 tỷ tham số) | ~8 GB |
| **Fine-tune adapter** | FinGPT LoRA adapter (PEFT) — chuyên biệt tài chính | ~1-2 GB |
| **Merged model** | FP16 after merge | ~16 GB |
| **OpenVINO IR** | INT4 symmetric quantized | ~4.5 GB |

## 2.3. Inference engine — OpenVINO GenAI

OpenVINO GenAI là runtime inference của Intel, cho phép deploy LLM lên NPU/GPU/CPU với lượng tử hóa INT4.

**Tại sao INT4 symmetric?**
- NPU Intel chỉ hỗ trợ INT4 symmetric — không hỗ trợ INT8 hay asymmetric
- Symmetric: `weight = scale × quantized_value` (một scale factor duy nhất)
- Asymmetric: `weight = scale × quantized_value + offset` → NPU không hỗ trợ

**Lượng tử hóa:**
- `weight_format: int4` — nén từ 16-bit xuống 4-bit (giảm 4× kích thước)
- `symmetric: true` — bắt buộc cho NPU Intel
- `ratio: 1.0` — 100% weights nén INT4, không trộn FP16/INT4
- `group_size: -1` — channel-wise quantization (chất lượng cao nhất cho 7B+)

## 2.4. Phần cứng triển khai

**Bảng 2 — So sánh thiết bị inference:**

| Thiết bị | Tốc độ | Mức tiêu thụ | Ghi chú |
|---|---|---|---|
| **GPU** | Nhanh nhất | ~25-40W | Intel Arc 140V 16GB — ưu tiên trên máy này |
| **NPU** | Trung bình | ~7-10W | Intel Core Ultra 7 258V (47 TOPS) — tiết kiệm điện |
| **CPU** | Chậm nhất | ~30-50W | Fallback khi GPU/NPU không khả dụng |

---

# 3. HƯỚNG DẪN TRIỂN KHAI

## 3.1. Triển khai bằng Docker *(Khuyến nghị)*

### 3.1.1. Bước 0 — Cài đặt Docker Desktop

Tải và cài Docker Desktop:
```
https://docker.com/desktop
```

Sau khi cài xong, mở Docker Desktop và đợi cho đến khi thấy icon "Docker is running".

> **Hình 1: Docker Desktop đang chạy**
> ![Hình 1](images/01-docker-running.png)

### 3.1.2. Bước 1 — Cấu hình HuggingFace Token

Mở PowerShell và chạy:
```powershell
copy .env.example .env; notepad .env
```

Sau đó thêm dòng này vào file `.env`:
```
HF_TOKEN=hf_YOUR_TOKEN_HERE
```

> **Hình 2: File `.env` đã lưu với `HF_TOKEN=...`**
> ![Hình 2](images/02-env-file.png)

### 3.1.3. Bước 2 — Accept Llama 3.1 License

Mở trình duyệt:
```
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

Đăng nhập → **Accept License**

> **Hình 3: Đã accept license thành công**
> ![Hình 3](images/03-hf-license.png)

### 3.1.4. Bước 3 — Chạy Pipeline

```powershell
.\run.ps1 -FullPipeline
```

Docker tự động làm hết: download → merge → convert. Script tự detect và skip các bước đã hoàn thành.

> **Hình 4: Output "Pipeline complete!"**
> ![Hình 4](images/04-pipeline-complete.png)

### 3.1.5. Bước 4 — Khởi động FinGPT

```powershell
.\run.ps1 -Server
```

Mở trình duyệt: `http://localhost:8000`

> **Hình 5: Server đang chạy**
> ![Hình 5](images/05-server-running.png)

### 3.1.6. Các lệnh bổ sung

| Lệnh | Mô tả |
|-------|--------|
| `.\run.ps1` | Menu tương tác |
| `.\run.ps1 -Inference` | Test nhanh CLI (auto-detect GPU/NPU/CPU) |
| `.\run.ps1 -Server` | Chạy API Server |
| `.\run.ps1 -FullPipeline -CleanUp` | Pipeline + xoá model cũ sau mỗi bước |

## 3.2. Triển khai Native

### 3.2.1. Bước 0 — Clone/Download Project

```powershell
git clone https://github.com/NguyenPhanKhoa/FinAI.git
cd FinAI
```

Hoặc tải ZIP từ GitHub và giải nén.

### 3.2.2. Bước 1 — Thêm HuggingFace Token

```powershell
copy .env.example .env
notepad .env
```

Thêm dòng sau vào `.env`:
```
HF_TOKEN=hf_your_token_here
```

### 3.2.3. Bước 2 — Accept Llama 3.1 License

Truy cập trình duyệt:
```
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

Đăng nhập → Accept License.

### 3.2.4. Bước 3 — Setup môi trường

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
.\.venv\Scripts\Activate.ps1
```

### 3.2.5. Bước 4 — Chạy Pipeline

```powershell
python scripts/01_download_models.py
python scripts/02_merge_lora.py
python scripts/03_convert_openvino.py
```

### 3.2.6. Bước 5 — Khởi động FinGPT

```powershell
python scripts/04_run_inference.py       # CLI
python server.py                         # API Server
```

Mở trình duyệt: `http://localhost:8000`

## 3.3. Khởi động và kiểm tra

Chạy script tự động kiểm tra:
```powershell
python scripts/00_prepare.py
```

Script sẽ kiểm tra: Python version, Docker, HF_TOKEN, License.

---

# 4. MÔ TẢ KỸ THUẬT TỪNG THÀNH PHẦN

## 4.1. Cấu trúc thư mục project

```
FinAI/
├── configs/
│   └── model_config.json       # ⬅ Single source of truth cho toàn bộ config
├── scripts/
│   ├── check_hardware.py       # Kiểm tra GPU + NPU driver + device
│   ├── 01_download_models.py   # Download HuggingFace models
│   ├── 02_merge_lora.py        # Merge LoRA vào base model
│   ├── 03_convert_openvino.py  # Export OpenVINO IR + INT4 quantization
│   └── 04_run_inference.py     # CLI inference tool (GPU/NPU/CPU)
├── models/                     # (gitignored — không đẩy lên git)
│   ├── base/                   #   models/base/llama3.1-8b/ + fingpt-lora/
│   ├── merged/                 #   models/merged/ (FP16, ~16 GB)
│   └── openvino/               #   models/openvino/ (INT4, ~4.5 GB)
├── tests/
│   ├── test_fingpt.py          # English test suite (14 cases)
│   └── test_report_*.json      # Kết quả test
├── server.py                   # FastAPI OpenAI-compatible server
├── app.py                      # Gradio web UI
├── setup.ps1                   # Windows setup script (native)
├── run.ps1                     # Docker runner script (tự động chọn GPU/CPU)
├── Dockerfile                  # Multi-stage Docker build (builder + runtime)
├── docker-compose.yml          # Docker Compose orchestrator
├── requirements.txt            # Python dependencies
├── CLAUDE.md                   # Claude Code project instructions
├── FinAI_Report_VN.md          # Báo cáo triển khai (tiếng Việt)
├── .env                        # HF_TOKEN (gitignored)
└── .env.example                # Template cho .env
```

## 4.2. Cấu hình — `configs/model_config.json`

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

**Bảng 3 — Trường cấu hình trong `model_config.json`:**

| Trường | Ý nghĩa | Tác dụng |
|---|---|---|
| `base_model` | HuggingFace repo ID của Llama 3.1 8B Instruct | Model gốc chưa fine-tune — nền tảng để apply LoRA |
| `lora_model` | Repo ID của FinGPT LoRA adapter | Trọng số fine-tune chuyên biệt tài chính — được cộng thêm vào base model |
| `quantization.weight_format: int4` | Mỗi weight nén từ 16 bit xuống 4 bit | Giảm 4× kích thước model (16 GB → 4 GB), NPU xử lý INT4 trực tiếp |
| `quantization.symmetric: true` | Symmetric quantization (không có offset/channel riêng) | **Bắt buộc cho NPU Intel** — NPU không hỗ trợ asymmetric |
| `quantization.ratio: 1.0` | 100% weights nén INT4, không trộn FP16/INT4 | Đảm bảo tất cả layers đều chạy trên NPU với INT4 |
| `quantization.group_size: -1` | Channel-wise quantization (mỗi channel 1 scale factor) | Chất lượng cao hơn group quantization — dùng cho 7B+ models |
| `inference.device` | Device inference mặc định (ưu tiên GPU → NPU → CPU) | Thay đổi: `--device GPU`, `--device NPU`, hoặc `--device CPU` |
| `inference.max_new_tokens: 512` | Số token tối đa model được sinh ra | Giới hạn độ dài response để tránh generation quá lâu |
| `paths.*` | Đường dẫn tương đối từ project root | Gitignored — model files không đẩy lên git |

## 4.3. Download models — `01_download_models.py`

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

## 4.4. Merge LoRA — `02_merge_lora.py`

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

## 4.5. Convert OpenVINO — `03_convert_openvino.py`

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

## 4.6. Inference — `04_run_inference.py`

**Chức năng:** Tool CLI để chạy inference tương tác trên GPU/NPU/CPU.

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
      Tải OpenVINO IR model vào device

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

## 4.7. API Server — `server.py`

**Chức năng:** Server REST API tương thích OpenAI, cho phép bất kỳ OpenAI client nào (Gradio, OpenClaw, LangChain, curl...) gọi FinGPT qua HTTP.

**Dependencies:** `fastapi`, `uvicorn`, `openvino-genai`.

**Bảng 4 — API Endpoints của server:**

| Endpoint | HTTP Method | Mục đích | Ai dùng |
|---|---|---|---|
| `/v1/chat/completions` | POST | Chat completions chuẩn OpenAI | Gradio app.py, OpenAI SDK, OpenClaw |
| `/v1/responses` | POST | OpenAI Responses API | OpenClaw agent mode |
| `/v1/completions` | POST | Legacy completions endpoint | Tương thích ngược |
| `/v1/models` | GET | Liệt kê model available | OpenAI client discovery |
| `/health` | GET | Health check + device info | Giám sát, kiểm tra |

**Luồng xử lý `/v1/chat/completions`:**

```
HTTP POST /v1/chat/completions
         │
         ▼
1. ChatCompletionRequest (Pydantic)
   model: str
   messages: list[Message]
   max_tokens: int
   temperature: float
   stream: bool
   (extra="allow" → chấp nhận extra fields từ OpenClaw)
         │
         ▼
2. build_prompt(messages)
   - get_text(msg.content) → extract str từ str | list | None
   - Ghép Llama 3.1 template
   - Truncate system prompt >2000 chars (MAX_PROMPT_CHARS)
         │
         ▼
3. ov_genai.GenerationConfig()
   max_new_tokens = request.max_tokens or config default
   temperature = request.temperature
   top_p = request.top_p
         │
         ▼
4. pipe.generate(prompt, gen_config)
   GPU/NPU/CPU: OpenVINO runtime inference
         │
   stream=True?
   /         \
Yes          No
 │            │
 ▼            ▼
SSE chunks  Full JSON
         │
         ▼
Client hiển thị kết quả
```

**Biến `pipe` — Global Model Object:**

```python
pipe = None  # global

def main():
    global pipe
    device = args.device or config["inference"]["device"]
    model_dir = str(Path(__file__).parent / config["paths"]["openvino_model_dir"])

    if device == "NPU":
        # MAX_PROMPT_LEN=8192: tăng prompt limit từ 1024 mặc định
        npu_config = {"MAX_PROMPT_LEN": 8192}
        pipe = ov_genai.LLMPipeline(model_dir, device, **npu_config)
    else:
        pipe = ov_genai.LLMPipeline(model_dir, device)

    uvicorn.run(app, host=args.host, port=args.port)
```

**Streaming SSE (Server-Sent Events):**

OpenClaw TUI yêu cầu streaming. Server không stream token-by-token (vì inference không hỗ trợ real-time streaming), mà sinh toàn bộ response rồi gửi một lần qua SSE:

```
1. pipe.generate(prompt, config)  → sinh full response text (toàn bộ trong RAM)
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

**Hàm `get_text()` — Xử lý content format:**

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

## 4.8. Giao diện Web — `app.py`

**Chức năng:** Giao diện web chat cho FinGPT, cho phép người dùng không biết code tương tác với model.

**Dependencies:** `gradio`, `requests`.

**Hai chế độ hoạt động:**

```
CÁCH 1: app.py tự load model (KHÔNG khuyến nghị)
────────────────────────────────────────────────────
python app.py (không --api-url)

app.py process:
  ├── ov_genai.LLMPipeline()  ← load ~4.5GB vào app.py
  └── gradio.ChatInterface()

  RAM: ~5 GB (model + UI)
  ⚠️ Load model LẦN 2 trong process Gradio
  ⚠️ Tốn RAM không cần thiết


CÁCH 2: app.py gọi server.py qua HTTP (KHUYẾN NGHỊ)
────────────────────────────────────────────────────────
python app.py --api-url http://127.0.0.1:8000

┌──────────────────┐    HTTP POST    ┌──────────────────────┐
│ app.py           │ ──────────────► │ server.py            │
│ (Gradio UI)      │                 │ (đã load model rồi)  │
│ RAM: ~500 MB     │ ◄───────────── │ RAM: ~4.5 GB         │
└──────────────────┘    JSON resp    └──────────────────────┘
→ Dùng chung model qua HTTP
→ ✅ Tiết kiệm RAM, ổn định hơn
```

**Luồng xử lý `respond()`:**

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
        timeout=120,
    )

    # 3. Return content từ response
    return resp.json()["choices"][0]["message"]["content"]
```

## 4.9. Setup Script — `setup.ps1`

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

# 5. LƯU ĐỒ GIẢI THUẬT

## 5.1. Pipeline 4 bước

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
│  • Cần: HF_TOKEN (gated model)                          │
└─────────────────────────┬───────────────────────────────┘
                          ▼
                   Download thành công?
                    /                  \
                 Có                   Không → THOÁT
                  │                   (báo lỗi HF_TOKEN)
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Bước 2: 02_merge_lora.py                               │
│  • AutoModelForCausalLM.from_pretrained(base)           │
│  • PeftModel.from_pretrained(base, lora)                │
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
│  • optimum.exporters.openvino CLI                       │
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
│  • 04_run_inference.py  → CLI tương tác                │
│  • server.py            → API server                    │
│  • app.py               → Gradio web UI                │
│  • ov_genai.LLMPipeline(model_dir, device)             │
│  • pipe.generate(prompt, gen_config)                    │
└─────────────────────────┬───────────────────────────────┘
                          ▼
                   ┌───────┴───────┐
                   │   KẾT THÚC    │
                   └───────────────┘
```

## 5.2. Inference từ HTTP Request

```
Client (Gradio / OpenClaw / curl)
  POST /v1/chat/completions
  {messages: [...], stream: true/false, ...}
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ server.py — Endpoint: @app.post("/v1/chat/completions")             │
│  1. Nhận ChatCompletionRequest (Pydantic model)                    │
│  2. Kiểm tra pipe is not None?                                      │
│     └─ Nếu None → HTTP 503 "Model not loaded"                       │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ server.py — build_prompt(messages)                                   │
│  for msg in messages:                                                │
│    content = get_text(msg.content)  ← str | list | None → str       │
│    if role == "system": → truncate >2000 chars (MAX_PROMPT_CHARS)   │
│    if role == "user":    → ghép Llama 3.1 template                  │
│    if role == "assistant": → ghép Llama 3.1 template                 │
│  final_prompt = "<|begin_of_text|>" + system + user + assistant     │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ov_genai.GenerationConfig()                                          │
│  gen_config.max_new_tokens = request.max_tokens or 512               │
│  gen_config.temperature  = request.temperature or 0.7                │
│  gen_config.top_p        = request.top_p or 0.9                      │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ pipe.generate(final_prompt, gen_config)                              │
│  GPU/NPU/CPU inference → sinh tokens → decode → string               │
└──────────┬───────────────────────────────────────────────────────────┘
           │
    stream?
    /       \
  Có        Không
   │          │
   ▼          ▼
SSE chunks  Full JSON response
   │          │
   └────┬─────┘
        ▼
Client hiển thị kết quả
```

## 5.3. Xây dựng Llama 3.1 Prompt

```
Input: messages = list of Message objects (từ OpenAI API request)

         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│ system_msg = ""; user_blocks = []; asst_blocks = []               │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────┐
│ LOOP: for msg in messages                                          │
│                                                                     │
│   content = get_text(msg.content)                                  │
│   # content có thể là: str, list[dict], None                     │
│                                                                     │
│   IF msg.role == "system":                                         │
│     IF len(content) > 2000:                                        │
│       system_msg = SYSTEM_PROMPT + content[:2000]                   │
│     ELSE:                                                          │
│       system_msg = content                                         │
│                                                                     │
│   ELIF msg.role == "user":                                        │
│     user_block = "<|start_header_id|>user<|end_header_id|>\n\n"   │
│                  + "{content}<|eot_id|>"                           │
│     APPEND user_block TO user_blocks                               │
│                                                                     │
│   ELIF msg.role == "assistant":                                    │
│     asst_block = "<|start_header_id|>assistant<|end_header_id|>\n\n"
│                  + "{content}<|eot_id|>"                           │
│     APPEND asst_block TO asst_blocks                              │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────┐
│ ASSEMBLE final_prompt:                                             │
│ final_prompt = (                                                    │
│   "<|begin_of_text|>"                                              │
│   "<|start_header_id|>system<|end_header_id|>\n\n"                 │
│   "{system_msg}<|eot_id|>"                                         │
│   + "".join(user_blocks)                                           │
│   + "".join(asst_blocks)                                           │
│   "<|start_header_id|>assistant<|end_header_id|>\n\n"             │
│ )                                                                   │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────┐
│ pipe.generate(final_prompt, gen_config)                             │
│   Inference → sinh tokens → decode → string text                    │
└────────────────────────────────────────────────────────────────────┘
```

## 5.4. OpenClaw Integration

```
Người dùng gõ lệnh trong OpenClaw TUI
         │
         ▼
OpenClaw TUI đọc config: ~/.openclaw/openclaw.json
{ "provider": "openai", "api": "openai-completions",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "fingpt-llama3.1-8b-npu" }
         │
         ▼
OpenClaw gửi HTTP POST /v1/responses
  Accept: text/event-stream  ← báo server gửi SSE
  Body: { "model": "...", "input": "...", "stream": true }
         │
         ▼
server.py xử lý:
  1. /v1/responses endpoint (OpenAI Responses API)
  2. Extract "input" from request
  3. build_prompt() → format Llama 3.1 template
  4. pipe.generate() → GPU/NPU/CPU inference
  5. Trả về SSE stream
         │
         ▼
OpenClaw TUI nhận SSE chunks
  Đọc chunk data: {delta: {content: "..."}}
  Cập nhật terminal: hiển thị text đang sinh ra
  Đọc chunk finish_reason: "stop" → kết thúc
         │
         ▼
OpenClaw TUI hiển thị câu trả lời hoàn chỉnh
```

## 5.5. Gradio + Server Architecture

```
CÁCH 1: app.py tự load model (KHÔNG khuyến nghị)
────────────────────────────────────────────────────
Terminal 2: python app.py

app.py process:
  ├── Gradio UI (listen port XXXXX)
  ├── requests.post() → KHÔNG gọi server.py vì KHÔNG có --api-url
  └── ov_genai.LLMPipeline() → load model vào app.py process
      → RAM: ~4.5 GB (model) + ~500 MB (Gradio) = ~5 GB

⚠️ Model được load LẦN 2 trong process Gradio
⚠️ Tốn RAM không cần thiết

────────────────────────────────────────────────────────

CÁCH 2: app.py gọi server.py qua HTTP (KHUYẾN NGHỊ)
────────────────────────────────────────────────────────

Terminal 1: python server.py
┌─────────────────────────────────┐
│ server.py:                      │
│   ov_genai.LLMPipeline()       │  ← Load model 1 lần duy nhất
│   uvicorn (HTTP server)        │  ← Listen port 8000
└────────────┬────────────────────┘
             │ JSON response
             ◄──────────────────
             │ HTTP POST /v1/chat/completions
             ▼
Terminal 2: python app.py --api-url http://127.0.0.1:8000
┌─────────────────────────────────┐
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

# 6. TÀI LIỆU THAM KHẢO HÌNH ẢNH

**Bảng 6 — Danh mục hình ảnh minh hoạ:**

| STT | File ảnh | Mô tả |
|-----|----------|--------|
| 0 | `images/01-docker-running.png` | Docker Desktop đang chạy |
| 1 | `images/02-env-file.png` | File `.env` đã lưu với `HF_TOKEN=...` |
| 2 | `images/03-hf-license.png` | Đã accept HuggingFace Llama 3.1 license |
| 3 | `images/04-pipeline-complete.png` | Output "Pipeline complete!" |
| 4 | `images/05-server-running.png` | Server đang chạy tại `http://localhost:8000` |
| 5 | `images/06-openclaw-running.png` | *(Tuỳ chọn)* OpenClaw gọi FinGPT thành công |

> **Lưu ảnh vào thư mục `images/`** trong project FinAI, đặt đúng tên file như bảng trên.

---

# PHỤ LỤC A: CẤU HÌNH OPENCLAW

File cấu hình OpenClaw nằm ở `~/.openclaw/openclaw.json`:

```json
{
  "provider": "openai",
  "api": "openai-completions",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "fingpt-llama3.1-8b-npu"
}
```

> **Hình 6: OpenClaw đang gọi FinGPT thành công** *(tuỳ chọn)*
> ![Hình 6](images/06-openclaw-running.png)

---

# PHỤ LỤC B: XỬ LÝ SỰ CỐ

**Bảng 5 — Troubleshooting thường gặp:**

| Vấn đề | Nguyên nhân | Giải pháp |
|---|---|---|
| `HF_TOKEN` not found | Chưa tạo `.env` hoặc chưa activate venv | Tạo `.env`, activate venv, restart terminal |
| GPU/NPU not detected | Driver chưa cài / chưa enable | Chạy `scripts/check_hardware.py`, cài Intel GPU/NPU driver |
| Out of memory khi merge | RAM < 32 GB, không đủ cho FP16 merge | Dùng `--device_map="cpu"` (đã có sẵn), đóng ứng dụng khác |
| Convert fail — asymmetric error | Config sai symmetric/ratio | Kiểm tra `model_config.json`: `symmetric: true`, `ratio: 1.0` |
| Server 503 — Model not loaded | Server chưa khởi động hoặc device lỗi | Chạy `python server.py --device CPU` thay vì GPU/NPU |
| OpenClaw không hiển thị text | SSE streaming lỗi | Kiểm tra `/health`, restart server, thử non-streaming |
| Gradio — Error: Cannot connect | app.py không tìm thấy server | Chạy `server.py` trước, dùng `--api-url` đúng |
| Python 3.14 — ModuleNotFoundError | Thiếu prebuilt wheels | Cài Python 3.11, xóa .venv, chạy lại `setup.ps1` |

---

*Bản quyền © 2026. Nội dung có thể được sử dụng cho mục đích học tập và nghiên cứu.*
