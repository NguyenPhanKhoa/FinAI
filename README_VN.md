# FinAI — FinGPT trên Intel NPU

Triển khai [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) (Llama 3.1 8B + LoRA) trên Intel Core Ultra 7 258V NPU sử dụng OpenVINO với lượng tử hóa INT4 đối xứng. Bao gồm API server tương thích OpenAI, giao diện web Gradio và tích hợp OpenClaw AI Agent.

## Tính Năng

- **Tăng tốc NPU** — Chạy trên Intel NPU (47 TOPS) với lượng tử hóa INT4 qua OpenVINO
- **API Tương thích OpenAI** — Server thay thế trực tiếp cho mọi client OpenAI
- **Hỗ trợ Streaming** — SSE streaming để truyền token theo thời gian thực
- **AI Tài chính** — Phân tích tâm lý thị trường, dự báo, đánh giá rủi ro, xử lý văn bản tài chính
- **Nhiều giao diện** — CLI, Web UI (Gradio), API server, OpenClaw TUI
- **Kiểm thử tự động** — Bộ test 14 trường hợp bao phủ tất cả tính năng và endpoint

## Hạn Chế

- **Không có dữ liệu thời gian thực** — Mô hình không có kết nối internet, không thể lấy tin tức, giá cổ phiếu hay dữ liệu thị trường trực tiếp. Mô hình chỉ phân tích văn bản mà bạn cung cấp.
- **Dữ liệu huấn luyện đến 2023** — Mô hình cơ sở (Llama 3.1) và FinGPT LoRA được huấn luyện trên dữ liệu đến năm 2023. Mô hình không biết các sự kiện sau thời điểm đó.
- **Không phải tư vấn tài chính** — Kết quả chỉ mang tính tham khảo/giáo dục, không nên dùng làm căn cứ duy nhất cho quyết định đầu tư.

## Có Thể Hỏi Gì?

Bạn cung cấp văn bản, mô hình phân tích cho bạn.

**Hoạt động tốt:**
- "Phân tích tâm lý của tin này: [dán tin tức]"
- "Tóm tắt báo cáo thu nhập này: [dán văn bản]"
- "Trích xuất các chỉ số tài chính từ: [dán dữ liệu]"
- "Rủi ro khi đầu tư tập trung vào một ngành là gì?"
- "Giải thích P/E ratio và cách sử dụng"
- "Yếu tố nào ảnh hưởng đến giá trái phiếu khi lãi suất tăng?"
- "So sánh đầu tư tăng trưởng và đầu tư giá trị"

**Không hoạt động:**
- "Giá cổ phiếu AAPL hôm nay là bao nhiêu?" (không có internet)
- "Chuyện gì xảy ra trong vụ sập thị trường 2025?" (dữ liệu huấn luyện đến 2023)
- "Tôi có nên mua TSLA ngay bây giờ không?" (không có dữ liệu trực tiếp)

## Yêu Cầu Phần Cứng

| Thành phần | Tối thiểu | Khuyến nghị |
|------------|-----------|-------------|
| **CPU** | Intel Core Ultra (Lunar Lake) có NPU | Intel Core Ultra 7 258V |
| **RAM** | 16 GB | 32 GB |
| **Bộ nhớ** | 30 GB trống | 50 GB trống |
| **Driver NPU** | [Intel NPU Driver](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html) | Phiên bản mới nhất |
| **Hệ điều hành** | Windows 11 | Windows 11 |

## Bắt Đầu Nhanh

```powershell
# 1. Cài đặt môi trường (yêu cầu Python 3.11)
powershell -ExecutionPolicy Bypass -File setup.ps1

# 2. Cấu hình token HuggingFace
copy .env.example .env
# Sửa file .env và thêm token HuggingFace (cần quyền truy cập Llama 3.1)

# 3. Kích hoạt môi trường
.\.venv\Scripts\Activate.ps1

# 4. Kiểm tra phần cứng
python scripts/check_hardware.py

# 5. Chạy pipeline (theo thứ tự)
python scripts/01_download_models.py
python scripts/02_merge_lora.py
python scripts/03_convert_openvino.py

# 6. Khởi động server
python server.py

# 7. Mở giao diện web (trong terminal khác)
python app.py
```

## Pipeline Mô Hình

```
Meta Llama 3.1 8B Instruct (cơ sở)  +  FinGPT LoRA adapter
                    ↓ gộp (PEFT)
              Mô hình FP16 đã gộp (~16 GB)
                    ↓ chuyển đổi (optimum-cli)
        OpenVINO IR + lượng tử hóa INT4 đối xứng (~4.5 GB)
                    ↓ triển khai
              Intel NPU qua openvino-genai
```

## Cách Sử Dụng

### API Server

API server tương thích OpenAI, mọi client OpenAI đều có thể kết nối:

```bash
python server.py                    # mặc định: NPU, cổng 8000
python server.py --device CPU       # chuyển sang CPU
python server.py --port 9000        # cổng tùy chỉnh
```

**Các Endpoint:**

| Endpoint | Phương thức | Mô tả |
|----------|-------------|-------|
| `/v1/chat/completions` | POST | Chat completions (streaming + non-streaming) |
| `/v1/responses` | POST | OpenAI Responses API |
| `/v1/completions` | POST | Legacy completions |
| `/v1/models` | GET | Danh sách mô hình khả dụng |
| `/health` | GET | Kiểm tra trạng thái server |

**Ví dụ request:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fingpt-llama3.1-8b-npu",
    "messages": [{"role": "user", "content": "Phân tích tâm lý thị trường: AAPL vượt kỳ vọng lợi nhuận 5%"}],
    "stream": false
  }'
```

### Giao Diện Web (Gradio)

```bash
python app.py                       # mở tại http://localhost:7860
python app.py --share               # tạo link công khai
```

Yêu cầu `server.py` đang chạy.

### CLI Tương Tác

```bash
python scripts/04_run_inference.py              # NPU
python scripts/04_run_inference.py --device CPU  # chuyển sang CPU
```

### OpenClaw AI Agent

Kết nối FinGPT như một AI agent trong [OpenClaw](https://openclaw.com):

1. Khởi động server: `python server.py`
2. Thêm provider trong `~/.openclaw/openclaw.json`:
```json
{
  "models": {
    "providers": {
      "fingpt-npu": {
        "baseUrl": "http://127.0.0.1:8000/v1",
        "apiKey": "no-key-needed",
        "api": "openai-completions",
        "models": [{
          "id": "fingpt-llama3.1-8b-npu",
          "name": "FinGPT Llama 3.1 8B (NPU)",
          "contextWindow": 32768
        }]
      }
    }
  }
}
```
3. Đặt làm mô hình mặc định trong `agents.defaults.model.primary`: `"fingpt-npu/fingpt-llama3.1-8b-npu"`

## Kiểm Thử

Chạy bộ kiểm thử tự động (yêu cầu `server.py` đang chạy):

```bash
python tests/test_fingpt.py
```

**Phạm vi kiểm thử (14 test):**

| Danh mục | Số test | Mô tả |
|----------|---------|-------|
| Phân tích tâm lý | 3 | Tin tích cực, tiêu cực và trung lập |
| Kiến thức tài chính | 3 | P/E ratio, thị trường tăng/giảm, đa dạng hóa |
| Phân tích thị trường | 2 | Phân kỳ doanh thu vs giá, chỉ báo suy thoái |
| Xử lý văn bản tài chính | 2 | Tóm tắt báo cáo, trích xuất chỉ số |
| Đánh giá rủi ro | 1 | Nhận thức rủi ro tập trung |
| Ranh giới chuyên môn | 1 | Xử lý yêu cầu ngoài tài chính |
| Streaming (SSE) | 1 | Kiểm tra endpoint streaming |
| Responses API | 1 | Kiểm tra endpoint /v1/responses |

Kết quả được lưu tại `test_results/` dưới dạng báo cáo `.txt` và dữ liệu `.json`.

## Cấu Trúc Dự Án

```
FinAI/
├── server.py                      # API server tương thích OpenAI (FastAPI)
├── app.py                         # Giao diện web Gradio (kết nối server.py)
├── configs/
│   └── model_config.json          # Cấu hình mô hình, lượng tử hóa và suy luận
├── scripts/
│   ├── check_hardware.py          # Kiểm tra tương thích phần cứng
│   ├── 01_download_models.py      # Tải base + LoRA từ HuggingFace
│   ├── 02_merge_lora.py           # Gộp LoRA vào mô hình cơ sở (PEFT)
│   ├── 03_convert_openvino.py     # Chuyển đổi sang OpenVINO IR INT4
│   └── 04_run_inference.py        # Suy luận CLI trên NPU
├── tests/
│   └── test_fingpt.py             # Bộ kiểm thử tự động (14 trường hợp)
├── test_results/                  # Báo cáo kiểm thử (.txt + .json)
├── models/                        # (gitignored) file mô hình
│   ├── base/                      # Base + LoRA đã tải
│   ├── merged/                    # Mô hình FP16 đã gộp
│   └── openvino/                  # OpenVINO IR INT4 hoàn chỉnh
├── requirements.txt
├── setup.ps1                      # Script cài đặt Windows
├── .env.example                   # Mẫu biến môi trường
└── CLAUDE.md                      # Hướng dẫn Claude Code
```

## Cấu Hình

Sửa `configs/model_config.json`:

| Cài đặt | Mặc định | Mô tả |
|---------|----------|-------|
| `base_model` | `meta-llama/Llama-3.1-8B-Instruct` | Mô hình cơ sở HuggingFace |
| `lora_model` | `FinGPT/fingpt-mt_llama3-8b_lora` | FinGPT LoRA adapter |
| `weight_format` | `int4` | Định dạng lượng tử hóa |
| `symmetric` | `true` | Lượng tử hóa đối xứng (bắt buộc cho NPU) |
| `group_size` | `-1` | Theo kênh cho mô hình 7B+ |
| `device` | `NPU` | Thiết bị suy luận (NPU/CPU/GPU) |
| `max_new_tokens` | `512` | Độ dài sinh tối đa |
| `temperature` | `0.7` | Nhiệt độ lấy mẫu |

## Chuyển Đổi Thiết Bị

Nếu NPU gặp vấn đề, chuyển sang CPU hoặc GPU:

```bash
python server.py --device CPU
python scripts/04_run_inference.py --device GPU
```

## Công Nghệ Sử Dụng

- **Mô hình**: Meta Llama 3.1 8B Instruct + FinGPT LoRA (tinh chỉnh tài chính)
- **Runtime**: OpenVINO GenAI với lượng tử hóa INT4 đối xứng
- **Phần cứng**: Intel Core Ultra 7 258V NPU (47 TOPS, Lunar Lake)
- **Server**: FastAPI + Uvicorn (API tương thích OpenAI)
- **Giao diện Web**: Gradio
- **Agent**: Nền tảng OpenClaw AI Agent
- **Ngôn ngữ**: Python 3.11

## Giấy Phép

Dự án này sử dụng Meta Llama 3.1 (theo [giấy phép Meta](https://llama.meta.com/llama3/license/)) và trọng số FinGPT LoRA từ [AI4Finance](https://github.com/AI4Finance-Foundation/FinGPT).
