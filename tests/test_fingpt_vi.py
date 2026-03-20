"""
FinGPT Bot Test Suite — Tiếng Việt
Gửi các câu hỏi tài chính bằng tiếng Việt đến server FinGPT API
và đánh giá xem phản hồi có liên quan, mạch lạc và chính xác không.

Usage:
  python tests/test_fingpt_vi.py              # mặc định: http://127.0.0.1:8000
  python tests/test_fingpt_vi.py --base-url http://127.0.0.1:9000
"""

import argparse
import json
import time
import re
import requests
from pathlib import Path
from datetime import datetime


BASE_URL = "http://127.0.0.1:8000"

# --- Các trường hợp kiểm thử ---

TEST_CASES = [
    # --- Phân tích tâm lý thị trường ---
    {
        "id": "SA-01-VI",
        "category": "Phân tích tâm lý thị trường",
        "question": "Phân tích tâm lý thị trường của tin sau: 'Apple báo cáo doanh thu quý kỷ lục 124 tỷ USD, vượt kỳ vọng của các nhà phân tích 5%.'",
        "expected_keywords": ["positive", "tích cực", "bullish", "tăng", "strong", "beat", "revenue", "doanh thu", "growth"],
        "reject_keywords": ["negative", "tiêu cực", "bearish", "giảm", "loss", "thua lỗ"],
        "description": "Tin tức thu nhập tích cực rõ ràng",
    },
    {
        "id": "SA-02-VI",
        "category": "Phân tích tâm lý thị trường",
        "question": "Đánh giá tâm lý thị trường: 'Cổ phiếu Tesla giảm mạnh 12% sau khi công ty không đạt mục tiêu giao hàng và thông báo sa thải 10% nhân sự.'",
        "expected_keywords": ["negative", "tiêu cực", "bearish", "giảm", "drop", "decline", "miss", "sa thải", "layoff"],
        "reject_keywords": ["positive", "tích cực", "bullish", "tăng trưởng", "beat"],
        "description": "Tin tức cổ phiếu tiêu cực rõ ràng",
    },
    {
        "id": "SA-03-VI",
        "category": "Phân tích tâm lý thị trường",
        "question": "Tâm lý thị trường của tin này là gì: 'Cục Dự trữ Liên bang Mỹ (Fed) giữ nguyên lãi suất, báo hiệu cách tiếp cận chờ đợi và quan sát giữa các dữ liệu kinh tế trái chiều.'",
        "expected_keywords": ["neutral", "trung lập", "mixed", "uncertain", "steady", "hold", "wait", "chờ", "giữ nguyên"],
        "reject_keywords": [],
        "description": "Tin tức chính sách tiền tệ trung lập/hỗn hợp",
    },

    # --- Kiến thức tài chính ---
    {
        "id": "FK-01-VI",
        "category": "Kiến thức tài chính",
        "question": "Tỷ lệ Giá trên Thu nhập (P/E) là gì và tại sao nó quan trọng trong việc định giá cổ phiếu?",
        "expected_keywords": ["price", "giá", "earnings", "thu nhập", "ratio", "tỷ lệ", "valuation", "định giá", "stock", "cổ phiếu"],
        "reject_keywords": [],
        "description": "Định nghĩa chỉ số tài chính cơ bản",
    },
    {
        "id": "FK-02-VI",
        "category": "Kiến thức tài chính",
        "question": "Giải thích sự khác biệt giữa thị trường tăng giá (bull market) và thị trường giảm giá (bear market).",
        "expected_keywords": ["bull", "bear", "tăng", "giảm", "market", "thị trường", "price", "giá"],
        "reject_keywords": [],
        "description": "Khái niệm thị trường cơ bản",
    },
    {
        "id": "FK-03-VI",
        "category": "Kiến thức tài chính",
        "question": "Đa dạng hóa danh mục đầu tư là gì và tại sao nó quan trọng?",
        "expected_keywords": ["risk", "rủi ro", "asset", "tài sản", "portfolio", "danh mục", "spread", "phân bổ", "invest", "đầu tư"],
        "reject_keywords": [],
        "description": "Nguyên tắc đầu tư cốt lõi",
    },

    # --- Phân tích thị trường ---
    {
        "id": "MA-01-VI",
        "category": "Phân tích thị trường",
        "question": "Nếu doanh thu của một công ty tăng 20% so với cùng kỳ năm trước nhưng giá cổ phiếu lại giảm 30%, điều gì có thể giải thích sự khác biệt này?",
        "expected_keywords": ["valuation", "định giá", "market", "thị trường", "expect", "kỳ vọng", "growth", "tăng trưởng", "price", "giá", "investor", "nhà đầu tư"],
        "reject_keywords": [],
        "description": "Phân tích doanh thu so với giá cổ phiếu",
    },
    {
        "id": "MA-02-VI",
        "category": "Phân tích thị trường",
        "question": "Các chỉ số tài chính quan trọng cần theo dõi trong thời kỳ suy thoái kinh tế là gì?",
        "expected_keywords": ["gdp", "unemployment", "thất nghiệp", "inflation", "lạm phát", "interest", "lãi suất", "consumer", "tiêu dùng", "rate"],
        "reject_keywords": [],
        "description": "Kiến thức về các chỉ báo suy thoái",
    },

    # --- Xử lý văn bản tài chính ---
    {
        "id": "FT-01-VI",
        "category": "Xử lý văn bản tài chính",
        "question": "Tóm tắt đoạn báo cáo thu nhập sau: 'Doanh thu Q3 đạt 50,7 tỷ USD, tăng 8% so với cùng kỳ. Thu nhập hoạt động tăng lên 13,2 tỷ USD từ 11,5 tỷ USD. Dòng tiền tự do đạt 12,1 tỷ USD. Công ty đã mua lại 5 tỷ USD cổ phiếu và tăng cổ tức hàng quý thêm 10%.'",
        "expected_keywords": ["revenue", "doanh thu", "growth", "tăng", "income", "thu nhập", "cash", "tiền", "dividend", "cổ tức", "share", "cổ phiếu"],
        "reject_keywords": [],
        "description": "Tóm tắt báo cáo thu nhập",
    },
    {
        "id": "FT-02-VI",
        "category": "Xử lý văn bản tài chính",
        "question": "Trích xuất các chỉ số tài chính chính từ: 'Công ty XYZ báo cáo EPS 3,42 USD so với kỳ vọng 3,15 USD, doanh thu 28,5 tỷ USD so với kỳ vọng 27,8 tỷ USD, và biên lợi nhuận gộp cải thiện lên 45,2% từ 43,8% năm trước.'",
        "expected_keywords": ["eps", "revenue", "doanh thu", "margin", "biên", "beat", "vượt", "expect", "kỳ vọng"],
        "reject_keywords": [],
        "description": "Trích xuất dữ liệu tài chính",
    },

    # --- Đánh giá rủi ro ---
    {
        "id": "RA-01-VI",
        "category": "Đánh giá rủi ro",
        "question": "Những rủi ro chính khi đầu tư tập trung vào một cổ phiếu công nghệ duy nhất là gì?",
        "expected_keywords": ["risk", "rủi ro", "concentration", "tập trung", "volatil", "biến động", "diversif", "đa dạng", "sector", "ngành", "loss", "tổn thất"],
        "reject_keywords": [],
        "description": "Nhận thức về rủi ro tập trung",
    },

    # --- Ranh giới lĩnh vực ---
    {
        "id": "DB-01-VI",
        "category": "Ranh giới lĩnh vực",
        "question": "Viết cho tôi một bài thơ về hoa.",
        "expected_keywords": [],
        "reject_keywords": [],
        "description": "Yêu cầu ngoài lĩnh vực tài chính — vẫn nên phản hồi nhưng có thể chuyển hướng về tài chính",
        "check_type": "domain_awareness",
    },

    # --- Kiểm thử Streaming ---
    {
        "id": "ST-01-VI",
        "category": "Streaming",
        "question": "Chia tách cổ phiếu là gì và nó ảnh hưởng đến cổ đông như thế nào?",
        "expected_keywords": ["stock", "cổ phiếu", "split", "chia tách", "share", "price", "giá", "value", "giá trị"],
        "reject_keywords": [],
        "description": "Kiểm thử endpoint streaming SSE",
        "use_streaming": True,
    },

    # --- Responses API ---
    {
        "id": "RP-01-VI",
        "category": "Responses API",
        "question": "Vốn hóa thị trường là gì?",
        "expected_keywords": ["market", "thị trường", "cap", "vốn hóa", "share", "cổ phiếu", "price", "giá", "value", "giá trị", "company", "công ty"],
        "reject_keywords": [],
        "description": "Kiểm thử endpoint /v1/responses",
        "use_responses_api": True,
    },
]


# --- Bộ chạy kiểm thử ---

def send_chat(question: str, base_url: str, stream: bool = False) -> dict:
    """Gửi yêu cầu chat completion và trả về thông tin phản hồi."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "fingpt-llama3.1-8b-npu",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": stream,
    }

    start = time.time()

    if stream:
        resp = requests.post(url, json=payload, stream=True, timeout=120)
        resp.raise_for_status()
        content_parts = []
        for line in resp.iter_lines(decode_unicode=True):
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    content_parts.append(delta["content"])
        elapsed = time.time() - start
        return {"text": "".join(content_parts), "elapsed": elapsed, "status": 200}

    resp = requests.post(url, json=payload, timeout=120)
    elapsed = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return {"text": text, "elapsed": elapsed, "status": resp.status_code}


def send_responses_api(question: str, base_url: str) -> dict:
    """Gửi yêu cầu đến endpoint /v1/responses."""
    url = f"{base_url}/v1/responses"
    payload = {
        "model": "fingpt-llama3.1-8b-npu",
        "input": question,
    }
    start = time.time()
    resp = requests.post(url, json=payload, timeout=120)
    elapsed = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    text = data.get("output_text", "")
    return {"text": text, "elapsed": elapsed, "status": resp.status_code}


def evaluate_response(test: dict, response_text: str) -> dict:
    """Đánh giá chất lượng phản hồi dựa trên từ khóa kỳ vọng/loại bỏ."""
    text_lower = response_text.lower()
    result = {
        "has_content": len(response_text.strip()) > 10,
        "matched_keywords": [],
        "missing_keywords": [],
        "rejected_keywords_found": [],
    }

    for kw in test.get("expected_keywords", []):
        if kw.lower() in text_lower:
            result["matched_keywords"].append(kw)
        else:
            result["missing_keywords"].append(kw)

    for kw in test.get("reject_keywords", []):
        if kw.lower() in text_lower:
            result["rejected_keywords_found"].append(kw)

    expected = test.get("expected_keywords", [])
    if expected:
        match_ratio = len(result["matched_keywords"]) / len(expected)
    else:
        match_ratio = 1.0 if result["has_content"] else 0.0

    no_rejected = len(result["rejected_keywords_found"]) == 0

    # Chấm điểm
    if test.get("check_type") == "domain_awareness":
        result["pass"] = result["has_content"]
        result["reason"] = "Có phản hồi" if result["pass"] else "Phản hồi trống"
    elif match_ratio >= 0.4 and no_rejected and result["has_content"]:
        result["pass"] = True
        result["reason"] = f"Khớp {len(result['matched_keywords'])}/{len(expected)} từ khóa, không có từ bị loại"
    elif match_ratio >= 0.2 and no_rejected and result["has_content"]:
        result["pass"] = True
        result["reason"] = f"Khớp một phần {len(result['matched_keywords'])}/{len(expected)} từ khóa (chấp nhận được)"
    else:
        result["pass"] = False
        reasons = []
        if not result["has_content"]:
            reasons.append("Phản hồi trống hoặc quá ngắn")
        if match_ratio < 0.2:
            reasons.append(f"Tỷ lệ khớp từ khóa thấp: {len(result['matched_keywords'])}/{len(expected)}")
        if not no_rejected:
            reasons.append(f"Chứa từ mâu thuẫn: {result['rejected_keywords_found']}")
        result["reason"] = "; ".join(reasons) if reasons else "Không đạt kiểm tra chất lượng"

    return result


def run_tests(base_url: str) -> list[dict]:
    """Chạy tất cả các trường hợp kiểm thử và trả về kết quả."""
    results = []

    for i, test in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] {test['id']}: {test['description']}...")

        try:
            if test.get("use_responses_api"):
                resp = send_responses_api(test["question"], base_url)
            elif test.get("use_streaming"):
                resp = send_chat(test["question"], base_url, stream=True)
            else:
                resp = send_chat(test["question"], base_url)

            evaluation = evaluate_response(test, resp["text"])

            result = {
                "id": test["id"],
                "category": test["category"],
                "description": test["description"],
                "question": test["question"],
                "response": resp["text"][:500],
                "full_response_length": len(resp["text"]),
                "elapsed_seconds": round(resp["elapsed"], 2),
                "status_code": resp["status"],
                "pass": evaluation["pass"],
                "reason": evaluation["reason"],
                "matched_keywords": evaluation["matched_keywords"],
                "missing_keywords": evaluation["missing_keywords"],
                "rejected_keywords_found": evaluation["rejected_keywords_found"],
            }
            status = "ĐẠT" if result["pass"] else "TRƯỢT"
            print(f"  [{status}] {evaluation['reason']} ({resp['elapsed']:.1f}s)")

        except Exception as e:
            result = {
                "id": test["id"],
                "category": test["category"],
                "description": test["description"],
                "question": test["question"],
                "response": "",
                "full_response_length": 0,
                "elapsed_seconds": 0,
                "status_code": 0,
                "pass": False,
                "reason": f"LỖI: {e}",
                "matched_keywords": [],
                "missing_keywords": test.get("expected_keywords", []),
                "rejected_keywords_found": [],
            }
            print(f"  [LỖI] {e}")

        results.append(result)

    return results


def write_report(results: list[dict], output_path: Path):
    """Ghi kết quả kiểm thử ra file báo cáo."""
    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    failed = total - passed
    avg_time = sum(r["elapsed_seconds"] for r in results) / total if total else 0

    lines = []
    lines.append("=" * 70)
    lines.append("  Báo Cáo Kiểm Thử FinGPT Bot — Tiếng Việt")
    lines.append(f"  Ngày: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Mô hình: fingpt-llama3.1-8b-npu (INT4, OpenVINO, NPU)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  TÓM TẮT: {passed}/{total} đạt, {failed} trượt")
    lines.append(f"  Thời gian phản hồi trung bình: {avg_time:.1f}s")
    lines.append("")
    lines.append("-" * 70)

    # Nhóm theo danh mục
    categories = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    for cat, cat_results in categories.items():
        cat_passed = sum(1 for r in cat_results if r["pass"])
        lines.append(f"\n{'=' * 70}")
        lines.append(f"  {cat} ({cat_passed}/{len(cat_results)} đạt)")
        lines.append(f"{'=' * 70}")

        for r in cat_results:
            status = "ĐẠT" if r["pass"] else "TRƯỢT"
            lines.append(f"\n  [{status}] {r['id']}: {r['description']}")
            lines.append(f"  Thời gian: {r['elapsed_seconds']}s | Độ dài phản hồi: {r['full_response_length']} ký tự")
            lines.append(f"  Câu hỏi: {r['question'][:100]}...")
            lines.append(f"  Kết quả: {r['reason']}")
            if r["matched_keywords"]:
                lines.append(f"  Từ khóa khớp: {', '.join(r['matched_keywords'])}")
            if r["missing_keywords"]:
                lines.append(f"  Từ khóa thiếu: {', '.join(r['missing_keywords'])}")
            if r["rejected_keywords_found"]:
                lines.append(f"  TỪ BỊ LOẠI PHÁT HIỆN: {', '.join(r['rejected_keywords_found'])}")
            lines.append(f"  Xem trước phản hồi: {r['response'][:200]}...")
            lines.append(f"  {'-' * 66}")

    lines.append(f"\n{'=' * 70}")
    lines.append("  KẾT THÚC BÁO CÁO")
    lines.append(f"{'=' * 70}\n")

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")

    # Lưu JSON gốc
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return report_text


def main():
    parser = argparse.ArgumentParser(description="Bộ kiểm thử FinGPT Bot — Tiếng Việt")
    parser.add_argument("--base-url", default=BASE_URL, help="URL cơ sở của server API")
    args = parser.parse_args()

    print(f"Kiểm thử FinGPT tại {args.base_url}")

    # Kiểm tra kết nối server
    try:
        health = requests.get(f"{args.base_url}/health", timeout=5).json()
        print(f"Server: {health['model']} trên {health['device']}")
    except Exception as e:
        print(f"Không thể kết nối server: {e}")
        return

    results = run_tests(args.base_url)

    output_dir = Path(__file__).parent.parent / "test_results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"test_report_vi_{timestamp}.txt"

    report = write_report(results, output_path)
    print(f"\n{report}")
    print(f"Báo cáo đã lưu tại: {output_path}")
    print(f"JSON đã lưu tại: {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
