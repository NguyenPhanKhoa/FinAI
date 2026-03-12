"""
FinGPT Bot Test Suite
Sends financial questions to the local FinGPT API server and evaluates
whether responses are relevant, coherent, and financially accurate.

Usage:
  python tests/test_fingpt.py              # default: http://127.0.0.1:8000
  python tests/test_fingpt.py --base-url http://127.0.0.1:9000
"""

import argparse
import json
import time
import re
import requests
from pathlib import Path
from datetime import datetime


BASE_URL = "http://127.0.0.1:8000"

# --- Test Cases ---
# Each test has: category, question, expected_keywords (at least some should appear),
# and reject_keywords (none should appear, indicates hallucination or wrong domain)

TEST_CASES = [
    # --- Sentiment Analysis ---
    {
        "id": "SA-01",
        "category": "Sentiment Analysis",
        "question": "What is the sentiment of this news: 'Apple reported record quarterly revenue of $124 billion, beating analyst expectations by 5%.'",
        "expected_keywords": ["positive", "bullish", "strong", "beat", "revenue", "growth"],
        "reject_keywords": ["negative", "bearish", "decline", "loss"],
        "description": "Clearly positive earnings news",
    },
    {
        "id": "SA-02",
        "category": "Sentiment Analysis",
        "question": "Analyze the sentiment: 'Tesla shares plunged 12% after the company missed delivery targets and announced layoffs affecting 10% of its workforce.'",
        "expected_keywords": ["negative", "bearish", "drop", "decline", "miss", "layoff"],
        "reject_keywords": ["positive", "bullish", "growth", "beat"],
        "description": "Clearly negative stock news",
    },
    {
        "id": "SA-03",
        "category": "Sentiment Analysis",
        "question": "What is the market sentiment of: 'The Federal Reserve held interest rates steady, signaling a wait-and-see approach amid mixed economic data.'",
        "expected_keywords": ["neutral", "mixed", "uncertain", "steady", "hold", "wait"],
        "reject_keywords": [],
        "description": "Neutral/mixed monetary policy news",
    },

    # --- Financial Knowledge ---
    {
        "id": "FK-01",
        "category": "Financial Knowledge",
        "question": "What is the Price-to-Earnings (P/E) ratio and why is it important for stock valuation?",
        "expected_keywords": ["price", "earnings", "ratio", "valuation", "stock", "share"],
        "reject_keywords": [],
        "description": "Basic financial metric definition",
    },
    {
        "id": "FK-02",
        "category": "Financial Knowledge",
        "question": "Explain the difference between a bull market and a bear market.",
        "expected_keywords": ["bull", "bear", "rise", "fall", "market", "price"],
        "reject_keywords": [],
        "description": "Fundamental market concepts",
    },
    {
        "id": "FK-03",
        "category": "Financial Knowledge",
        "question": "What is diversification in investing and why is it important?",
        "expected_keywords": ["risk", "asset", "portfolio", "spread", "invest"],
        "reject_keywords": [],
        "description": "Core investment principle",
    },

    # --- Market Analysis ---
    {
        "id": "MA-01",
        "category": "Market Analysis",
        "question": "If a company's revenue is growing 20% year-over-year but its stock price has dropped 30%, what could explain this divergence?",
        "expected_keywords": ["valuation", "market", "expect", "growth", "price", "investor"],
        "reject_keywords": [],
        "description": "Revenue vs stock price analysis",
    },
    {
        "id": "MA-02",
        "category": "Market Analysis",
        "question": "What are the key financial indicators to watch during a recession?",
        "expected_keywords": ["gdp", "unemployment", "inflation", "interest", "consumer", "rate"],
        "reject_keywords": [],
        "description": "Recession indicators knowledge",
    },

    # --- Financial Text Processing ---
    {
        "id": "FT-01",
        "category": "Financial Text Processing",
        "question": "Summarize this earnings report excerpt: 'Q3 revenue was $50.7B, up 8% YoY. Operating income rose to $13.2B from $11.5B. Free cash flow was $12.1B. The company repurchased $5B in shares and increased its quarterly dividend by 10%.'",
        "expected_keywords": ["revenue", "growth", "income", "cash", "dividend", "share"],
        "reject_keywords": [],
        "description": "Earnings report summarization",
    },
    {
        "id": "FT-02",
        "category": "Financial Text Processing",
        "question": "Extract the key financial metrics from: 'XYZ Corp reported EPS of $3.42 vs expected $3.15, revenue of $28.5B vs expected $27.8B, and gross margin improved to 45.2% from 43.8% last year.'",
        "expected_keywords": ["eps", "revenue", "margin", "beat", "expect"],
        "reject_keywords": [],
        "description": "Financial data extraction",
    },

    # --- Risk Assessment ---
    {
        "id": "RA-01",
        "category": "Risk Assessment",
        "question": "What are the main risks of investing heavily in a single technology stock?",
        "expected_keywords": ["risk", "concentration", "volatil", "diversif", "sector", "loss"],
        "reject_keywords": [],
        "description": "Concentration risk awareness",
    },

    # --- Domain Boundary ---
    {
        "id": "DB-01",
        "category": "Domain Boundary",
        "question": "Write me a poem about flowers.",
        "expected_keywords": [],
        "reject_keywords": [],
        "description": "Non-financial request — should still respond but may redirect to finance",
        "check_type": "domain_awareness",
    },

    # --- Streaming Endpoint ---
    {
        "id": "ST-01",
        "category": "Streaming",
        "question": "What is a stock split and how does it affect shareholders?",
        "expected_keywords": ["stock", "split", "share", "price", "value"],
        "reject_keywords": [],
        "description": "Test streaming SSE endpoint",
        "use_streaming": True,
    },

    # --- Responses API ---
    {
        "id": "RP-01",
        "category": "Responses API",
        "question": "Define market capitalization.",
        "expected_keywords": ["market", "cap", "share", "price", "value", "company"],
        "reject_keywords": [],
        "description": "Test /v1/responses endpoint",
        "use_responses_api": True,
    },
]


# --- Test Runner ---

def send_chat(question: str, base_url: str, stream: bool = False) -> dict:
    """Send a chat completion request and return response info."""
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
    """Send a request to /v1/responses endpoint."""
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
    """Evaluate response quality against expected/reject keywords."""
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

    # Scoring
    if test.get("check_type") == "domain_awareness":
        result["pass"] = result["has_content"]
        result["reason"] = "Response generated" if result["pass"] else "Empty response"
    elif match_ratio >= 0.4 and no_rejected and result["has_content"]:
        result["pass"] = True
        result["reason"] = f"Matched {len(result['matched_keywords'])}/{len(expected)} keywords, no rejected terms"
    elif match_ratio >= 0.2 and no_rejected and result["has_content"]:
        result["pass"] = True
        result["reason"] = f"Partial match {len(result['matched_keywords'])}/{len(expected)} keywords (acceptable)"
    else:
        result["pass"] = False
        reasons = []
        if not result["has_content"]:
            reasons.append("Empty or very short response")
        if match_ratio < 0.2:
            reasons.append(f"Low keyword match: {len(result['matched_keywords'])}/{len(expected)}")
        if not no_rejected:
            reasons.append(f"Contains contradictory terms: {result['rejected_keywords_found']}")
        result["reason"] = "; ".join(reasons) if reasons else "Failed quality check"

    return result


def run_tests(base_url: str) -> list[dict]:
    """Run all test cases and return results."""
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
                "response": resp["text"][:500],  # Truncate for readability
                "full_response_length": len(resp["text"]),
                "elapsed_seconds": round(resp["elapsed"], 2),
                "status_code": resp["status"],
                "pass": evaluation["pass"],
                "reason": evaluation["reason"],
                "matched_keywords": evaluation["matched_keywords"],
                "missing_keywords": evaluation["missing_keywords"],
                "rejected_keywords_found": evaluation["rejected_keywords_found"],
            }
            status = "PASS" if result["pass"] else "FAIL"
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
                "reason": f"ERROR: {e}",
                "matched_keywords": [],
                "missing_keywords": test.get("expected_keywords", []),
                "rejected_keywords_found": [],
            }
            print(f"  [ERROR] {e}")

        results.append(result)

    return results


def write_report(results: list[dict], output_path: Path):
    """Write test results to a readable report file."""
    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    failed = total - passed
    avg_time = sum(r["elapsed_seconds"] for r in results) / total if total else 0

    lines = []
    lines.append("=" * 70)
    lines.append("  FinGPT Bot Test Report")
    lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Model: fingpt-llama3.1-8b-npu (INT4, OpenVINO, NPU)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  SUMMARY: {passed}/{total} passed, {failed} failed")
    lines.append(f"  Average response time: {avg_time:.1f}s")
    lines.append("")
    lines.append("-" * 70)

    # Group by category
    categories = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    for cat, cat_results in categories.items():
        cat_passed = sum(1 for r in cat_results if r["pass"])
        lines.append(f"\n{'=' * 70}")
        lines.append(f"  {cat} ({cat_passed}/{len(cat_results)} passed)")
        lines.append(f"{'=' * 70}")

        for r in cat_results:
            status = "PASS" if r["pass"] else "FAIL"
            lines.append(f"\n  [{status}] {r['id']}: {r['description']}")
            lines.append(f"  Time: {r['elapsed_seconds']}s | Response length: {r['full_response_length']} chars")
            lines.append(f"  Question: {r['question'][:100]}...")
            lines.append(f"  Result: {r['reason']}")
            if r["matched_keywords"]:
                lines.append(f"  Matched: {', '.join(r['matched_keywords'])}")
            if r["missing_keywords"]:
                lines.append(f"  Missing: {', '.join(r['missing_keywords'])}")
            if r["rejected_keywords_found"]:
                lines.append(f"  REJECTED TERMS FOUND: {', '.join(r['rejected_keywords_found'])}")
            lines.append(f"  Response preview: {r['response'][:200]}...")
            lines.append(f"  {'-' * 66}")

    lines.append(f"\n{'=' * 70}")
    lines.append("  END OF REPORT")
    lines.append(f"{'=' * 70}\n")

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")

    # Also save raw JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return report_text


def main():
    parser = argparse.ArgumentParser(description="FinGPT Bot Test Suite")
    parser.add_argument("--base-url", default=BASE_URL, help="API server base URL")
    args = parser.parse_args()

    print(f"Testing FinGPT at {args.base_url}")

    # Check server health
    try:
        health = requests.get(f"{args.base_url}/health", timeout=5).json()
        print(f"Server: {health['model']} on {health['device']}")
    except Exception as e:
        print(f"Server not reachable: {e}")
        return

    results = run_tests(args.base_url)

    output_dir = Path(__file__).parent.parent / "test_results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"test_report_{timestamp}.txt"

    report = write_report(results, output_path)
    print(f"\n{report}")
    print(f"Report saved to: {output_path}")
    print(f"JSON saved to: {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
