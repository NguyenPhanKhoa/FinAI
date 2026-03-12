---
name: fingpt-finance
description: FinGPT Finance AI Agent — fetches live Vietnam and global financial news, sentiment analysis, stock forecasting, Q&A, trading signals, and internet search.
version: 1.1.0
author: FinAI
openclaw:
  bins: [node]
  install:
    npm: install
tags: [finance, stocks, trading, news, fingpt, vietnam, search]
---

# FinGPT Finance AI Agent

IMPORTANT: You MUST run the appropriate command for every finance request. Do NOT reply from memory — always run the command first, then present the result to the user.

## Trigger Rules

| User says | Run this command |
|---|---|
| news / tin tức / bản tin / hôm nay | `fin-news` |
| tin Việt Nam / Vietnam news | `fin-news vn` |
| tin thế giới / global news | `fin-news global` |
| news about TICKER (e.g. AAPL, VCB) | `fin-news TICKER` |
| sentiment / cảm xúc / phân tích + text | `fin-analyze sentiment "text"` |
| forecast / dự báo / price of TICKER | `fin-analyze forecast TICKER` |
| buy or sell / mua hay bán / signal TICKER | `fin-analyze signal TICKER` |
| financial question / câu hỏi | `fin-analyze qa "question"` |
| search / tìm kiếm / tìm / find on internet | `fin-search "query"` |

## Commands

```
fin-news                          Vietnam + global hot news
fin-news vn                       Vietnam news only
fin-news global                   Global hot news only
fin-news AAPL                     News for specific stock

fin-analyze sentiment "text"      Sentiment: positive/neutral/negative
fin-analyze forecast TICKER       Price forecast: up/down/stable
fin-analyze signal TICKER         Trading signal: BUY/SELL/HOLD
fin-analyze qa "question"         Answer financial questions

fin-search "query"                Search internet for financial info
```

## Examples

- "Tin tức hôm nay?" → `fin-news`
- "Cho tôi xem tin Việt Nam" → `fin-news vn`
- "VN-Index hôm nay thế nào?" → `fin-search "VN-Index hôm nay"`
- "Nên mua VCB không?" → `fin-analyze signal VCB`
- "Dự báo cổ phiếu MSN" → `fin-analyze forecast MSN`
- "Lãi suất ngân hàng 2026?" → `fin-search "lãi suất ngân hàng Việt Nam 2026"`
- "Phân tích: VN-Index giảm 15 điểm" → `fin-analyze sentiment "VN-Index giảm 15 điểm"`

Always show full results. For trading signals add:
> ⚠️ AI-generated analysis only, not financial advice.
