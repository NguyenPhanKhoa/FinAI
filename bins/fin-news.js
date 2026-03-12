#!/usr/bin/env node
/**
 * fin-news — Fetch financial news
 * Usage:
 *   fin-news                → Vietnam news + hot global news
 *   fin-news vn             → Vietnam news only
 *   fin-news global         → Hot global news only
 *   fin-news AAPL           → Company-specific news (Finnhub)
 */

import "dotenv/config";
import {
  fetchVietnamNews,
  fetchGlobalHotNews,
  fetchFinnhubNews,
  formatNewsBriefing,
} from "../src/news.js";

const arg = process.argv[2]?.toLowerCase();

async function main() {
  try {
    if (!arg || arg === "all") {
      // Default: Vietnam + hot global
      const [vn, global] = await Promise.all([
        fetchVietnamNews(3),
        fetchGlobalHotNews(5),
      ]);
      console.log(formatNewsBriefing(vn.slice(0, 9), "🇻🇳 Tin Tài Chính Việt Nam"));
      console.log(formatNewsBriefing(global, "🌍 Tin Nóng Thế Giới"));

    } else if (arg === "vn") {
      const articles = await fetchVietnamNews(4);
      console.log(formatNewsBriefing(articles, "🇻🇳 Tin Tài Chính Việt Nam"));

    } else if (arg === "global") {
      const articles = await fetchGlobalHotNews(10);
      console.log(formatNewsBriefing(articles, "🌍 Tin Nóng Thế Giới"));

    } else {
      // Assume stock ticker
      const articles = await fetchFinnhubNews(arg.toUpperCase());
      console.log(formatNewsBriefing(articles, `Tin về ${arg.toUpperCase()}`));
    }
  } catch (err) {
    console.error("Lỗi:", err.message);
    process.exit(1);
  }
}

main();
