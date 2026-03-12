/**
 * Daily Scheduler
 * Runs at 8:00 AM (Vietnam time, UTC+7) to deliver a combined briefing:
 *   - Vietnam finance news (CafeF, VnEconomy, VNExpress, VietnamBiz, TNCK)
 *   - Hot global news (NewsAPI top headlines or Finnhub)
 */

import "dotenv/config";
import cron from "node-cron";
import { fetchVietnamNews, fetchGlobalHotNews, formatNewsBriefing } from "./news.js";
import { analyzeSentiment } from "./fingpt.js";

async function dailyBriefing() {
  console.log("[FinGPT] Chạy bản tin tài chính hàng ngày...\n");

  try {
    // Fetch both in parallel
    const [vnArticles, globalArticles] = await Promise.all([
      fetchVietnamNews(3),
      fetchGlobalHotNews(5),
    ]);

    // Vietnam section
    console.log(formatNewsBriefing(vnArticles.slice(0, 8), "🇻🇳 Tin Tài Chính Việt Nam"));

    // Global hot section
    console.log(formatNewsBriefing(globalArticles, "🌍 Tin Nóng Thế Giới"));

    // Sentiment on top VN headlines
    console.log("📊 *Phân Tích Cảm Xúc — Tin Việt Nam*\n");
    for (const article of vnArticles.slice(0, 5)) {
      const { label } = await analyzeSentiment(article.title);
      const emoji =
        label === "positive" ? "🟢" : label === "negative" ? "🔴" : "🟡";
      console.log(`${emoji} [${label.toUpperCase()}] ${article.title}`);
    }
  } catch (err) {
    console.error("[FinGPT] Lỗi bản tin:", err.message);
  }
}

// 8:00 AM Vietnam time (UTC+7) = 1:00 AM UTC
cron.schedule("0 1 * * *", dailyBriefing, {
  timezone: "Asia/Ho_Chi_Minh",
});

console.log("[FinGPT] Scheduler đã khởi động. Bản tin chạy lúc 8:00 SA (GMT+7).");

if (process.argv.includes("--now")) {
  dailyBriefing();
}
