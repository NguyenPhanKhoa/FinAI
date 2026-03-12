/**
 * Financial News Fetcher
 * Sources:
 *   - Vietnam: CafeF, VnEconomy, VNExpress, VietnamBiz, Tinnhanhchungkhoan (RSS, free)
 *   - Global hot: NewsAPI top headlines sorted by popularity (free tier)
 */

import axios from "axios";
import RSSParser from "rss-parser";

const rss = new RSSParser({ timeout: 8000 });

// Global hot news RSS feeds (all free, no key needed)
const GLOBAL_RSS_FEEDS = [
  { name: "Reuters Business", url: "https://feeds.reuters.com/reuters/businessNews", flag: "🌍" },
  { name: "CNBC Finance",     url: "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664", flag: "🌍" },
  { name: "BBC Business",     url: "https://feeds.bbci.co.uk/news/business/rss.xml", flag: "🌍" },
  { name: "MarketWatch",      url: "https://feeds.content.dowjones.io/public/rss/mw_topstories", flag: "🌍" },
];

// Vietnam finance RSS feeds (all free, no key needed)
const VIETNAM_RSS_FEEDS = [
  {
    name: "CafeF",
    url: "https://cafef.vn/rss/thi-truong-chung-khoan.rss",
    flag: "🇻🇳",
  },
  {
    name: "VnEconomy",
    url: "https://vneconomy.vn/rss/home.rss",
    flag: "🇻🇳",
  },
  {
    name: "VNExpress Kinh Doanh",
    url: "https://vnexpress.net/rss/kinh-doanh.rss",
    flag: "🇻🇳",
  },
  {
    name: "VietnamBiz",
    url: "https://vietnambiz.vn/rss.rss",
    flag: "🇻🇳",
  },
  {
    name: "Tin Nhanh Chứng Khoán",
    url: "https://tinnhanhchungkhoan.vn/rss/tin-moi-nhat.rss",
    flag: "🇻🇳",
  },
];

/**
 * Fetch Vietnam financial news from multiple RSS feeds.
 * @param {number} perFeed - Articles to take from each feed (default: 3)
 */
export async function fetchVietnamNews(perFeed = 3) {
  const results = await Promise.allSettled(
    VIETNAM_RSS_FEEDS.map((feed) => rss.parseURL(feed.url))
  );

  const articles = [];
  results.forEach((result, i) => {
    if (result.status !== "fulfilled") return;
    const feed = VIETNAM_RSS_FEEDS[i];
    result.value.items.slice(0, perFeed).forEach((item) => {
      articles.push({
        title: item.title?.trim(),
        description: item.contentSnippet?.trim() || item.summary?.trim() || "",
        url: item.link,
        publishedAt: item.pubDate || item.isoDate || new Date().toISOString(),
        source: `${feed.flag} ${feed.name}`,
        region: "VN",
      });
    });
  });

  // Sort by most recent
  return articles
    .filter((a) => a.title)
    .sort((a, b) => new Date(b.publishedAt) - new Date(a.publishedAt));
}

/**
 * Fetch hot global financial news from free RSS feeds.
 * No API key required.
 * @param {number} perFeed - Articles per feed (default: 2)
 */
export async function fetchGlobalHotNews(perFeed = 2) {
  const results = await Promise.allSettled(
    GLOBAL_RSS_FEEDS.map((feed) => rss.parseURL(feed.url))
  );

  const articles = [];
  results.forEach((result, i) => {
    if (result.status !== "fulfilled") return;
    const feed = GLOBAL_RSS_FEEDS[i];
    result.value.items.slice(0, perFeed).forEach((item) => {
      articles.push({
        title: item.title?.trim(),
        description: item.contentSnippet?.trim() || item.summary?.trim() || "",
        url: item.link,
        publishedAt: item.pubDate || item.isoDate || new Date().toISOString(),
        source: `${feed.flag} ${feed.name}`,
        region: "GLOBAL",
      });
    });
  });

  return articles
    .filter((a) => a.title)
    .sort((a, b) => new Date(b.publishedAt) - new Date(a.publishedAt));
}

/**
 * Fetch company-specific news from Finnhub.
 * @param {string} ticker
 */
export async function fetchFinnhubNews(ticker) {
  const today = new Date();
  const weekAgo = new Date(today - 7 * 24 * 60 * 60 * 1000);

  const res = await axios.get("https://finnhub.io/api/v1/company-news", {
    params: {
      symbol: ticker.toUpperCase(),
      from: weekAgo.toISOString().split("T")[0],
      to: today.toISOString().split("T")[0],
      token: process.env.FINNHUB_API_KEY,
    },
  });

  return res.data.slice(0, 10).map((a) => ({
    title: a.headline,
    description: a.summary,
    url: a.url,
    publishedAt: new Date(a.datetime * 1000).toISOString(),
    source: `🌍 ${a.source}`,
    region: "GLOBAL",
  }));
}

/**
 * Format a news list as a readable briefing string.
 */
export function formatNewsBriefing(articles, title = "Financial News Briefing") {
  const date = new Date().toLocaleDateString("vi-VN", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "Asia/Ho_Chi_Minh",
  });

  const lines = [`📰 *${title}* — ${date}`, ""];

  articles.forEach((a, i) => {
    lines.push(`${i + 1}. *${a.title}*`);
    if (a.description) lines.push(`   ${a.description.slice(0, 130)}...`);
    lines.push(`   📌 ${a.source}  🔗 ${a.url}`);
    lines.push("");
  });

  return lines.join("\n");
}
