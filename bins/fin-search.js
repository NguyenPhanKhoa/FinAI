#!/usr/bin/env node
/**
 * fin-search — Search the internet for financial information
 * Uses DuckDuckGo (free, no API key needed)
 * Usage: fin-search "lãi suất ngân hàng 2026"
 */

import "dotenv/config";
import axios from "axios";

const query = process.argv.slice(2).join(" ");

if (!query) {
  console.error("Usage: fin-search \"your query\"");
  process.exit(1);
}

async function search(q) {
  // DuckDuckGo instant answer API (free, no key)
  const res = await axios.get("https://api.duckduckgo.com/", {
    params: {
      q: `${q} finance`,
      format: "json",
      no_html: 1,
      skip_disambig: 1,
    },
    headers: { "User-Agent": "FinGPT-Agent/1.0" },
  });

  const data = res.data;
  const results = [];

  // Abstract (main answer)
  if (data.AbstractText) {
    results.push(`📖 ${data.AbstractText}`);
    if (data.AbstractURL) results.push(`🔗 ${data.AbstractURL}`);
  }

  // Related topics
  if (data.RelatedTopics?.length > 0) {
    results.push("\n📌 Related:");
    data.RelatedTopics.slice(0, 5).forEach((t) => {
      if (t.Text) results.push(`  • ${t.Text}`);
    });
  }

  // Infobox
  if (data.Infobox?.content?.length > 0) {
    results.push("\n📊 Data:");
    data.Infobox.content.slice(0, 6).forEach((item) => {
      results.push(`  ${item.label}: ${item.value}`);
    });
  }

  if (results.length === 0) {
    // Fallback: show DuckDuckGo search link
    results.push(`No instant answer found.`);
    results.push(`🔍 Search manually: https://duckduckgo.com/?q=${encodeURIComponent(q + " finance")}`);
  }

  return results.join("\n");
}

async function main() {
  try {
    console.log(`\n🔍 Searching: "${query}"\n`);
    const result = await search(query);
    console.log(result);
  } catch (err) {
    console.error("Search error:", err.message);
    process.exit(1);
  }
}

main();
