/**
 * Stock Data Fetcher
 * Uses yahoo-finance2 for free real-time stock data.
 */

import yahooFinance from "yahoo-finance2";

/**
 * Get current stock quote.
 * @param {string} ticker - e.g. "AAPL", "TSLA"
 */
export async function getQuote(ticker) {
  const quote = await yahooFinance.quote(ticker.toUpperCase());
  return {
    ticker: quote.symbol,
    name: quote.shortName,
    price: quote.regularMarketPrice,
    change: quote.regularMarketChange,
    changePercent: quote.regularMarketChangePercent,
    volume: quote.regularMarketVolume,
    marketCap: quote.marketCap,
    peRatio: quote.trailingPE,
    high52w: quote.fiftyTwoWeekHigh,
    low52w: quote.fiftyTwoWeekLow,
  };
}

/**
 * Get historical price data for the last N days.
 * @param {string} ticker
 * @param {number} days
 */
export async function getHistoricalPrices(ticker, days = 30) {
  const endDate = new Date();
  const startDate = new Date(endDate - days * 24 * 60 * 60 * 1000);

  const result = await yahooFinance.historical(ticker.toUpperCase(), {
    period1: startDate.toISOString().split("T")[0],
    period2: endDate.toISOString().split("T")[0],
    interval: "1d",
  });

  return result.map((r) => ({
    date: r.date.toISOString().split("T")[0],
    open: r.open,
    high: r.high,
    low: r.low,
    close: r.close,
    volume: r.volume,
  }));
}

/**
 * Format stock quote as a readable summary.
 */
export function formatQuote(q) {
  const sign = q.change >= 0 ? "+" : "";
  const emoji = q.change >= 0 ? "📈" : "📉";

  return [
    `${emoji} *${q.ticker}* — ${q.name}`,
    `Price: $${q.price?.toFixed(2)} (${sign}${q.change?.toFixed(2)}, ${sign}${q.changePercent?.toFixed(2)}%)`,
    `Volume: ${(q.volume / 1e6).toFixed(2)}M | Market Cap: $${(q.marketCap / 1e9).toFixed(2)}B`,
    `P/E Ratio: ${q.peRatio?.toFixed(2) ?? "N/A"}`,
    `52W Range: $${q.low52w?.toFixed(2)} — $${q.high52w?.toFixed(2)}`,
  ].join("\n");
}
