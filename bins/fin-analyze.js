#!/usr/bin/env node
/**
 * fin-analyze — CLI tool for FinGPT financial analysis
 * Usage:
 *   fin-analyze sentiment "Apple beats earnings"   → sentiment analysis
 *   fin-analyze forecast AAPL                      → stock forecast
 *   fin-analyze qa "What is Tesla's P/E ratio?"   → financial Q&A
 *   fin-analyze signal TSLA                        → trading signal
 */

import "dotenv/config";
import {
  analyzeSentiment,
  forecastStock,
  answerFinancialQuestion,
  generateTradingSignal,
} from "../src/fingpt.js";
import { fetchFinnhubNews } from "../src/news.js";
import { getQuote, formatQuote } from "../src/stocks.js";

const [, , command, ...args] = process.argv;

async function main() {
  if (!command) {
    console.log(`Usage:
  fin-analyze sentiment "<text>"     Analyze sentiment of a financial text
  fin-analyze forecast <ticker>      Forecast stock movement from news
  fin-analyze qa "<question>"        Answer a financial question
  fin-analyze signal <ticker>        Generate a trading signal`);
    process.exit(0);
  }

  try {
    switch (command.toLowerCase()) {
      case "sentiment": {
        const text = args.join(" ");
        if (!text) throw new Error("Provide text to analyze. E.g.: fin-analyze sentiment \"AAPL beats earnings\"");
        console.log(`\n📊 Analyzing sentiment for:\n"${text}"\n`);
        const result = await analyzeSentiment(text);
        const emoji =
          result.label === "positive" ? "🟢" : result.label === "negative" ? "🔴" : "🟡";
        console.log(`${emoji} Sentiment: ${result.label.toUpperCase()}`);
        break;
      }

      case "forecast": {
        const ticker = args[0];
        if (!ticker) throw new Error("Provide a ticker. E.g.: fin-analyze forecast AAPL");

        console.log(`\n🔮 Fetching news and forecasting ${ticker.toUpperCase()}...\n`);

        const [articles, quote] = await Promise.all([
          fetchFinnhubNews(ticker),
          getQuote(ticker),
        ]);

        console.log(formatQuote(quote));
        console.log();

        const headlines = articles.map((a) => a.title);
        const forecast = await forecastStock(ticker, headlines);

        const arrow = forecast.direction === "up" ? "⬆️" : forecast.direction === "down" ? "⬇️" : "➡️";
        console.log(`${arrow} Forecast: ${forecast.direction.toUpperCase()}`);
        console.log(`\nReasoning:\n${forecast.reasoning}`);
        break;
      }

      case "qa": {
        const question = args.join(" ");
        if (!question) throw new Error("Provide a question. E.g.: fin-analyze qa \"What is a P/E ratio?\"");
        console.log(`\n💬 Question: ${question}\n`);
        const result = await answerFinancialQuestion(question);
        console.log(`Answer:\n${result.answer}`);
        break;
      }

      case "signal": {
        const ticker = args[0];
        if (!ticker) throw new Error("Provide a ticker. E.g.: fin-analyze signal TSLA");

        console.log(`\n📡 Generating trading signal for ${ticker.toUpperCase()}...\n`);

        const [articles, quote] = await Promise.all([
          fetchFinnhubNews(ticker),
          getQuote(ticker),
        ]);

        console.log(formatQuote(quote));
        console.log();

        // Analyze sentiment of each headline
        const sentiments = await Promise.all(
          articles.slice(0, 8).map((a) => analyzeSentiment(a.title))
        );

        const signal = await generateTradingSignal(ticker, sentiments);

        const emoji =
          signal.signal === "BUY" ? "🟢" : signal.signal === "SELL" ? "🔴" : "🟡";
        console.log(`${emoji} Signal: ${signal.signal} (${signal.confidence}% confidence)`);
        console.log(`\nReasoning: ${signal.reasoning}`);
        console.log(
          "\n⚠️  This is AI-generated analysis, not financial advice. Always do your own research."
        );
        break;
      }

      default:
        console.error(`Unknown command: ${command}`);
        process.exit(1);
    }
  } catch (err) {
    console.error("Error:", err.message);
    process.exit(1);
  }
}

main();
