/**
 * FinGPT Local Client
 * Uses Ollama running locally — completely FREE, no API key needed.
 * Model: fingpt (llama3.1:8b + FinGPT system prompt via Modelfile)
 *
 * Ollama exposes an OpenAI-compatible API at http://localhost:11434
 */

import axios from "axios";

const OLLAMA_URL = "http://localhost:11434/api/generate";
const MODEL_NAME = "fingpt"; // built from Modelfile

/**
 * Send a prompt to the local FinGPT model via Ollama.
 * @param {string} prompt
 * @param {number} maxTokens
 */
async function query(prompt, maxTokens = 200) {
  const res = await axios.post(OLLAMA_URL, {
    model: MODEL_NAME,
    prompt,
    stream: false,
    options: { num_predict: maxTokens },
  });
  return res.data.response.trim();
}

/**
 * Analyze sentiment of a financial text.
 * Returns: { label: "positive"|"negative"|"neutral" }
 */
export async function analyzeSentiment(text) {
  const prompt = `Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.
Input: ${text}
Answer:`;

  const answer = await query(prompt, 10);
  const lower = answer.toLowerCase();
  const label = ["positive", "negative", "neutral"].find((l) =>
    lower.includes(l)
  ) ?? "neutral";

  return { label, raw: answer };
}

/**
 * Forecast stock movement based on news headlines.
 * Returns: { direction: "up"|"down"|"stable", reasoning: string }
 */
export async function forecastStock(ticker, headlines) {
  const headlineList = headlines
    .slice(0, 5)
    .map((h, i) => `${i + 1}. ${h}`)
    .join("\n");

  const prompt = `Instruction: Based on the following news headlines for ${ticker}, predict whether the stock price will go up, down, or stay stable. Explain briefly.
Headlines:
${headlineList}
Answer:`;

  const text = await query(prompt, 150);
  const lower = text.toLowerCase();
  const direction = lower.includes("up")
    ? "up"
    : lower.includes("down")
      ? "down"
      : "stable";

  return { direction, reasoning: text };
}

/**
 * Answer a financial question.
 * Returns: { answer: string }
 */
export async function answerFinancialQuestion(question) {
  const prompt = `Instruction: You are a financial expert. Answer the following question accurately and concisely.
Question: ${question}
Answer:`;

  const answer = await query(prompt, 300);
  return { answer };
}

/**
 * Generate a trading signal based on sentiment of recent news.
 * Returns: { signal: "BUY"|"SELL"|"HOLD", confidence: number, reasoning: string }
 */
export async function generateTradingSignal(ticker, sentimentResults) {
  const positive = sentimentResults.filter((s) => s.label === "positive").length;
  const negative = sentimentResults.filter((s) => s.label === "negative").length;
  const total = sentimentResults.length || 1;

  const positiveRatio = positive / total;
  const negativeRatio = negative / total;

  let signal, confidence, reasoning;

  if (positiveRatio >= 0.6) {
    signal = "BUY";
    confidence = Math.round(positiveRatio * 100);
    reasoning = `${positive}/${total} headlines are positive for ${ticker}.`;
  } else if (negativeRatio >= 0.6) {
    signal = "SELL";
    confidence = Math.round(negativeRatio * 100);
    reasoning = `${negative}/${total} headlines are negative for ${ticker}.`;
  } else {
    signal = "HOLD";
    confidence = 50;
    reasoning = `Mixed sentiment for ${ticker}: ${positive} positive, ${negative} negative, ${total - positive - negative} neutral.`;
  }

  return { signal, confidence, reasoning };
}
