/**
 * FinGPT Keyword Middleware
 * Sits between OpenClaw (port 11436) and the NPU server (port 11435).
 *
 * When a user message matches a finance keyword:
 *   1. Automatically runs the matching fin-* command
 *   2. Injects the result into the conversation context
 *   3. Forwards to NPU server so the model just summarises the data
 *
 * OpenClaw → port 11436 (this middleware) → port 11435 (NPU server)
 */

import express from "express";
import axios from "axios";
import { execFile } from "child_process";
import { promisify } from "util";
import path from "path";
import { fileURLToPath } from "url";

const exec = promisify(execFile);
const app = express();
app.use(express.json({ limit: "2mb" }));

const NPU_URL = "http://127.0.0.1:11435";
const PORT = 11436;
const BINS = path.join(path.dirname(fileURLToPath(import.meta.url)), "bins");

// ── Keyword routing table ─────────────────────────────────────────────────────
const ROUTES = [
  // News
  { pattern: /tin\s*(tức|nóng|nhanh|hôm nay|mới|thị trường)|bản\s*tin|news|today.*news|morning/i,
    cmd: ["fin-news.js"] },
  { pattern: /tin\s*(việt\s*nam|vn\b|trong\s*nước)/i,
    cmd: ["fin-news.js", "vn"] },
  { pattern: /tin\s*(thế\s*giới|quốc\s*tế|global|world)/i,
    cmd: ["fin-news.js", "global"] },

  // Sentiment
  { pattern: /cảm\s*xúc|sentiment|phân\s*tích.*[:"]|analyze.*[:"]|tích\s*cực|tiêu\s*cực/i,
    cmd: null, handler: "sentiment" },

  // Signal / buy-sell
  { pattern: /nên\s*(mua|bán)|buy\s*or\s*sell|trading\s*signal|tín\s*hiệu|signal\s+([A-Z]{2,5})/i,
    cmd: null, handler: "signal" },

  // Forecast
  { pattern: /dự\s*báo|forecast|predict|giá\s*cổ\s*phiếu|price\s*prediction/i,
    cmd: null, handler: "forecast" },

  // Search
  { pattern: /tìm\s*(kiếm)?|search|tra\s*cứu|lãi\s*suất|tỷ\s*giá|giá\s*vàng|chỉ\s*số|vn.?index/i,
    cmd: null, handler: "search" },
];

// Extract ticker from message (e.g. "VCB", "MSN", "AAPL")
function extractTicker(msg) {
  const m = msg.match(/\b([A-Z]{2,5})\b/);
  return m ? m[1] : null;
}

// Extract quoted text from message
function extractQuoted(msg) {
  const m = msg.match(/["""](.*?)["""]/);
  return m ? m[1] : msg.replace(/^.*?[:：]/,'').trim();
}

// Extract search query (everything after the trigger word)
function extractQuery(msg) {
  return msg.replace(/tìm kiếm|tìm|search|tra cứu/i, "").trim() || msg;
}

async function runCommand(args) {
  const [script, ...rest] = args;
  const scriptPath = path.join(BINS, script);
  const { stdout, stderr } = await exec("node", [scriptPath, ...rest], {
    cwd: path.dirname(fileURLToPath(import.meta.url)),
    timeout: 30000,
    env: { ...process.env }
  });
  return stdout || stderr;
}

async function detectAndRun(userMessage) {
  for (const route of ROUTES) {
    if (!route.pattern.test(userMessage)) continue;

    if (route.cmd) {
      // Direct command
      return await runCommand(route.cmd);
    }

    switch (route.handler) {
      case "sentiment": {
        const text = extractQuoted(userMessage);
        return await runCommand(["fin-analyze.js", "sentiment", text]);
      }
      case "signal": {
        const ticker = extractTicker(userMessage.toUpperCase()) || "VCB";
        return await runCommand(["fin-analyze.js", "signal", ticker]);
      }
      case "forecast": {
        const ticker = extractTicker(userMessage.toUpperCase()) || "VCB";
        return await runCommand(["fin-analyze.js", "forecast", ticker]);
      }
      case "search": {
        const query = extractQuery(userMessage);
        return await runCommand(["fin-search.js", query]);
      }
    }
  }
  return null; // No match — let model handle it normally
}

// ── Proxy endpoints ───────────────────────────────────────────────────────────

app.get("/api/tags", async (_, res) => {
  const r = await axios.get(`${NPU_URL}/api/tags`);
  res.json(r.data);
});

app.post("/api/chat", async (req, res) => {
  const body = req.body;
  const messages = body.messages || [];
  const lastUser = [...messages].reverse().find(m => m.role === "user");
  const userText = lastUser?.content || "";

  let injected = false;
  try {
    const cmdResult = await detectAndRun(userText);
    if (cmdResult) {
      // Inject live data as system context before the model responds
      body.messages = [
        ...messages,
        {
          role: "system",
          content: `[Live data fetched automatically]\n\n${cmdResult}\n\nSummarize the above data for the user in a clear, friendly way. Respond in the same language the user used.`
        }
      ];
      injected = true;
    }
  } catch (e) {
    console.error("[middleware] command failed:", e.message);
  }

  const r = await axios.post(`${NPU_URL}/api/chat`, body);
  res.json(r.data);
  if (injected) console.log(`[middleware] injected live data for: "${userText.slice(0,60)}"`);
});

app.post("/api/generate", async (req, res) => {
  const r = await axios.post(`${NPU_URL}/api/generate`, req.body);
  res.json(r.data);
});

app.get("/v1/models", async (_, res) => {
  const r = await axios.get(`${NPU_URL}/v1/models`);
  res.json(r.data);
});

app.post("/v1/chat/completions", async (req, res) => {
  const r = await axios.post(`${NPU_URL}/v1/chat/completions`, req.body);
  res.json(r.data);
});

app.get("/health", async (_, res) => {
  const r = await axios.get(`${NPU_URL}/health`);
  res.json({ ...r.data, middleware: "ok", port: PORT });
});

app.listen(PORT, "127.0.0.1", () => {
  console.log(`[FinGPT Middleware] Running on http://127.0.0.1:${PORT}`);
  console.log(`[FinGPT Middleware] Proxying to NPU server at ${NPU_URL}`);
});
