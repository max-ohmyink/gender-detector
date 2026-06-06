/**
 * Gender Detection - Node.js API Server
 *
 * Setup:
 *   npm install express node-fetch@2
 *
 * Run:
 *   node server.js
 *
 * Usage:
 *   GET http://localhost:3000/api/detect?path=C:/photos/face.jpg
 */

const express = require("express");
const fetch = require("node-fetch");
const app = express();

const PYTHON_API = process.env.PYTHON_API || "http://localhost:5000";

app.get("/api/detect", async (req, res) => {
  const imagePath = req.query.path;
  if (!imagePath) {
    return res.status(400).json({ error: "Missing 'path' query parameter" });
  }

  try {
    const response = await fetch(`${PYTHON_API}/api/detect?path=${encodeURIComponent(imagePath)}`);
    const data = await response.json();
    res.status(response.status).json(data);
  } catch (err) {
    res.status(502).json({ error: `Python backend error: ${err.message}` });
  }
});

app.get("/api/health", async (_req, res) => {
  try {
    const r = await fetch(`${PYTHON_API}/api/health`);
    const data = await r.json();
    res.json({ node: "ok", python: data.status });
  } catch {
    res.json({ node: "ok", python: "unreachable" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Node.js API running at http://localhost:${PORT}`);
  console.log(`  GET /api/detect?path=<image_path>`);
  console.log(`  Python backend: ${PYTHON_API}`);
});
