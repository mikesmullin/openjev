import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const port = Number(process.env.PORT ?? 3000);
const layaUrl = process.env.LAYA_URL ?? "http://127.0.0.1:8787";

app.disable("x-powered-by");
app.use(express.json({ limit: "512kb" }));

async function proxyJSON(pathname, options = {}) {
  const response = await fetch(`${layaUrl}${pathname}`, {
    ...options,
    headers: {
      accept: "application/json",
      ...(options.body ? { "content-type": "application/json" } : {}),
      ...options.headers,
    },
  });
  const body = await response.text();
  return { response, body };
}

app.get("/api/health", async (_request, response) => {
  try {
    const upstream = await proxyJSON("/healthz");
    response.status(upstream.response.status).type("json").send(upstream.body);
  } catch (error) {
    response.status(503).json({ status: "unavailable", error: String(error) });
  }
});

app.post("/api/laya/tetris/score", async (request, response) => {
  try {
    const upstream = await proxyJSON("/v1/laya/tetris/score", {
      method: "POST",
      body: JSON.stringify(request.body),
    });
    response.status(upstream.response.status).type("json").send(upstream.body);
  } catch (error) {
    response.status(502).json({ error: `Laya API unavailable: ${String(error)}` });
  }
});

app.use(express.static(path.join(__dirname, "public"), { extensions: ["html"] }));
app.use((_request, response) => {
  response.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(port, "127.0.0.1", () => {
  console.log(`Browser app: http://127.0.0.1:${port}`);
  console.log(`Laya API:    ${layaUrl}`);
});
