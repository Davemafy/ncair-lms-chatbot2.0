const DEFAULT_UPSTREAM =
  "https://8000-01m3v6dv3dbzrxn6resd7ms958.cloudspaces.litng.ai";

function upstreamBase() {
  return String(process.env.NCAIR_UPSTREAM_URL || DEFAULT_UPSTREAM).replace(/\/+$/, "");
}

async function forward(req, res, path, methods) {
  if (!methods.includes(req.method)) {
    res.setHeader("Allow", methods.join(", "));
    return res.status(405).json({ detail: "Method not allowed." });
  }

  try {
    const upstream = await fetch(`${upstreamBase()}${path}`, {
      method: req.method,
      headers: {
        "Content-Type": "application/json",
      },
      body: req.method === "GET" ? undefined : JSON.stringify(req.body ?? {}),
    });

    const contentType = upstream.headers.get("content-type") || "application/json";
    const body = await upstream.text();

    res.status(upstream.status);
    res.setHeader("Content-Type", contentType);
    return res.send(body);
  } catch {
    return res.status(503).json({
      detail: "The NCAIR V2 backend is temporarily unavailable.",
    });
  }
}

module.exports = { forward };
