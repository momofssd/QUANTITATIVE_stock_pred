/**
 * Tencent EdgeOne Pages API proxy; not imported into the Vite application.
 *
 * Pages settings: root = frontend, output = dist, build = npm run build.
 * Set TENCENT_BACKEND_ORIGIN in the Tencent project environment, for example
 * https://your-python-service.onrender.com (no /api suffix), then redeploy.
 * The backend must run the existing FastAPI app and expose /api/*.
 *
 * This uses Pages' functions/ routing convention. If your Tencent console uses
 * the newer Makers edge-functions/ convention, move this file to
 * frontend/edge-functions/api/[[path]].js instead; do not deploy both copies.
 * Vite and the existing FastAPI deployment do not load this file.
 */

const METHODS = {
  "/api/health": "GET",
  "/api/analyze": "POST",
  "/api/interpret": "POST",
  "/api/analyze-and-interpret": "POST",
};

function jsonError(detail, status, extraHeaders = {}) {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      ...extraHeaders,
    },
  });
}

export async function onRequest({ request, env }) {
  const incoming = new URL(request.url);
  const method = METHODS[incoming.pathname.replace(/\/$/, "")];
  if (!method) return jsonError("Unknown API endpoint.", 404);
  if (request.method !== method) {
    return jsonError("Method not allowed.", 405, { Allow: method });
  }

  let backend;
  try {
    backend = new URL(env?.TENCENT_BACKEND_ORIGIN);
    if (
      backend.protocol !== "https:" ||
      backend.username || backend.password ||
      backend.pathname !== "/" || backend.search || backend.hash ||
      backend.hostname === incoming.hostname
    ) {
      throw new Error("Invalid backend origin");
    }
  } catch {
    return jsonError(
      "Set TENCENT_BACKEND_ORIGIN in Tencent to your Python backend's HTTPS origin (no /api suffix and not this Tencent site), then redeploy.",
      503,
    );
  }

  const target = new URL(incoming.pathname + incoming.search, backend.origin);
  try {
    const upstream = await fetch(target.toString(), {
      method,
      headers: {
        Accept: "application/json",
        ...(method === "POST" ? { "Content-Type": "application/json" } : {}),
      },
      ...(method === "POST" ? { body: await request.arrayBuffer() } : {}),
      redirect: "error",
    });

    const contentType = upstream.headers.get("Content-Type") || "";
    const mediaType = contentType.split(";")[0].trim().toLowerCase();
    if (mediaType !== "application/json" && !mediaType.endsWith("+json")) {
      return jsonError(
        "The backend returned a non-JSON response. Check that TENCENT_BACKEND_ORIGIN points to the Python API service, not a static frontend.",
        502,
      );
    }

    const headers = new Headers({
      "Content-Type": contentType,
      "Cache-Control": "no-store",
    });
    const retryAfter = upstream.headers.get("Retry-After");
    if (retryAfter) headers.set("Retry-After", retryAfter);
    return new Response(upstream.body, { status: upstream.status, headers });
  } catch {
    return jsonError(
      "Unable to reach the Python backend. Check its URL, service status, and Tencent connectivity.",
      502,
    );
  }
}

export default onRequest;
