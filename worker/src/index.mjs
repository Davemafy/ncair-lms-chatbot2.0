import { getPortalLink, getStepGuidance, searchKnowledgeBase } from "./knowledge.mjs";

export const MODEL = "@cf/zai-org/glm-4.7-flash";

const TOOLS = [
  {
    type: "function",
    function: {
      name: "get_portal_link",
      description: "Return a verified NCAIR or LMS URL. Use for requests to open, find or access a portal page.",
      parameters: {
        type: "object",
        properties: {
          action: {
            type: "string",
            enum: ["main", "login", "profile", "courses", "support"],
          },
        },
        required: ["action"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_step_guidance",
      description: "Explain the full onboarding process or one numbered onboarding step.",
      parameters: {
        type: "object",
        properties: {
          step: {
            type: "integer",
            minimum: 0,
            maximum: 4,
            description: "Use 0 for the full sequence, otherwise a step from 1 to 4.",
          },
        },
        required: ["step"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "search_knowledge_base",
      description: "Search the official NCAIR onboarding guide for policies, courses, attendance, SIWES, NYSC, registration and troubleshooting.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", minLength: 1, maxLength: 500 },
        },
        required: ["query"],
        additionalProperties: false,
      },
    },
  },
];

function json(data, status = 200, headers = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      ...headers,
    },
  });
}

function allowedOrigin(request, env) {
  const origin = request.headers.get("origin");
  if (!origin) return "*";
  const configured = String(env.ALLOWED_ORIGINS || "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  return configured.includes(origin) ? origin : null;
}

function corsHeaders(origin) {
  return {
    "access-control-allow-origin": origin || "null",
    "access-control-allow-methods": "POST, GET, OPTIONS",
    "access-control-allow-headers": "Content-Type",
    "access-control-max-age": "86400",
    vary: "Origin",
  };
}

function assistantMessage(result) {
  return result?.choices?.[0]?.message || result?.result?.choices?.[0]?.message || null;
}

function readToolCall(result) {
  const message = assistantMessage(result);
  const rawCall = message?.tool_calls?.[0] || result?.tool_calls?.[0];
  if (!rawCall) return null;
  const name = rawCall.function?.name || rawCall.name;
  const rawArguments = rawCall.function?.arguments ?? rawCall.arguments ?? {};
  let args = rawArguments;
  if (typeof rawArguments === "string") {
    try {
      args = JSON.parse(rawArguments);
    } catch {
      return null;
    }
  }
  return { name, args: args && typeof args === "object" ? args : {} };
}

function validateToolCall(call, userMessage) {
  if (!call) return { name: "search_knowledge_base", args: { query: userMessage }, routing: "safe_fallback" };

  if (
    call.name === "get_portal_link" &&
    ["main", "login", "profile", "courses", "support"].includes(call.args.action)
  ) {
    return { ...call, routing: "workers_ai" };
  }

  if (
    call.name === "get_step_guidance" &&
    Number.isInteger(Number(call.args.step)) &&
    Number(call.args.step) >= 0 &&
    Number(call.args.step) <= 4
  ) {
    return { name: call.name, args: { step: Number(call.args.step) }, routing: "workers_ai" };
  }

  if (call.name === "search_knowledge_base" && typeof call.args.query === "string") {
    const query = call.args.query.trim().slice(0, 500);
    if (query) return { name: call.name, args: { query }, routing: "workers_ai" };
  }

  return { name: "search_knowledge_base", args: { query: userMessage }, routing: "safe_fallback" };
}

async function selectTool(ai, userMessage) {
  const result = await ai.run(MODEL, {
    messages: [
      {
        role: "system",
        content:
          "You route questions for the NCAIR LMS assistant. Always call exactly one provided tool. Never answer directly. Use search_knowledge_base for any factual policy or programme question.",
      },
      { role: "user", content: userMessage },
    ],
    tools: TOOLS,
    tool_choice: "required",
    parallel_tool_calls: false,
    temperature: 0,
    max_completion_tokens: 180,
  });
  return validateToolCall(readToolCall(result), userMessage);
}

async function answerFromEvidence(ai, userMessage, evidence) {
  const result = await ai.run(MODEL, {
    messages: [
      {
        role: "system",
        content:
          "You are the NCAIR LMS Assistant. Answer concisely using only the supplied official evidence. Do not invent dates, URLs, contacts or rules. If the evidence says it is insufficient, say you could not verify the answer and ask the intern to confirm with a facilitator.",
      },
      {
        role: "user",
        content: `Question:\n${userMessage}\n\nOfficial evidence:\n${evidence}`,
      },
    ],
    temperature: 0.1,
    max_completion_tokens: 320,
  });

  const message = assistantMessage(result);
  const answer = message?.content || result?.response || result?.result?.response;
  if (typeof answer !== "string" || !answer.trim()) {
    throw new Error("Workers AI returned an empty answer.");
  }
  return answer.trim();
}

async function runChat(env, userMessage) {
  const selection = await selectTool(env.AI, userMessage);

  if (selection.name === "get_portal_link") {
    const result = getPortalLink(selection.args.action);
    return { ...result, tool: selection.name, routing: selection.routing, model: MODEL };
  }

  if (selection.name === "get_step_guidance") {
    const result = getStepGuidance(selection.args.step);
    return { ...result, tool: selection.name, routing: selection.routing, model: MODEL };
  }

  const result = searchKnowledgeBase(selection.args.query);
  const answer = await answerFromEvidence(env.AI, userMessage, result.evidence);
  return {
    answer,
    tool: selection.name,
    routing: selection.routing,
    evidence: result.sources.join("; "),
    model: MODEL,
  };
}

export async function handleRequest(request, env) {
  const origin = allowedOrigin(request, env);
  const cors = corsHeaders(origin);

  if (request.method === "OPTIONS") {
    return origin ? new Response(null, { status: 204, headers: cors }) : json({ detail: "Origin not allowed." }, 403, cors);
  }

  const url = new URL(request.url);
  if (request.method === "GET" && url.pathname === "/api/health") {
    return json({ status: "ok", backend: "cloudflare-workers-ai", model: MODEL }, 200, cors);
  }

  if (request.method !== "POST" || url.pathname !== "/api/chat") {
    return json({ detail: "Not found." }, 404, cors);
  }

  if (!origin) return json({ detail: "Origin not allowed." }, 403, cors);

  let body;
  try {
    body = await request.json();
  } catch {
    return json({ detail: "Request body must be valid JSON." }, 400, cors);
  }

  const message = typeof body?.message === "string" ? body.message.trim() : "";
  if (!message || message.length > 2000) {
    return json({ detail: "Message must contain between 1 and 2,000 characters." }, 400, cors);
  }

  try {
    return json(await runChat(env, message), 200, cors);
  } catch (error) {
    console.error("chat_failed", error);
    return json(
      { detail: "The AI backend is temporarily unavailable. Please try again shortly." },
      503,
      cors,
    );
  }
}

export default {
  fetch: handleRequest,
};
