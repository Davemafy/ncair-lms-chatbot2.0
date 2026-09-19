import test from "node:test";
import assert from "node:assert/strict";

import { handleRequest } from "../src/index.mjs";
import { getPortalLink, getStepGuidance, searchKnowledgeBase } from "../src/knowledge.mjs";

const origin = "https://ncair-lms-chatbotv1.vercel.app";
const envBase = { ALLOWED_ORIGINS: origin };

test("knowledge search returns the attendance rule", () => {
  const result = searchKnowledgeBase("Can I pass with less than 75% attendance?");
  assert.match(result.evidence, /75% attendance/);
  assert.match(result.evidence, /automatic retake/);
});

test("deterministic tools preserve verified URLs and onboarding guidance", () => {
  assert.match(getPortalLink("login").answer, /intern\/signin/);
  assert.match(getStepGuidance(2).answer, /PSIN 50-Seater Hall/);
});

test("health endpoint reports the Cloudflare backend", async () => {
  const response = await handleRequest(
    new Request("https://worker.example/api/health", { headers: { origin } }),
    envBase,
  );
  assert.equal(response.status, 200);
  assert.equal((await response.json()).backend, "cloudflare-workers-ai");
});

test("structured tool selection executes portal tool", async () => {
  const env = {
    ...envBase,
    AI: {
      async run() {
        return {
          choices: [{
            message: {
              tool_calls: [{
                id: "call_1",
                type: "function",
                function: { name: "get_portal_link", arguments: '{"action":"login"}' },
              }],
            },
          }],
        };
      },
    },
  };

  const response = await handleRequest(
    new Request("https://worker.example/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json", origin },
      body: JSON.stringify({ message: "Where do I sign in?" }),
    }),
    env,
  );
  const body = await response.json();
  assert.equal(response.status, 200);
  assert.equal(body.tool, "get_portal_link");
  assert.equal(body.routing, "workers_ai");
  assert.match(body.answer, /intern\/signin/);
});

test("knowledge tool grounds the final Workers AI answer", async () => {
  let call = 0;
  const env = {
    ...envBase,
    AI: {
      async run() {
        call += 1;
        if (call === 1) {
          return {
            choices: [{
              message: {
                tool_calls: [{
                  id: "call_2",
                  type: "function",
                  function: {
                    name: "search_knowledge_base",
                    arguments: '{"query":"attendance requirement"}',
                  },
                }],
              },
            }],
          };
        }
        return {
          choices: [{
            message: {
              content: "You need at least **75% attendance** to pass the cohort.",
            },
          }],
        };
      },
    },
  };

  const response = await handleRequest(
    new Request("https://worker.example/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json", origin },
      body: JSON.stringify({ message: "What attendance do I need?" }),
    }),
    env,
  );
  const body = await response.json();
  assert.equal(body.tool, "search_knowledge_base");
  assert.match(body.answer, /75% attendance/);
  assert.match(body.evidence, /Attendance and grading/);
});

test("unapproved browser origins are rejected", async () => {
  const response = await handleRequest(
    new Request("https://worker.example/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json", origin: "https://evil.example" },
      body: JSON.stringify({ message: "hello" }),
    }),
    { ...envBase, AI: { run: async () => ({}) } },
  );
  assert.equal(response.status, 403);
});
