# NCAIR LMS Assistant 2.0

An NCAIR onboarding assistant with a Vercel-hosted web interface and a Cloudflare Worker backend.

## Current architecture

- **Web UI:** static HTML, CSS and JavaScript on Vercel.
- **Primary API:** Cloudflare Worker at `/api/chat`.
- **Model:** Workers AI `@cf/zai-org/glm-4.7-flash`.
- **Routing:** the model must return one structured function call.
- **Tools:** `get_portal_link`, `get_step_guidance`, and `search_knowledge_base`.
- **Grounding:** answers are generated only from the bundled official NCAIR guide.
- **Resilience:** the web UI retains a small read-only preview fallback if the Worker is unavailable.

The original FastAPI/Ollama implementation remains in `src/` as the preserved v1 reference. It is no longer required by the Cloudflare deployment.

## Deploy the Cloudflare backend

You need a free Cloudflare account, Node.js and npm. You do not need an OpenAI, Groq or Gemini key.

```bash
cd worker
npm install
npx wrangler login
npm test
npm run deploy
```

Wrangler prints a URL similar to:

```text
https://ncair-lms-chatbot-api.<your-subdomain>.workers.dev
```

Copy that origin into `web/config.js`:

```js
window.NCAIR_API_URL = "https://ncair-lms-chatbot-api.<your-subdomain>.workers.dev";
```

Commit and push that one-line change. Vercel will automatically redeploy the connected `main` branch.

## Verify

```bash
curl https://ncair-lms-chatbot-api.<your-subdomain>.workers.dev/api/health
```

Expected response:

```json
{
  "status": "ok",
  "backend": "cloudflare-workers-ai",
  "model": "@cf/zai-org/glm-4.7-flash"
}
```

Then open the Vercel UI and ask:

- “Where do I sign in?”
- “Can I pass if my attendance is below 75%?”
- “I cannot find my invitation email.”

The response chip should show the selected tool rather than `preview mode`.

## Local Worker tests

The unit tests mock Workers AI, so they run without a Cloudflare account:

```bash
cd worker
npm test
```

## Legacy FastAPI/Ollama v1

```bash
ollama pull llama3.2:3b
ollama serve
pip install -r requirements.txt
uvicorn src.api:app --reload
```

Open `http://localhost:8000`.
