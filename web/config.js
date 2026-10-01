// Use the Lightning V2 API only when this UI is hosted on Vercel.
// When FastAPI serves the UI directly (for example on Lightning), stay same-origin.
const LIGHTNING_V2_API = "https://8000-01m3v6dv3dbzrxn6resd7ms958.cloudspaces.litng.ai";
const EXTERNAL_FRONTENDS = new Set([
  "ncair-lms-chatbotv1.vercel.app",
  "tbotv1.vercel.app",
]);

window.NCAIR_API_URL = EXTERNAL_FRONTENDS.has(window.location.hostname)
  ? LIGHTNING_V2_API
  : "";
