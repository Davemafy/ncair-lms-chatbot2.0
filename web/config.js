// Browser requests always stay on the current origin.
// Vercel proxies /api/* to the configured Lightning backend; when FastAPI
// serves this UI directly, the same paths are handled locally.
window.NCAIR_API_URL = "";
